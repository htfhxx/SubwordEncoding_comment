# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-15 14:11:08
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-01-18 22:29:08

import time
import sys
import argparse
import random
import copy
import torch
import gc
import pickle as pickle
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from utils.metric import get_ner_fmeasure
from model.bilstmcrf import BiLSTM_CRF as SeqModel
from utils.data import Data

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


#调用来自：data_initialization(data, gaz_file, train_file, dev_file, test_file)
##data.word_alphabet  biword_alphabet   char_alphabet  label_alphabet   训练测试验证集的字、字+下个字、字、字的标签
##data.gaz的ent2type和ent2id：gaz_file的word type id
##data.gaz_alphabet：所有的子词
seed_num = 100
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)



#调用来自：data_initialization(data, gaz_file, train_file, dev_file, test_file)
##data.word_alphabet  biword_alphabet   char_alphabet  label_alphabet   训练测试验证集的字、字+下个字、字、字的标签
##data.gaz的ent2type和ent2id：gaz_file的word type id
##data.gaz_alphabet：所有的子词



def data_initialization(data, gaz_file, train_file, dev_file, test_file):
	#将训练集验证集等放入data.word_alphabet\biword_alphabet\char_alphabet\label_alphabet中
    data.build_alphabet(train_file)  
    data.build_alphabet(dev_file)
    data.build_alphabet(test_file)
	
	#把gaz_file的词加入到data.gaz的ent2type和ent2id中  ent2id
    data.build_gaz_file(gaz_file)
	
	#data.gaz_alphabet中 放入： 训练集\测试集等等 每句话中在gaz_file中能匹配到的子词
    data.build_gaz_alphabet(train_file)
    data.build_gaz_alphabet(dev_file)
    data.build_gaz_alphabet(test_file)
	
	#keep_growing=False
    data.fix_alphabet()
    return data

#right, whole = predict_check(tag_seq, batch_label, mask)
def predict_check(pred_variable, gold_variable, mask_variable):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()
    # print("right: %s, total: %s"%(right_token, total_token))
    return right_token, total_token


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    
    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = gold_variable.size(0)
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        # print "p:",pred, pred_tag.tolist()
        # print "g:", gold, gold_tag.tolist()
        assert(len(pred)==len(gold))
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label

#调用来自：save_data_setting(data, save_data_name)
def save_data_setting(data, save_file):
    new_data = copy.deepcopy(data)  #复制data到new_data
    ## remove input instances
    new_data.train_texts = []
    new_data.dev_texts = []
    new_data.test_texts = []
    new_data.raw_texts = []

    new_data.train_Ids = []
    new_data.dev_Ids = []
    new_data.test_Ids = []
    new_data.raw_Ids = []
    ## save data settings



    with open(save_file, 'wb') as fp:
        pickle.dump(new_data, fp)

    print("Data setting saved to file: ", save_file)
    #with open(save_file, 'wb') as fp:
    #    pickle.dump(new_data, fp)   #wb is bytes
    #print( "Data setting saved to file: ", save_file)


def load_data_setting(save_file):
    with open(save_file, 'r') as fp:
        data = pickle.load(fp)
   # print ("Data setting loaded from file: ", save_file)
    data.show_data_summary()
    return data


#调整学习率
#from:optimizer = lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr) 
#									decay_rate：HP_lr_decay=0.05		HP_lr=0.01 
def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr * ((1-decay_rate)**epoch)
    #print( " Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr   # 得到学习率：optimizer.param_groups[0]["lr"] 
    return optimizer



def evaluate(data, model, name):
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    elif name == 'raw':
        instances = data.raw_Ids
    else:
        print( "Error: wrong evaluate name,", name)
    right_token = 0
    whole_token = 0
    pred_results = []
    gold_results = []
    ## set model in eval model
    model.eval()
    batch_size = 1
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num//batch_size+1
    for batch_id in range(total_batch):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size 
        if end >train_num:
            end =  train_num
        instance = instances[start:end]
        if not instance:
            continue
        gaz_list,batch_word, batch_biword, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask  = batchify_with_label(instance, data.HP_gpu, True)
        tag_seq = model(gaz_list,batch_word, batch_biword, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask)
        # print ("tag:",tag_seq)
        pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover)
        pred_results += pred_label
        gold_results += gold_label
    decode_time = time.time() - start_time
    speed = len(instances)/decode_time
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
    return speed, acc, p, r, f, pred_results  

# gaz_list,  batch_word, batch_biword, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask = batchify_with_label(instance, data.HP_gpu)  instance=train_ids[.]
def batchify_with_label(input_batch_list, gpu, volatile_flag=False):   #input_batch_list = data.train_Ids[start:end]
    """
        input: list of words, chars and labels, various length. [[words,biwords,chars,gaz, labels],[words,biwords,chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len) 
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order 
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len) 
    """
    f=open('data/bug_batch.txt','a+')

    #从input_batch_list抽取出来words、biwords、chars、 gazs、 labels
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    biwords = [sent[1] for sent in input_batch_list]
    chars = [sent[2] for sent in input_batch_list]
    gazs = [sent[3] for sent in input_batch_list]
    labels = [sent[4] for sent in input_batch_list]
    print("words: ",words   ,file=f)
    print( "biwords: ",biwords   ,file=f)
    print("chars: ",chars   ,file=f)
    print("gazs: ",gazs   ,file=f)
    print("labels:",labels   ,file=f)

    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    print("word_seq_lengths",word_seq_lengths   ,file=f)
    max_seq_len = word_seq_lengths.max()  #句子最大长度
    print("max_seq_len: ",max_seq_len   ,file=f)

	#包装一个Tensor，并记录用在它身上的operations
    word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile =  volatile_flag).long()
    print("word_seq_tensor: ",word_seq_tensor   ,file=f)
    biword_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile =  volatile_flag).long()
    print("biword_seq_tensor: ",biword_seq_tensor   ,file=f)
    label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)),volatile =  volatile_flag).long()
    print("label_seq_tensor: ",label_seq_tensor   ,file=f)

    mask = autograd.Variable(torch.zeros((batch_size, max_seq_len)),volatile =  volatile_flag).byte()
    print("mask: ",mask   ,file=f)


	#初始化的tensor加入数值
    for idx, (seq, biseq, label, seqlen) in enumerate(zip(words, biwords, labels, word_seq_lengths)):
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        biword_seq_tensor[idx, :seqlen] = torch.LongTensor(biseq)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        #mask[idx, :seqlen] = torch.Tensor([1] * seqlen)
        for i in range(seqlen):
            mask[idx, :seqlen] = torch.Tensor([1])
    print("word_seq_tensor2: ",word_seq_tensor   ,file=f)
    print("biword_seq_tensor2: ", biword_seq_tensor   ,file=f)
    print("label_seq_tensor2: ", label_seq_tensor   ,file=f)
    print("mask2: ", mask   ,file=f)

	#返回元组 (sorted_tensor, sorted_indices). 	sorted_indices 为原始输入中的下标
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)  #对输入张量input沿着指定维按降序排序
    print("word_seq_tensor3: ", word_seq_tensor   ,file=f)

	#得到排序后的word_seq_tensor、biword_seq_tensor、label_seq_tensor      mask未排序
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    biword_seq_tensor = biword_seq_tensor[word_perm_idx]
    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]   ## not reorder label
    print("word_seq_tensor3: ",word_seq_tensor   ,file=f)
    print("biword_seq_tensor3: ", biword_seq_tensor   ,file=f)
    print("label_seq_tensor3: ", label_seq_tensor   ,file=f)
    print("mask3: ", mask   ,file=f)

    ### deal with char
    # #用0补全，与最长句子对齐 (batch_size, max_seq_len)
    #pad_chars = [chars[idx] + [[0]] * (max_seq_len-len(chars[idx])) for idx in range(len(chars))]
    pad_chars = [torch.tensor(chars[idx]) + torch.tensor([[0]]) * (max_seq_len - len(chars[idx])) for idx in range(len(chars))]
    print("pad_chars: ",pad_chars   ,file=f)
    #length_list = [map(len, pad_char) for pad_char in pad_chars]  #pad_chars的长度list
    length_list = [pad_char.size() for pad_char in pad_chars]  #pad_chars的长度list
    print("length_list:",length_list   ,file=f)
    max_word_len = max(map(max, length_list))
    print("max_word_len: ",max_word_len   ,file=f)

    char_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len, max_word_len)), volatile =  volatile_flag).long()
    print("char_seq_tensor:", char_seq_tensor   ,file=f)

    # pad_chars的长度list
    char_seq_lengths = torch.LongTensor(length_list)
    print("char_seq_lengths:", char_seq_lengths   ,file=f)

    #得到char_seq_tensor
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            # print len(word), wordlen
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)
    print("char_seq_tensor2:", char_seq_tensor   ,file=f)
    #降维，排序
    #char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size*max_seq_len,-1)
    #char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size*max_seq_len,)
    #char_seq_tensor = char_seq_tensor[word_perm_idx].view ( -1,int(max_word_len) )
    #char_seq_lengths = char_seq_lengths[word_perm_idx].view(-1,1 )


    char_seq_tensor = char_seq_tensor[word_perm_idx].view ( -1,int(max_word_len) )
    print("char_seq_tensor3:",char_seq_tensor   ,file=f)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(-1, 1)
    print("char_seq_lengths3:",char_seq_lengths   ,file=f)

    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    print( "char_perm_idx:",char_perm_idx   ,file=f)

    char_seq_tensor = char_seq_tensor[char_perm_idx]            #!!!!!!!!
    print("char_seq_tensor4:", char_seq_tensor   ,file=f)
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)

    _, word_seq_recover = word_perm_idx.sort(0, descending=False)   #word_seq_recover为原始输入中的下标
    
    ## keep the gaz_list in orignial order
    
    gaz_list = [ gazs[i] for i in word_perm_idx]
    gaz_list.append(volatile_flag)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        biword_seq_tensor = biword_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        mask = mask.cuda()
    return gaz_list, word_seq_tensor, biword_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask




#调用来自：train(data, save_model_dir, seg)
def train(data, save_model_dir, seg=True):

    print('start train')
    print("Training model...")
    data.show_data_summary()  # 打印data信息
    save_data_name = save_model_dir + ".dset"
    save_data_setting(data, save_data_name)  # 保存数据：pickle.dump(new_data, fp)

    model = SeqModel(data)  # ！！！！！！！！！！！！！！！！！！
    print( "finished built model.")



    loss_function = nn.NLLLoss()
	#筛选出需要需要梯度的变量
    parameters = filter(lambda p: p.requires_grad, model.parameters())   #filter(func, seq)    提取出seq中能使func为true的元素序列
    optimizer = optim.SGD(parameters, lr=data.HP_lr, momentum=data.HP_momentum) #动量因子
    best_dev = -1
    ## start training
    for idx in range(data.HP_iteration):   #HP_iteration=50
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" %(idx,data.HP_iteration) )


		#调整学习率
        optimizer = lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)

        instance_count = 0
        sample_id = 0
        sample_loss = 0
        batch_loss = 0
        total_loss = 0
        right_token = 0
        whole_token = 0
        random.shuffle(data.train_Ids)
		
        ## set model in train model
        model.train()  #将module设置为 training mode。
        model.zero_grad()  #将 module 中的所有模型参数的梯度设置为0
        batch_size = data.HP_batch_size ## current only support batch size = 1 to compulate and accumulate to data.HP_batch_size update weights
        batch_id = 0
        train_num = len(data.train_Ids)
        print("train_num:",train_num)
        total_batch = train_num//batch_size+1
        print("total_batch:",total_batch)
        for batch_id in range(total_batch):
            start = batch_id*batch_size
            end = (batch_id+1)*batch_size 
            if end >train_num:
                end = train_num
            instance = data.train_Ids[start:end]



            if not instance:
                continue


            f=open('data/bug_data.txt','a+')
            print( "-----------------------------------------------------------------------------------------------------------",file=f)
            print("end:  ",end    ,file=f )
            print("data.train_texts:",data.train_texts[start:end]   ,file=f)
            gaz_list,  batch_word, batch_biword, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask  = batchify_with_label(instance, data.HP_gpu)

            print("-----------------------------------------------------------------------------------------------------------",file=f)
            '''
        
        
            print("--------------------------------------------------------------------------------")
            print( "gaz_list:",gaz_list )
            print("batch_word:",batch_word )
            print("batch_biword:",batch_biword)
            print("batch_wordlen:",batch_wordlen)
            print("batch_wordrecover:",batch_wordrecover)
            print("batch_char:",batch_char)
            print("batch_charlen:",batch_charlen)
            print("batch_charrecover:",batch_charrecover)
            print("batch_label:",batch_label)
            print("mask:",mask)
            print("--------------------------------------------------------------------------------")
            '''
            instance_count += 1
            loss, tag_seq = model.neg_log_likelihood_loss(gaz_list, batch_word, batch_biword, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label, mask)

            right, whole = predict_check(tag_seq, batch_label, mask)
            right_token += right
            whole_token += whole
            sample_loss += loss.data[0]
            total_loss += loss.data[0]
            batch_loss += loss

            if end%5 == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f"%(end, temp_cost, sample_loss, right_token, whole_token,(right_token+0.)/whole_token))
                sys.stdout.flush()
                sample_loss = 0
            if end%data.HP_batch_size == 0:
                batch_loss.backward()
                optimizer.step()
                model.zero_grad()
                batch_loss = 0

        temp_time = time.time()
        temp_cost = temp_time - temp_start
        print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f"%(end, temp_cost, sample_loss, right_token, whole_token,(right_token+0.)/whole_token))       
        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s"%(idx, epoch_cost, train_num/epoch_cost, total_loss))
        # exit(0)
        # continue
        #验证集的预测和评估
        speed, acc, p, r, f, _ = evaluate(data, model, "dev")

        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish
        if seg:
            current_score = f
            print("Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(dev_cost, speed, acc, p, r, f))
        else:
            current_score = acc
            print("Dev: time: %.2fs speed: %.2fst/s; acc: %.4f"%(dev_cost, speed, acc))
        if current_score > best_dev:
            if seg:
                print ("Exceed previous best f score:", best_dev)
            else:
                print( "Exceed previous best acc score:", best_dev)
            model_name = save_model_dir +'.'+ str(idx) + ".model"
            torch.save(model.state_dict(), model_name)
            best_dev = current_score

        # ## 测试集的预测和评估
        speed, acc, p, r, f, _ = evaluate(data, model, "test")
        test_finish = time.time()
        test_cost = test_finish - dev_finish
        if seg:
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(test_cost, speed, acc, p, r, f))
        else:
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f"%(test_cost, speed, acc))
        gc.collect() 



def load_model_decode(model_dir, data, name, gpu, seg=True):
    data.HP_gpu = gpu
    print("Load Model from file: ", model_dir)
    model = SeqModel(data)
    ## load model need consider if the model trained in GPU and load in CPU, or vice versa
    # if not gpu:
    #     model.load_state_dict(torch.load(model_dir), map_location=lambda storage, loc: storage)
    #     # model = torch.load(model_dir, map_location=lambda storage, loc: storage)
    # else:
    model.load_state_dict(torch.load(model_dir))
        # model = torch.load(model_dir)
    
    print("Decode %s data ..."%(name))
    start_time = time.time()
    speed, acc, p, r, f, pred_results = evaluate(data, model, name)
    end_time = time.time()
    time_cost = end_time - start_time
    if seg:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(name, time_cost, speed, acc, p, r, f))
    else:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f"%(name, time_cost, speed, acc))
    return pred_results




if __name__ == '__main__':

    '''
    parser = argparse.ArgumentParser(description='Subword Encoding in Lattice LSTM for Chinese Word SegmentationF')
    parser.add_argument('--embedding',  help='Embedding for words', default='None')
    parser.add_argument('--status', choices=['train', 'test', 'decode'], help='update algorithm', default='train')
    parser.add_argument('--savemodel', default="data/model/saved_model.lstmcrf.")
    parser.add_argument('--savedset', help='Dir of saved data setting', default="data/save.dset")

    parser.add_argument('--train', default="data/bala_train")
    parser.add_argument('--dev', default="data/bala_dev" )
    parser.add_argument('--test', default="data/bala_test")

    parser.add_argument('--seg', default="True") 
    parser.add_argument('--extendalphabet', default="True") 
    parser.add_argument('--raw') 
    parser.add_argument('--loadmodel')
    parser.add_argument('--output') 
    args = parser.parse_args() 
    
    train_file = args.train
    dev_file = args.dev
    test_file = args.test
    raw_file = args.raw
    model_dir = args.loadmodel
    dset_dir = args.savedset
    output_file = args.output
    if args.seg.lower() == "true":
        seg = True 
    else:
        seg = False
    status = args.status.lower()
    
    '''

    torch.cuda.set_device(1)
    gpu = True  # torch.cuda.is_available()  #确定系统是否支持CUDA

    savemodel ="data/model/saved_model.lstmcrf."

    #'''
    train_file = "data/pku_train"#"data/bala_train"
    dev_file = "data/pku_dev"
    test_file = "data/pku_test"
    
    '''
    train_file = "data/bala_train"#"data/bala_train"
    dev_file = "data/bala_dev"
    test_file = "data/bala_test"
    
    '''


    raw_file = 'data/raw_file'    #????????????????????
    model_dir = "model_path/"
    dset_dir = "data/save.dset"
    output_file = 'data/output_file'
    status='train'
    seg="True"
    if seg.lower() == "true":
        seg = True 
    else:
        seg = False
    status = status.lower()



    save_model_dir = savemodel




    '''
    char_emb = "../SubwordEncoding_download_data/gigaword_chn.all.a2b.uni.ite50.vec"
    bichar_emb = "../SubwordEncoding_download_data/gigaword_chn.all.a2b.bi.ite50.vec"
    #gaz_file = "../../data/ctb.50d.vec"
    gaz_file = "../SubwordEncoding_download_data/zh.wiki.bpe.vs200000.d50.w2v.txt"    
    '''

    char_emb = "../2/gigaword_chn.all.a2b.uni.ite50.vec"
    bichar_emb = "../2/gigaword_chn.all.a2b.bi.ite50.vec"
    gaz_file = "../../data/ctb.50d.vec"
    gaz_file = "../2/zh.wiki.bpe.vs200000.d50.w2v.txt"
    #'''


    print( "CuDNN:", torch.backends.cudnn.enabled )   #cuDNN用的是非确定算法,torch.backends.cudnn.enabled=False禁用cuDNN
    #gpu = False
    print( "GPU available:", gpu)
    print ("Status:", status)
    print( "Seg: ", seg)
    print( "Train file:", train_file)
    print( "Dev file:", dev_file)
    print( "Test file:", test_file)
    print ("Raw file:", raw_file)
    print( "Char emb:", char_emb)
    print( "Bichar emb:", bichar_emb)
    print( "Gaz file:",gaz_file)
    if status == 'train':
        print( "Model saved to:", save_model_dir)
    sys.stdout.flush()



    if status == 'train':
        data = Data()
        data.HP_gpu = gpu
        data.HP_use_char = False
        data.HP_batch_size = 1    ##############################
        data.use_bigram = True
        data.gaz_dropout = 0.5
        data.HP_lr = 0.01
        data.HP_dropout = 0.5
        data.HP_iteration = 50
        data.norm_gaz_emb = True
        data.HP_fix_gaz_emb = False
		
		#data.word_alphabet  biword_alphabet   char_alphabet  label_alphabet   训练测试验证集的字、字+下个字、字、字的标签
		##data.gaz的ent2type和ent2id：gaz_file的word type id
		##data.gaz_alphabet：所有的子词
        data_initialization(data, gaz_file, train_file, dev_file, test_file)

		#data.train_texts：[words, biwords, chars, gazs, labels])
		#data.train_Ids：[word_Ids, biword_Ids, char_Ids, gaz_Ids, label_Ids]
        data.generate_instance_with_gaz(train_file,'train')
        data.generate_instance_with_gaz(dev_file,'dev')
        data.generate_instance_with_gaz(test_file,'test')

        # 得到的data.train_texts的维度：np.array(data.train_texts).shape   (15L, 5L)
        # 意思是 15个句子  5个维度[words, biwords, chars, gazs, labels]

        # 字的embedding：data.pretrain_word_embedding	  data.word_emb_dim
        data.build_word_pretrain_emb(char_emb)

        #biword的embedding：data.pretrain_biword_embedding	   data.biword_emb_dim
        data.build_biword_pretrain_emb(bichar_emb)

        # 子词的embedding：data.pretrain_gaz_embedding, data.gaz_emb_dim
        data.build_gaz_pretrain_emb(gaz_file)

        a=['迈', '向', '充', '满', '希', '望', '的', '新', '世', '纪', '—', '—', '一', '九', '九', '八', '年', '新', '年', '讲', '话', '(', '附', '图', '片', '0', '张', ')']
        b=['中', '共', '中', '央', '总', '书', '记', '、', '国', '家', '主', '席', '江', '泽', '民']
        list_id=[]
        for char_index in range(len(b)):
            list_id.append( data.word_alphabet.get_index(b[char_index]))
        print("list_id:  ", list_id)
        print(" ")




        #train(data, save_model_dir, seg)

    elif status == 'test':      
        data = load_data_setting(dset_dir)
        data.generate_instance_with_gaz(dev_file,'dev')
        load_model_decode(model_dir, data , 'dev', gpu, seg)
        data.generate_instance_with_gaz(test_file,'test')
        load_model_decode(model_dir, data, 'test', gpu, seg)
    elif status == 'decode':       
        data = load_data_setting(dset_dir)
        data.generate_instance_with_gaz(raw_file,'raw')
        decode_results = load_model_decode(model_dir, data, 'raw', gpu, seg)
        data.write_decoded_results(output_file, decode_results, 'raw')
    else:
        print( "Invalid argument! Please use valid arguments! (train/test/decode)" )




