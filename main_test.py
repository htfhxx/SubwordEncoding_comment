
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



if __name__ == '__main__':
    input_batch_list= [ [[277, 650, 414, 479, 13, 43, 14, 45, 651, 26, 33, 183, 13, 246, 390, 876, 813, 197, 1243, 474, 368, 26, 33, 390, 876, 813, 197, 452, 143, 1305, 95, 1243, 474, 368, 46, 540, 701, 69, 200, 545, 866, 1463, 178, 374, 51, 1243, 474, 368, 8, 183, 13, 246, 362, 78, 414, 479, 41, 44, 1932, 362, 78, 540, 701, 444, 59, 247, 668, 1045, 178, 374, 51], [2126, 2127, 1201, 14259, 13757, 94165, 69993, 2128, 3462, 117, 34225, 1424, 444, 25710, 7529, 15037, 7994, 123351, 6905, 6906, 24883, 117, 76639, 7529, 15037, 7994, 123352, 16951, 123353, 25680, 123354, 6905, 6906, 13787, 4245, 2697, 3873, 332, 5974, 123355, 46141, 62255, 7338, 6820, 68099, 6905, 6906, 17819, 1298, 1424, 444, 60093, 21254, 123356, 1201, 2684, 6474, 96590, 123357, 21254, 123358, 2697, 72637, 85849, 1758, 123359, 5506, 123360, 7338, 6820, 331], [[277], [650], [414], [479], [13], [43], [14], [45], [651], [26], [33], [183], [13], [246], [390], [876], [813], [197], [1243], [474], [368], [26], [33], [390], [876], [813], [197], [452], [143], [1305], [95], [1243], [474], [368], [46], [540], [701], [69], [200], [545], [866], [1463], [178], [374], [51], [1243], [474], [368], [8], [183], [13], [246], [362], [78], [414], [479], [41], [44], [1932], [362], [78], [540], [701], [444], [59], [247], [668], [1045], [178], [374], [51]], [[], [], [[770], [2]], [], [[6432], [2]], [[29881], [2]], [[24543], [2]], [], [], [[35783, 20636, 74], [5, 4, 2]], [], [[6901, 873], [3, 2]], [[295], [2]], [], [[3746], [2]], [[6918], [2]], [[4021], [2]], [], [[3395, 3396], [3, 2]], [], [[10527], [2]], [[74], [2]], [], [[3746], [2]], [[6918], [2]], [[4021], [2]], [], [[7615], [2]], [], [[10823], [2]], [], [[3395, 3396], [3, 2]], [], [], [], [[1482], [2]], [], [[3010, 228], [3, 2]], [[2976], [2]], [], [[35784, 17868], [4, 2]], [], [[3650], [2]], [], [], [[11187, 3395, 3396], [4, 3, 2]], [], [[7950], [2]], [[6900, 871, 813], [4, 3, 2]], [[6901, 873], [3, 2]], [[295], [2]], [], [[9351], [2]], [], [[770], [2]], [], [[30399, 3200], [3, 2]], [], [], [[9351], [2]], [], [[1482], [2]], [], [[35785, 28273], [3, 2]], [[1052], [2]], [], [[35786, 2779], [4, 2]], [], [[3650], [2]], [], []], [1, 2, 1, 2, 1, 2, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 1, 2, 1, 4, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 4, 2, 3, 1, 2, 3, 1, 2, 1, 2, 1, 2, 3, 1, 4, 2, 3, 1, 2, 3, 1, 2, 1, 2, 1, 4, 2, 1, 2, 1, 2, 3, 1, 2, 1, 2, 1, 2, 3] ]]


    volatile_flag=False

    f=open('data/batch_data_test.txt','a+')

    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    biwords = [sent[1] for sent in input_batch_list]
    chars = [sent[2] for sent in input_batch_list]
    gazs = [sent[3] for sent in input_batch_list]
    labels = [sent[4] for sent in input_batch_list]
    print("words: ", words, file=f)
    print("biwords: ", biwords, file=f)
    print("chars: ", chars, file=f)
    print("gazs: ", gazs, file=f)
    print("labels:", labels, file=f)

    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    print("word_seq_lengths", word_seq_lengths, file=f)
    max_seq_len = word_seq_lengths.max()  # 句子最大长度
    print("max_seq_len: ", max_seq_len, file=f)

    # 包装一个Tensor，并记录用在它身上的operations
    word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).long()
    print("word_seq_tensor: ", word_seq_tensor, file=f)
    biword_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).long()
    print("biword_seq_tensor: ", biword_seq_tensor, file=f)
    label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).long()
    print("label_seq_tensor: ", label_seq_tensor, file=f)

    mask = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).byte()
    print("mask: ", mask, file=f)

    # 初始化的tensor加入数值
    for idx, (seq, biseq, label, seqlen) in enumerate(zip(words, biwords, labels, word_seq_lengths)):
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        biword_seq_tensor[idx, :seqlen] = torch.LongTensor(biseq)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        # mask[idx, :seqlen] = torch.Tensor([1] * seqlen)
        # for i in range(seqlen):
        #   mask[idx, :seqlen] = torch.Tensor([1])
        mask[idx, :seqlen] = torch.Tensor([1])
    print("word_seq_tensor2: ", word_seq_tensor, file=f)
    print("biword_seq_tensor2: ", biword_seq_tensor, file=f)
    print("label_seq_tensor2: ", label_seq_tensor, file=f)
    print("mask2: ", mask, file=f)

    # 返回元组 (sorted_tensor, sorted_indices). 	sorted_indices 为原始输入中的下标
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)  # 对输入张量input沿着指定维按降序排序
    print("word_seq_tensor3: ", word_seq_tensor, file=f)

    # 得到排序后的word_seq_tensor、biword_seq_tensor、label_seq_tensor      mask未排序
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    biword_seq_tensor = biword_seq_tensor[word_perm_idx]
    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]  ## not reorder label
    print("word_seq_tensor3: ", word_seq_tensor, file=f)
    print("biword_seq_tensor3: ", biword_seq_tensor, file=f)
    print("label_seq_tensor3: ", label_seq_tensor, file=f)
    print("mask3: ", mask, file=f)

    ### deal with char
    # #用0补全，与最长句子对齐 (batch_size, max_seq_len)
    # pad_chars = [chars[idx] + [[0]] * (max_seq_len-len(chars[idx])) for idx in range(len(chars))]
    pad_chars = [torch.tensor(chars[idx]) + torch.tensor([[0]]) * (max_seq_len - len(chars[idx])) for idx in range(len(chars))]
    print("pad_chars: ", pad_chars, file=f)
    # length_list = [map(len, pad_char) for pad_char in pad_chars]  #pad_chars的长度list
    # length_list = [len(pad_char) for pad_char in pad_chars]  #pad_chars的长度list
    length_list = [len(pad_char) for pad_char in pad_chars]




    print("length_list:", length_list,file=f)

    max_word_len = max(length_list)
    print("max_word_len: ", max_word_len,file=f)
    #print("max_word_len: ", max_word_len)









    char_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len, max_word_len)),volatile=volatile_flag).long()
    print("char_seq_tensor:", char_seq_tensor, file=f)
    #print("char_seq_tensor:", char_seq_tensor)
    print( "char_seq_tensor.size():", char_seq_tensor.size(), file=f)
    #print("char_seq_tensor:", char_seq_tensor)
    #print("char_seq_tensor.size():", char_seq_tensor.size())

    # pad_chars的长度list
    char_seq_lengths = torch.LongTensor(length_list)
    print("char_seq_lengths:", char_seq_lengths, file=f)
    print("char_seq_lengths:", char_seq_lengths)

    # 得到char_seq_tensor
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):

        #for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
        for idy, (word) in enumerate(zip(seq)):
            wordlen=seqlen.item()
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)
    print("char_seq_tensor2:", char_seq_tensor, file=f)
    print("char_seq_tensor2:", char_seq_tensor)
    # 降维，排序
    # char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size*max_seq_len,-1)
    # char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size*max_seq_len,)
    # char_seq_tensor = char_seq_tensor[word_perm_idx].view ( -1,int(max_word_len) )
    # char_seq_lengths = char_seq_lengths[word_perm_idx].view(-1,1 )




    char_seq_tensor = char_seq_tensor[word_perm_idx].view(-1, int(max_word_len))
    print("char_seq_tensor3:", char_seq_tensor, file=f)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(-1, 1)
    print("char_seq_lengths3:", char_seq_lengths, file=f)

    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    print("char_perm_idx:", char_perm_idx, file=f)

    char_seq_tensor = char_seq_tensor[char_perm_idx]  # !!!!!!!!
    print("char_seq_tensor4:", char_seq_tensor, file=f)
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)

    _, word_seq_recover = word_perm_idx.sort(0, descending=False)  # word_seq_recover为原始输入中的下标

    ## keep the gaz_list in orignial order

    gaz_list = [gazs[i] for i in word_perm_idx]
    gaz_list.append(volatile_flag)


'''

'''
















