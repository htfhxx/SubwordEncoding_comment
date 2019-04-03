# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import codecs
import os
import re
DATA_PATH='data'

#MSR
#TRAIN_MSR='data/original_data/msr_training.utf8'
#TEST_MSR='data/original_data/msr_test_gold.utf8'

#PKU
TRAIN_MSR='data/original_data/pku_training.utf8'
TEST_MSR='data/original_data/pku_test_gold.utf8'

#BALA(test)
#TRAIN_BALA='data/original_data/bala_training.utf8'
#TEST_BALA='data/original_data/bala_test_gold.utf8'

CHINESE_IDIOMS='data/original_data/idioms'

rNUM = '(-|\+)?\d+((\.|·)\d+)?%?'
rENG = '[A-Za-z_.]+'

def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)#返回对应的 ASCII 数值，或者 Unicode 数值
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        #rstring += unichr(inside_code)
        rstring += chr(inside_code)
    return rstring

#全角转半角、成语替换为'I',数字替换为0,字母替换为'X'
def preprocess(input,output):
    output_filename = os.path.join(DATA_PATH,output)

    #取出成语词典中的成语
    idioms=dict()
    with codecs.open(CHINESE_IDIOMS,'r','utf-8') as f:    
        for line in f:  
            idioms[line.strip()]=1    #返回移除字符串头尾指定的字符(空格)生成的新字符串                              标记为1
    
    count_idioms = 0
    sents=[]
    #pku_training.utf8的文本处理：
    with codecs.open(input,'r','utf-8') as fin:
        with codecs.open(output_filename,'w','utf-8') as fout:
            for line in fin:   #取pku_training 的每一行
                sent=strQ2B(line).split( ) #此行 词 的集合
                new_sent=[]
                for word in sent:
                    word=re.sub(rNUM,'0',word)  
                    word=re.sub(rENG,'X',word)
                    if idioms.get(word) is not None:
                        count_idioms+=1
                        word=u'I'
                    new_sent.append(word)
                sents.append(new_sent)
            for sent in sents:
                fout.write('  '.join(sent))
                fout.write('\n')
    print( 'replaced idioms count:%d' % count_idioms)


def split(dataset): #划分出验证集
    dataset=os.path.join(DATA_PATH,dataset)
    print('split '+dataset )
    with codecs.open(dataset+'_train_all','r','utf-8') as f:
        lines = f.readlines()
        idx = int(len(lines)*0.9)
        with codecs.open(dataset+'_train_tmp','wb','utf-8') as fo:
            for line in lines[:idx]:
                fo.write(line.strip()+'\r')
        with codecs.open(dataset+'_dev_tmp','wb','utf-8') as fo:
            for line in lines[idx:]:
                fo.write(line.strip()+'\r')
    os.remove(dataset+'_train_all')

def word2tag(word):
    if len(word)==1:
        return ['S-SEG']
    if len(word)==2:
        return ['B-SEG','E-SEG']
    tag=[]
    tag.append('B-SEG')
    for i in range(1,len(word)-1):
        tag.append('M-SEG')
    tag.append('E-SEG')
    return tag

def sentence_2_word(filename=None):
    filename=os.path.join(DATA_PATH,filename)
    x,y=[],[]
    with codecs.open(filename+'_tmp','r','utf-8') as f:
        with codecs.open(filename,'wb','utf-8') as fo:
            for line in f:
                word_list=line.strip().split()  #每行单词存到list中
                for word in word_list:
                    tag=word2tag(word)
                    for i in range(len(tag)):
                        fo.write(word[i]+' '+tag[i]+'\n')
                fo.write('\n')
    f.close()
    os.remove(filename+'_tmp')
				

if __name__ == '__main__':
    '''
    bala
	#预处理
    preprocess(TRAIN_BALA,'bala_train_all') 
    preprocess(TEST_BALA,'bala_test_tmp')
	#划分验证集
    split('bala')             
    #处理格式
    sentence_2_word('bala_train')		 
    sentence_2_word('bala_dev')		 
    sentence_2_word('bala_test')	
	'''

    '''
    msr
    #预处理
    preprocess(TRAIN_MSR,'msr_train_all') 
    preprocess(TEST_MSR,'msr_test_tmp')
	#划分验证集
    split('msr')             
    #处理格式
    sentence_2_word('msr_train')		 
    sentence_2_word('msr_dev')		 
    sentence_2_word('msr_test')	
    
    '''
    #pku
    #预处理
    preprocess(TRAIN_MSR,'pku_train_all')
    preprocess(TEST_MSR,'pku_test_tmp')
	#划分验证集
    split('pku')
    #处理格式
    sentence_2_word('pku_train')
    sentence_2_word('pku_dev')
    sentence_2_word('pku_test')
	







