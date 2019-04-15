import time
import os
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

import codecs

from utils.metric import get_ner_fmeasure
from model.bilstmcrf import BiLSTM_CRF as SeqModel
from utils.data import Data


if __name__ == '__main__':

    data=Data()
    data.word_alphabet.instance2index={23:'鬼',16:'王',48:'宗'}

    batch_word_id=[[  23,   16,   48]]

    batch_word=[[]]
    for i in batch_word_id[0]:
        batch_word[0].append(data.word_alphabet.instance2index[i])

    print(batch_word)


    print(sys.argv)



    # domain_word_path='data/dictionary_zx.txt'
    #
    # #存入词典
    # domain_word_lists = dict()
    # with codecs.open(domain_word_path,'r','utf-8') as f:
    #     for line in f:
    #         line=line.strip().split()[0]
    #         domain_word_lists[line]=1



    batch_dict=[[]]
    for i in range(len(batch_word[0])): #为每个字构造特征向量
        word_tag=[]
        #第i字为结尾的词是否存在于字典中
        for j in range(4,0,-1):  #j=3 2 1 0
            if (i-j)<0:
                word_tag.append(0)
                continue
            word=''.join(batch_word[0][i-j:i+1]) #j=3 2 1 0 时 代表以第i字为结尾的五字四字三字二字的词
            if domain_word_lists.get(word) is not None:
                word_tag.append(1)
            else:
                word_tag.append(0)

        for j in range(1,5):
            if (i+j)>=len(batch_word[0]):
                word_tag.append(0)
                continue
            word=''.join(batch_word[0][i:i+j+1])
            if domain_word_lists.get(word) is not None:
                word_tag.append(1)
            else:
                word_tag.append(0)
        batch_dict[0].append(word_tag)

    print(batch_dict)

















