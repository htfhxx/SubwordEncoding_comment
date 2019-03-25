# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2017-12-06 16:21:33
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np


#from： self.char_feature = CharBiLSTM(   # self=BiLSTM		30						50					0.5	
							#data.char_alphabet.size(), self.char_embedding_dim, self.char_hidden_dim, data.HP_dropout, self.gpu)
class CharBiLSTM(nn.Module):
    def __init__(self, alphabet_size, embedding_dim, hidden_dim, dropout, gpu, bidirect_flag = True):
        super(CharBiLSTM, self).__init__()
        print "build batched char bilstm..."
        self.gpu = gpu
        self.hidden_dim = hidden_dim    # LSTM的hidden dimension到底是什么？？
        if bidirect_flag:
            self.hidden_dim = hidden_dim // 2
        self.char_drop = nn.Dropout(dropout)
		#保存了固定字典和大小的简单查找表  		嵌入字典的大小*每个嵌入向量的大小
        self.char_embeddings = nn.Embedding(alphabet_size, embedding_dim) 
		#均匀分布中抽取样本 ??????????
        self.char_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(alphabet_size, embedding_dim)))
		#将一个多层的 (LSTM) 应用到输入序列
        self.char_lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=bidirect_flag)
        if self.gpu:
            self.char_drop = self.char_drop.cuda()
            self.char_embeddings = self.char_embeddings.cuda()
            self.char_lstm = self.char_lstm.cuda()
			
	#均匀分布中抽取样本 
	#from: self.char_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(alphabet_size, embedding_dim)))
    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])  #从均匀分布中抽取样本
        return pretrain_emb


    def get_last_hiddens(self, input, seq_lengths):
        """
            input:  
                input: Variable(batch_size, word_length)
                seq_lengths: numpy array (batch_size,  1)
            output: 
                Variable(batch_size, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = input.size(0)
        char_embeds = self.char_drop(self.char_embeddings(input))
        char_hidden = None
        pack_input = pack_padded_sequence(char_embeds, seq_lengths, True)
        char_rnn_out, char_hidden = self.char_lstm(pack_input, char_hidden)
        char_rnn_out, _ = pad_packed_sequence(char_rnn_out)
        return char_hidden[0].transpose(1,0).contiguous().view(batch_size,-1)

    def get_all_hiddens(self, input, seq_lengths):
        """
            input:  
                input: Variable(batch_size,  word_length)
                seq_lengths: numpy array (batch_size,  1)
            output: 
                Variable(batch_size, word_length, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = input.size(0)
        char_embeds = self.char_drop(self.char_embeddings(input))
        char_hidden = None
        pack_input = pack_padded_sequence(char_embeds, seq_lengths, True)
        char_rnn_out, char_hidden = self.char_lstm(pack_input, char_hidden)
        char_rnn_out, _ = pad_packed_sequence(char_rnn_out)
        return char_rnn_out.transpose(1,0)


    def forward(self, input, seq_lengths):
        return self.get_all_hiddens(input, seq_lengths)
        