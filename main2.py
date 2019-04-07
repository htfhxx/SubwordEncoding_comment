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


if __name__ == '__main__':
    print("aaa")




