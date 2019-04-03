
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
    gpu = torch.cuda.is_available()
    print(gpu)

    print("num:",torch.cuda.device_count())
    torch.cuda.set_device(0)
    print(torch.cuda.current_device())









