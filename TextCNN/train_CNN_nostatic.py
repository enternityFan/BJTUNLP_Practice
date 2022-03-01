# @Time : 2022-03-01 13:07
# @Author : Phalange
# @File : train_CNN_nostatic.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D

import torch
from torch import nn
from d2l import torch as d2l
from tqdm import *
import os
import sys
import numpy as np
import pickle
sys.path.append(r'F:\Study\NLP\BJTUNLP_Practice\TextCNN')
import TextCNN.Function as Function
import TextCNN.Module.CNN
import TextCNN.Module.trick


def init_weights(m):
    if type(m) in (nn.Linear, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)



if __name__ == "__main__":
    batch_size = 512
    data_dir_path = "./DATA/aclImdb"

    train_iter, test_iter, vocab = Function.load_data_aclImdb(data_dir_path, batch_size)

    embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
    lr, num_epochs,devices = 0.001, 30,d2l.try_all_gpus()
    net = TextCNN.Module.CNN.CNN_randModel(len(vocab),embed_size,kernel_sizes,nums_channels)
    glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
    embeds = glove_embedding[vocab.idx_to_token]
    net.embedding.weight.data.copy_(embeds)
    net.embedding.weight.requires_grad = True
    net.apply(init_weights)
    loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.Adam(net.parameters(),lr=lr)
    cosScheduler = TextCNN.Module.trick.CosineScheduler(max_update=30, warmup_steps=10,base_lr=lr, final_lr=0.0007)
    TextCNN.Module.trick.train_scheduler(net, train_iter, test_iter, loss, trainer, num_epochs,
                                 devices)
    d2l.plt.show()
    print("train success!")
    torch.save(net.state_dict(), './Cache/AttentionWeights.pth')