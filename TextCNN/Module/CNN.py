# @Time : 2022-03-01 9:29
# @Author : Phalange
# @File : CNN.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D
import torch
from torch import nn

class CNN_randModel(nn.Module):
    def __init__(self,vocab_size,embed_size,kernel_sizes,num_channels,**kwargs):
        super(CNN_randModel, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.dropout = nn.Dropout(0.5)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu  =nn.ReLU()
        self.convs = nn.ModuleList()
        self.decode = nn.Linear(sum(num_channels),2)

        for c,k in zip(num_channels,kernel_sizes):
            self.convs.append(nn.Conv1d(embed_size,c,k))

    def forward(self,inputs):
        embedding = self.embedding(inputs).permute(0,2,1)
        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(embedding))),dim=-1)
            for conv in self.convs
        ],dim=1)
        outputs = self.decode(self.dropout(encoding))

        return outputs


class CNN_Model(nn.Module):
    def __init__(self,vocab_size,embed_size,kernel_sizes,num_channels,**kwargs):
        super(CNN_Model, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.constant_embedding = nn.Embedding(vocab_size,embed_size)
        self.dropout = nn.Dropout(0.5)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu  =nn.ReLU()
        self.convs = nn.ModuleList()
        self.decode = nn.Linear(sum(num_channels),2)

        for c,k in zip(num_channels,kernel_sizes):
            self.convs.append(nn.Conv1d(embed_size * 2,c,k))

    def forward(self,inputs):
        embeddings = torch.cat((
            self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        embeddings = embeddings.permute(0,2,1)
        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(embeddings))),dim=-1)
            for conv in self.convs
        ],dim=1)
        outputs = self.decode(self.dropout(encoding))

        return outputs