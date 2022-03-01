# @Time : 2022-03-01 9:42
# @Author : Phalange
# @File : Function.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D

import torch
from d2l import torch as d2l
from tqdm import *
import os
import numpy as np
import pickle

def read_acllmdb(data_dir_path,save_path="F:/Study/NLP/BJTUNLP_Practice/TextCNN/DATA/"):
    """读取IMDB评论数据集文本序列和标签,并且把训练集和测试集结合起来"""
    data,labels = [],[]
    if os.path.exists(save_path + "labels.pkl") and os.path.exists(save_path + "data.pkl"):
        with open(os.path.join(save_path,"labels.pkl"),'rb') as f:
            labels = pickle.loads(f.read())
        with open(os.path.join(save_path,"data.pkl"),'rb') as f:
            data = pickle.loads(f.read())
        print("loading success!")

        return data,labels

    for dir in "test","train":
        data_path = os.path.join(data_dir_path,dir)
        for label in "pos","neg":
            data_label_path = os.path.join(data_path,label)
            loop = tqdm(os.listdir(data_label_path),leave=True)
            for each_file in loop:
                # each_file: data_dir_path/test/pos/0_2.txt

                labels.append(1 if label=='pos' else 0)
                with open(os.path.join(data_label_path,each_file),'r',encoding='utf-8') as f:
                    data.append(f.read().replace('\n',''))

                # 更新信息
                loop.set_description(f'reading {data_label_path}')
    output_hal = open(os.path.join(save_path,"labels.pkl"),'wb')
    str = pickle.dumps(labels)
    output_hal.write(str)
    output_hal.close()
    output_hal = open(os.path.join(save_path,"data.pkl"), 'wb')
    str = pickle.dumps(data)
    output_hal.write(str)
    output_hal.close()


    return data,labels

def tokenize(lines):
    return [line.split() for line in lines]


def load_data_aclImdb(data_dir_path,batch_size,num_steps = 500):
    data,labels = read_acllmdb(data_dir_path)
    # 随机打乱一下数据
    index = list(np.arange(len(data)))
    np.random.shuffle(index)
    data = np.array(data)[index]
    labels = np.array(labels)[index]

    train_data,train_label = list(data[:int(len(data)*0.9)]),list(labels[:int(len(labels)*0.9)])
    test_data,test_label = list(data[int(len(data)*0.9):]),list(labels[int(len(labels)*0.9):])
    train_tokens = tokenize(train_data)
    vocab = d2l.Vocab(train_tokens,min_freq=5,reserved_tokens=['<pad>'])
    test_tokens = tokenize(test_data)

    train_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, torch.tensor(train_label)),
                                batch_size)
    test_iter = d2l.load_array((test_features, torch.tensor(test_label)),
                               batch_size,
                               is_train=False)
    return train_iter,test_iter,vocab




if __name__ == "__main__":
    data_dir_path = "./DATA/aclImdb"
    myTimer = d2l.Timer()
    train_iter,test_iter,vocab = load_data_aclImdb(data_dir_path,64)
    # 处理时间需要2分钟大概。。
    print(f'read success,speed time:  {myTimer.stop()}')


    #d2l.set_figsize()
    #d2l.plt.xlabel('# tokens per review')
    #d2l.plt.ylabel('count')
    #d2l.plt.hist([len(line) for line in tokens], bins=range(0, 1000, 50))
    #d2l.plt.show()
    #print("success!")
