import pandas as pd
from collections import Counter
import torch
import time
import binascii
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from Code.config import cfg


def feat_conver(array):
    lie = []
    for str in array[:, 1]:  # 取出数组第2列数据
        hexst = binascii.hexlify(str.encode())
        str = [int(hexst[i:i + 2], 16) for i in range(0, len(hexst), 2)]  # 编码
        lie.append(str)
    lie = np.array(lie)
    lie = (lie - np.min(lie)) / (np.max(lie) - (np.min(lie)))   # 归一化
    target = array[:, 0:1]  # 取出数组第一列数据
    array = np.hstack((lie, target))  # 堆叠
    array = array.astype(float)
    return array


def load_train_data(file_path, ratio=0.7978):
    x = time.time()
    df = pd.read_csv(file_path, engine="python")
    df = np.array(df)
    dataset = feat_conver(df)
    len_data = len(dataset)
    y = time.time()
    import math
    len_valid = math.floor(ratio * len_data)
    print("读取训练集时间长为 === >> {:.4f} s".format(y - x))
    load_info = "读取训练集时间长为 === >> " + format(y - x, '.4f').__str__() + " s\r\n"
    train_data = dataset[:len_valid]
    print('train label :::')
    load_info += "train label :::\r\n"
    print("target  === >:", Counter(list(train_data[:, 200])), len(train_data))
    load_info += "target ----> " + Counter(list(train_data[:, 200])).__str__() + str(len(train_data)) + "\r\n"
    valid_data = dataset[len_valid:]
    print('valid label :::')
    print("target  === >:", Counter(list(valid_data[:, 200])), len(valid_data))
    load_info += "valid label ::: \n"
    load_info += "target  === >:" + Counter(list(valid_data[:, 200])).__str__() + str(len(valid_data)) + "\r\n"
    return train_data, valid_data, load_info


def load_test_data(file_path):
    x = time.time()
    df = pd.read_csv(file_path, engine="python")
    df = np.array(df)
    dataset = feat_conver(df)
    y = time.time()
    print("读取测试集时间长为 === >> {:.4f} s".format(y - x))
    test_data = dataset
    print('test label :::')
    print("target  === >:", Counter(list(test_data[:, 200])), len(test_data))
    return test_data


class Url_Dataset(Dataset):
    def __init__(self, data):
        self.dataset = data
        self.dataset = np.array(self.dataset)
        self.len = len(self.dataset)

    def __len__(self):
        return int(self.len)

    def __getitem__(self, index):
        data = self.dataset[:, :-1][index]
        data = torch.tensor(data, dtype=torch.float32)
        target = int(self.dataset[index][200])

        return data, target


if __name__ == '__main__':
    from config import *
    x, y = load_train_data(cfg.data_dir)
    print(x)
    print(x.shape)
    data = Url_Dataset(x)
    data_loader = DataLoader(data, batch_size=10, shuffle=True)
    for i, (data, lable1) in enumerate(data_loader):
        print(data)
        print(lable1)

