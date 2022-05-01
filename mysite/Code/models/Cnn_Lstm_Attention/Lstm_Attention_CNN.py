import torch
import torch.nn as nn
from Code.models.Cnn_Lstm_Attention.Ghost_net import ghost_net
from Code.models.Cnn_Lstm_Attention.LstmNet import Lstm_Attention


def Matrix_trans(x):
    return x.repeat(1, 1, 32).reshape(-1, 1, 32, 32)


class Lstm_Attention_CNN(nn.Module):
    def __init__(self, cfg):
        super(Lstm_Attention_CNN, self).__init__()
        self.lstm = Lstm_Attention(cfg)
        self.cnn = ghost_net()

    def forward(self, x):
        x = self.lstm(x)
        x = Matrix_trans(x)
        x = self.cnn(x)
        x = torch.sigmoid(x)
        return x


if __name__ == '__main__':
    from config import *
    mode = Lstm_Attention_CNN(cfg)
    mode.eval()
    input = torch.randn(1, 1, 200)
    p = mode(input)
    print(p)
    print(p.shape)

