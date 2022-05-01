import math
import torch
import torch.nn as nn


class Gru(nn.Module):
    def __init__(self, cfg, init_weight=True):
        super(Gru, self).__init__()
        self.hidden_size = cfg.hidden_size
        self.num_layers = cfg.num_layers
        self.input_size = cfg.input_size

        self.gru = nn.GRU(self.input_size,
                          self.hidden_size,
                          self.num_layers,
                          batch_first=True,
                          bidirectional=False)

        self.fc = nn.Linear(cfg.hidden_size, cfg.Class_No)
        self.dropout = nn.Dropout(0.1)
        if init_weight:
            self._initialize_weights()

    def forward(self, x):
        out, _ = self.gru(x, None)
        # out = self.dropout(out)
        out = self.fc(out)
        out = torch.sigmoid(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.normal_(0.0, 0.001)


# if __name__ == '__main__':
#     from config import *
#     mode = Gru(cfg)
#     mode.eval()
#     input = torch.randn(1, 1, 9)
#     # print(input)
#     p = mode(input)
#     print(p)
#     print(p.shape)
