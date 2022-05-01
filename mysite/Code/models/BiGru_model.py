import math
import torch
import torch.nn as nn


class BiGru(nn.Module):
    def __init__(self, cfg, init_weight=True):
        super(BiGru, self).__init__()
        self.hidden_size = cfg.hidden_size
        self.num_layers = 2
        self.input_size = cfg.input_size

        self.bigru = nn.GRU(self.input_size,
                            self.hidden_size,
                            self.num_layers,
                            batch_first=True,
                            bidirectional=True)

        self.fc = nn.Linear(2 * cfg.hidden_size, cfg.Class_No)
        self.dropout = nn.Dropout(0.1)
        if init_weight:
            self._initialize_weights()

    def forward(self, x):
        out, _ = self.bigru(x, None)
        out = self.dropout(out)
        out = self.fc(out)
        out = torch.sigmoid(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.normal_(0.0, 0.001)