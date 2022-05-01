import math
import torch
import torch.nn as nn


class Gru_Attention(nn.Module):
    def __init__(self, cfg, init_weight=True):
        super(Gru_Attention, self).__init__()
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

        query = self.dropout(out)
        key = self.dropout(out)
        value = self.dropout(out)

        out, weight = self.attention_net(query, key, value)
        out = self.fc(out)
        out = torch.sigmoid(out)
        return out

    def attention_net(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = torch.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.normal_(0.0, 0.001)