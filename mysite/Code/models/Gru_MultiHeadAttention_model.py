import math
import torch
import torch.nn as nn


class Attention(nn.Module):

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = torch.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):

    def __init__(self, d_model, head, dropout=0.3):
        super().__init__()
        assert d_model % head == 0

        self.d_k = d_model // head
        self.h = head
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):

        batch_size = query.size(0)

        # 1) 从d_model => h x d_k批处理所有线性投影
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) 分批注意所有投影向量。
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) 使用视图并应用。
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.output_linear(x)


class Gru_MultiHeadedAttention(nn.Module):
    def __init__(self, cfg, init_weight=True):
        super(Gru_MultiHeadedAttention, self).__init__()
        self.hidden_size = cfg.hidden_size_Head
        self.num_layers = cfg.num_layers
        self.input_size = cfg.input_size
        self.attention = MultiHeadedAttention(self.hidden_size, 12)

        self.gru = nn.GRU(self.input_size,
                          self.hidden_size,
                          self.num_layers,
                          batch_first=True,
                          bidirectional=False)

        self.fc = nn.Linear(self.hidden_size, cfg.Class_No)
        self.dropout = nn.Dropout(0.1)
        if init_weight:
            self._initialize_weights()

    def forward(self, x):

        out, _ = self.gru(x, None)
        out = self.attention(out, out, out)
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
#     mode = Gru_MultiHeadedAttention(cfg)
#     mode.eval()
#     input = torch.randn(1, 1, 200)
#     # print(input)
#     p = mode(input)
#     print(p)
#     print(p.shape)