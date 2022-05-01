import torch
import torch.nn as nn


class Conv_Block_33(nn.Module):
    def __init__(self, chans=64):
        super(Conv_Block_33, self).__init__()
        self.conv_1x1_first = nn.Sequential(
            nn.Conv1d(chans, 128,
                      kernel_size=1), nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.conv_3x3_second = nn.Sequential(
            nn.Conv1d(128, 128,
                      kernel_size=3, padding=1), nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.conv_1x1_third = nn.Sequential(
            nn.Conv1d(128, chans,
                      kernel_size=1), nn.BatchNorm1d(chans),
            nn.ReLU()
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.conv_1x1_first(x)
        x = self.conv_3x3_second(x)
        x = self.conv_1x1_third(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.normal_(0.0, 0.001)


class Res_CnnNet(nn.Module):
    def __init__(self, n_chans=64):
        super(Res_CnnNet, self).__init__()
        self.n_chans1 = n_chans
        self.conv1 = nn.Conv1d(1, n_chans, kernel_size=3, stride=2,  padding=1)
        self.conv2 = nn.Conv1d(n_chans, n_chans, kernel_size=3, stride=2, padding=1)

        self.max_poold = nn.MaxPool1d(2, 2)
        self.batch_norm = nn.BatchNorm1d(num_features=n_chans)
        self.batch_norm2 = nn.BatchNorm1d(num_features=n_chans)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0)
        # self.dropout = nn.Dropout(p=0)

        self.resblock = Conv_Block_33(n_chans)

        self.fc = nn.Linear(64 * 25, 2)

        torch.nn.init.kaiming_normal_(self.conv1.weight,
                                      nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.conv2.weight,
                                      nonlinearity='relu')

        torch.nn.init.constant_(self.batch_norm.weight, 0.5)
        torch.nn.init.zeros_(self.batch_norm.bias)

        nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, x_frist):

        x_cov1 = self.conv1(x_frist)
        x_normal = self.relu(self.batch_norm(x_cov1))       # 0
        x_res1 = self.resblock(x_normal)                  # 1
        x_res2 = self.resblock(x_res1 + x_normal)         # 2
        x_res3 = self.resblock(x_res2 + x_res1)           # 3
        x_res4 = self.resblock(x_res3 + x_res2)           # 4

        x_cov2 = self.conv2(x_res4 + x_res3)
        x = self.relu(self.batch_norm2(x_cov2))
        x = self.max_poold(x)
        x = self.dropout(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x



# if __name__ == '__main__':
#     model = Res_CnnNet()
#     model.eval()
#     input = torch.randn(1, 1, 200)
#     y = model(input)
#     print(y)
#     print(y.size())






