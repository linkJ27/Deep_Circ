from Code.Custom_tools import *
from Code.data.dataloader import *
from torch.utils.data import DataLoader
from Code.models.Gru_model import Gru
from Code.models.BiGru_model import BiGru
from Code.models.Res_CnnNet_model import Res_CnnNet
from Code.models.Gru_Attention_model import Gru_Attention
from Code.models.Gru_MultiHeadAttention_model import Gru_MultiHeadedAttention
from Code.models.Cnn_Lstm_Attention.Lstm_Attention_CNN import Lstm_Attention_CNN

setup_seed(1)   # 保证和训练时随机种子一样


def data_loader(cfg, shuffle=False):
    use_cuda = torch.cuda
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    test_data = load_test_data(cfg.test_data_dir)

    batch_size = len(test_data)
    test_data = Url_Dataset(test_data)

    test_ = DataLoader(dataset=test_data,
                       batch_size=batch_size,
                       shuffle=shuffle,
                       **kwargs)

    return test_, batch_size


def application(cfg):
    test_acc = 0
    with torch.no_grad():
        for dataset, label in test_loader:
            dataset = dataset.unsqueeze(1).to(cfg.device)
            label = label.clone().detach().to(cfg.device)
            output = MODEL(dataset)
            output = output.squeeze(1)
            argmax = torch.argmax(output, 1)
            print(output)
            test_acc += (argmax == label).sum()
            torch.cuda.empty_cache()

        print("the test_Accuracy is: {}".format(torch.true_divide(test_acc, batch)))
        return torch.true_divide(test_acc, batch), output.cpu().numpy(), label.cpu().numpy(), argmax.cpu().numpy()


if __name__ == '__main__':

    print('==> Model test......')
    MODEL = Gru(cfg).to(cfg.device)
    # MODEL = BiGru(cfg).to(cfg.device)
    # MODEL = Gru_Attention(cfg).to(cfg.device)
    # MODEL = Gru_MultiHeadedAttention(cfg).to(cfg.device)
    # MODEL = Res_CnnNet().to(cfg.device)
    # MODEL = Lstm_Attention_CNN(cfg).to(cfg.device)
    MODEL.load_state_dict(torch.load(cfg.weight2))
    MODEL.eval()    # eval模式停止dropout并且不用训练时的BatchNorm
    test_loader, batch = data_loader(cfg)

    te_acc, pred, target, arx = application(cfg=cfg)
    # Custom_auc(target, arx, "Class_Roc_Human_Lstm_Attention_CNN")      # ROC曲线
    # # p, recall, f1_score = show_confMat(arx, target,
    # #                                    r"D:\circRNA\A deep learning approach to identify circRNA\Code\Evaluate_result\Matrix"
    # #                                    r"\MatrixConfusion_Matrix_Human_Lstm_Attention_CNN.png",
    # #                                    "Human_Res_Lstm_Attention_CNN")  # 混淆矩阵
    #
    # p, recall, f1_score = show_confMat(arx, target,
    #                                    cfg.root_dir + r"\Code\Evaluate_result\Matrix"
    #                                    r"\MatrixConfusion_Matrix_Human_Lstm_Attention_CNN.png",
    #                                    "Human_Lstm_Attention_CNN")  # 混淆矩阵
    #
    # # https://blog.csdn.net/hfutdog/article/details/88085878
    # print("Accuracy:", p)  # 精度
    # print("recall:", recall)  # 召回率
    # print("f1_score:", f1_score)  # F1 值


def test_best_model(rna_list, identify_method):

    if identify_method == 0:
        model = Res_CnnNet().to(cfg.device)
        model.dropout = nn.Dropout(p=0)
        weight = cfg.weight0
    elif identify_method == 1:
        model = Lstm_Attention_CNN(cfg).to(cfg.device)
        weight = cfg.weight1
    elif identify_method == 2:
        model = Gru(cfg).to(cfg.device)
        weight = cfg.weight2
    elif identify_method == 3:
        model = BiGru(cfg).to(cfg.device)
        weight = cfg.weight3
    elif identify_method == 4:
        model = Gru_Attention(cfg).to(cfg.device)
        weight = cfg.weight4
    elif identify_method == 5:
        model = Gru_MultiHeadedAttention(cfg).to(cfg.device)
        weight = cfg.weight5
    else:
        model = Res_CnnNet().to(cfg.device)
        weight = cfg.weight0

    model.load_state_dict(torch.load(weight, map_location=cfg.device))
    if 1 <= identify_method <= 5:
        model.eval()

    rna_array = np.array(rna_list)
    lie = []
    for s in rna_array:
        hexst = binascii.hexlify(s.encode())
        s = [int(hexst[x:x + 2], 16) for x in range(0, len(hexst), 2)]  # 编码
        lie.append(s)
    lie = np.array(lie)
    lie = (lie - np.min(lie)) / (np.max(lie) - (np.min(lie)))
    rna_data = lie.astype(float)

    batch_size = len(rna_array)
    kwargs = {'num_workers': 0, 'pin_memory': True} if torch.cuda else {}
    shuffle = False
    rna_data = Rna_Dataset(rna_data)
    rna_loader = DataLoader(dataset=rna_data,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           **kwargs)

    result_list = []
    with torch.no_grad():
        for rna_input in rna_loader:
            rna_input = rna_input.unsqueeze(1).to(cfg.device)
            output = model(rna_input)
            output = output.squeeze(1)
            print(output)
            temp_result = torch.argmax(output, 1)
            if torch.cuda:
                temp_result = temp_result.cpu()
            temp_list = temp_result.numpy().tolist()
            result_list += temp_list
            torch.cuda.empty_cache()

    return result_list


class Rna_Dataset(Dataset):
    def __init__(self, data):
        self.dataset = data
        # self.dataset = np.array(self.dataset)
        self.len = len(self.dataset)

    def __len__(self):
        return int(self.len)

    def __getitem__(self, index):
        data = self.dataset[index]
        data = torch.tensor(data, dtype=torch.float32)
        return data