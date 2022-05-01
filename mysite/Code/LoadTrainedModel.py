from models.Gru_model import Gru
from models.BiGru_model import BiGru
from models.Res_CnnNet_model import Res_CnnNet
from models.Gru_Attention_model import Gru_Attention
from models.Gru_MultiHeadAttention_model import Gru_MultiHeadedAttention
from models.Cnn_Lstm_Attention.Lstm_Attention_CNN import Lstm_Attention_CNN
import torch
from config import *
import binascii
import numpy as np
from data.dataloader import *


if __name__ == '__main__':
    # MODEL = Gru(cfg).to(cfg.device)
    # MODEL = BiGru(cfg).to(cfg.device)
    # MODEL = Gru_Attention(cfg).to(cfg.device)
    # MODEL = Gru_MultiHeadedAttention(cfg).to(cfg.device)
    MODEL = Res_CnnNet().to(cfg.device)
    # MODEL = Lstm_Attention_CNN(cfg).to(cfg.device)
    state_dict = torch.load(cfg.weight0)
    torch.no_grad()
    MODEL.load_state_dict(state_dict)
    df = pd.read_csv(cfg.root_dir + r"Code/data/Humen/HumanTest2.csv", engine="python")
    df = np.array(df)
    y = len(df)
    print(y)
    dataset = feat_conver(df)
    test2 = Url_Dataset(dataset)
    use_cuda = torch.cuda
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    test2 = DataLoader(dataset=test2,
                       batch_size=64,
                       shuffle=False,
                       **kwargs)
    val_acc = 0
    MODEL.eval()
    with torch.no_grad():
        for dataset, label in test2:
            dataset = dataset.unsqueeze(1).to(cfg.device)
            label = label.clone().detach().to(cfg.device)
            output = MODEL(dataset)
            output = output.squeeze(1)
            print(output)
            argmax = torch.argmax(output, 1)
            print(argmax)
            print(label)
            print('---------------------------------')
            val_acc += (argmax == label).sum()

        print("the test_Accuracy is: {}".format(torch.true_divide(val_acc, y)))




    # test = 'AAACAUUAUUUCUUACUUAAUUAUUUGGCUGAAGAAAAUACAGAAGUGUCCCGUCGGGUUACCUAGGUUACCACUACUUCUCUGGAGUCAACAUACCGUAUGGUACUUCUGAACUUGAAUUAUGUCGUUCUUUCUGGUAGUUCAAAUGGUCAUUCUGUAAUAACACGACUAAACCUUUACAUUACUCAAUUUCUGAAAAU'
    # hexst = binascii.hexlify(test.encode())
    # test_data = [int(hexst[i:i + 2], 16) for i in range(0, len(hexst), 2)]
    # test_data = np.array(test_data)
    # test_data = (test_data - np.min(test_data)) / (np.max(test_data) - (np.min(test_data)))
    #
    #
    # test_data = test_data.astype(float)

    # test_data = torch.tensor(test_data)
    # test_data = test_data.unsqueeze(0)
    # test_data = test_data.unsqueeze(0)
    # test_data = test_data.type(torch.FloatTensor)
    # ttt = torch.randn(1, 1, 200)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # test_data = test_data.to(device)
    #
    # MODEL.eval()
    # outputs = MODEL(test_data)
    # outputs = outputs.squeeze(1)
    # argmax = torch.argmax(outputs, 1)
    # print(argmax)
