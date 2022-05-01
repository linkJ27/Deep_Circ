import torch.optim as optim
from Code.Custom_tools import *
import os
from Code.data.dataloader import *
from Code.models.Gru_model import Gru
from Code.models.BiGru_model import BiGru
from Code.models.Res_CnnNet_model import Res_CnnNet
from Code.models.Gru_Attention_model import Gru_Attention
from Code.models.Gru_MultiHeadAttention_model import Gru_MultiHeadedAttention
from Code.models.Cnn_Lstm_Attention.Lstm_Attention_CNN import Lstm_Attention_CNN
from Code.config import cfg
import zipfile
from torchinfo import summary

m_best_Acc = 0
stage = 0
info = ""
train_loss_dir = ""


def train_new_model(f_train, model_type, params):
    global stage
    global info
    global train_loss_dir
    stage = 0
    if model_type == 0:
        model = Res_CnnNet().to(cfg.device)
        weight = cfg.weight100
        title_name = "Res_CnnNet"
    elif model_type == 1:
        model = Lstm_Attention_CNN(cfg).to(cfg.device)
        weight = cfg.weight101
        title_name = "Lstm_Attention_CNN"
    elif model_type == 2:
        model = Gru(cfg).to(cfg.device)
        weight = cfg.weight102
        title_name = "Gru"
    elif model_type == 3:
        model = BiGru(cfg).to(cfg.device)
        weight = cfg.weight103
        title_name = "BiGru"
    elif model_type == 4:
        model = Gru_Attention(cfg).to(cfg.device)
        weight = cfg.weight104
        title_name = "Gru_Attention"
    elif model_type == 5:
        model = Gru_MultiHeadedAttention(cfg).to(cfg.device)
        weight = cfg.weight105
        title_name = "Gru_MultiHeadedAttention"
    else:
        model = Res_CnnNet().to(cfg.device)
        weight = cfg.weight100
        title_name = "Res_CnnNet"

    ratio = params.get('ratio')                 # 训练集和验证集比例  default:0.7978
    criterion_type = params.get('criterion')    # 损失函数  default:0
    optimizer_type = params.get('optimizer')    # 优化器   default:0
    m_lr = params.get('lr')                     # 学习率   default:0.001
    m_batch_size = params.get('batch_size')     # 每次训练大小    default:128
    is_shuffle = params.get('shuffle')          # 是否打乱  default:false
    patience = params.get('patience')           # EarlyStopping能够容忍多少个epoch内都没有improvement  default:24
    m_delta = params.get('delta')               # 每个epoch只有大于delta才算作improvement    default:0
    m_epoch = params.get('epoch')               # 最大迭代次数    default:2000

    if criterion_type == 0:
        m_criterion = nn.CrossEntropyLoss().to(cfg.device)
    elif criterion_type == 1:
        m_criterion = LabelSmoothingCrossEntropy(0.1).to(cfg.device)
    elif criterion_type == 2:
        m_criterion = torch.nn.BCEWithLogitsLoss().to(cfg.device)
    elif criterion_type == 3:
        m_criterion = torch.nn.L1Loss().to(cfg.device)
    else:
        m_criterion = nn.CrossEntropyLoss().to(cfg.device)

    if optimizer_type == 0:
        m_optimizer = optim.Adam(model.parameters(), m_lr)
    elif optimizer_type == 1:
        m_optimizer = optim.RMSprop(model.parameters(), m_lr)
    elif optimizer_type == 2:
        m_optimizer = optim.Adamax(model.parameters(), m_lr)
    elif optimizer_type == 3:
        m_optimizer = optim.SparseAdam(model.parameters(), m_lr)
    else:
        m_optimizer = optim.Adam(model.parameters(), m_lr)

    if cfg.device == "cuda":
        torch.backends.cudnn.benchmark = True

    model_struct = summary(model, input_size=(m_batch_size, 1, 200))

    info = "加载模型成功...\r\n读取文件完成...\r\n"
    info += model_struct.__str__()
    info += "\r\n"
    m_train_loader, m_val_loader, m_x, m_y = \
        m_data_loader(f_train, m_batch_size, ratio, is_shuffle)
    info += " _____ _             _     _____         _       _                          \r\n"
    info += "/  ___| |           | |   |_   _|       (_)     (_)                         \r\n"
    info += "\ `--.| |_ __ _ _ __| |_    | |_ __ __ _ _ _ __  _ _ __   __ _              \r\n"
    info += " `--. \ __/ _` | '__| __|   | | '__/ _` | | '_ \| | '_ \ / _` |             \r\n"
    info += "/\__/ / || (_| | |  | |_    | | | | (_| | | | | | | | | | (_| |   _   _   _ \r\n"
    info += "\____/ \__\__,_|_|   \__|   \_/_|  \__,_|_|_| |_|_|_| |_|\__, |  (_) (_) (_)\r\n"
    info += "                                                          __/ |             \r\n"
    info += "                                                         |___/              \r\n"
    stage += 1
    time.sleep(1)

    m_tra_lo, m_test_a, m_train_a, m_valid_a = [], [], [], []
    m_epoch_list = []
    m_early_stopping = EarlyStopping(weight=weight, patience=patience, verbose=True, delta=m_delta)
    m_start = time.time()

    for n in range(m_epoch):
        m_epoch_list.append(n)

        m_tra_loss, m_tra_acc, train_info = m_train(model, n, m_optimizer, m_train_loader, m_criterion, m_x)
        m_tra_lo.append(m_tra_loss)
        m_train_a.append(m_tra_acc)

        m_va_loss, m_va_acc, val_info = m_val(model, m_val_loader, m_criterion, m_y)
        m_valid_a.append(m_va_acc)

        es_info = m_early_stopping(m_va_loss, model)  # early_stopping

        info = train_info + val_info + es_info

        # wait
        if m_early_stopping.early_stop:
            print(" === > Early stopping ! ! ! ! ! ")
            info = "=== > Early stopping ! ! ! ! ! \r\n\r\n"
            break
        else:
            next_epoch = (n+1).__str__().ljust(4)
            info += "┌─────────────────────────────────┐\r\n"
            info += "│                                                                  │\r\n"
            info += "│                          epoch " + next_epoch + "                              │\r\n"
            info += "│                                                                  │\r\n"
            info += "└─────────────────────────────────┘\r\n"
            info += "training......\r\n"
            stage += 1
            time.sleep(0.1)
        print("....................... . Next . .......................")
        print("\n")

    m_end = time.time()
    train_time = format(m_end - m_start, '.2f').__str__() + "s"
    train_time = train_time.ljust(8)
    print("训练结束！训练时间长度为  ==== > {} s".format(m_end - m_start))
    info += "☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆\r\n"
    info += "☆                                                                                      ☆\r\n"
    info += "☆               训练结束，模型已保存！训练时间长度为  ==== > " + train_time + "                  ☆\r\n"
    info += "☆                                                                                      ☆\r\n"
    info += "☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆\r\n\r\n"
    info += "可在下方上传测试文件进行独立测试！\r\n"
    train_loss_dir = cfg.root_dir + r"Code/Evaluate_result/LOSS/Custom/" + "loss_and_acc_" + title_name + ".png"
    plot_curve(m_epoch_list, m_tra_lo, m_train_a, m_valid_a, title_name, "Human", train_loss_dir)
    stage = -999
    return


def m_test_model(model_type):
    if model_type == 0:
        model = Res_CnnNet().to(cfg.device)
        weight = cfg.weight100
        title_name = "Res_CnnNet"
    elif model_type == 1:
        model = Lstm_Attention_CNN(cfg).to(cfg.device)
        weight = cfg.weight101
        title_name = "Lstm_Attention_CNN"
    elif model_type == 2:
        model = Gru(cfg).to(cfg.device)
        weight = cfg.weight102
        title_name = "Gru"
    elif model_type == 3:
        model = BiGru(cfg).to(cfg.device)
        weight = cfg.weight103
        title_name = "BiGru"
    elif model_type == 4:
        model = Gru_Attention(cfg).to(cfg.device)
        weight = cfg.weight104
        title_name = "Gru_Attention"
    elif model_type == 5:
        model = Gru_MultiHeadedAttention(cfg).to(cfg.device)
        weight = cfg.weight105
        title_name = "Gru_MultiHeadedAttention"
    else:
        model = Res_CnnNet().to(cfg.device)
        weight = cfg.weight100
        title_name = "Res_CnnNet"

    model.load_state_dict(torch.load(weight, map_location=cfg.device))
    if 1 <= model_type <= 5:
        model.eval()

    m_test_loader, m_batch = test_data_loader()
    te_acc, pred, label, y_pred, arx = m_application(model, m_test_loader, m_batch)

    Custom_auc(label, y_pred, "Roc_Human_" + title_name,
               cfg.root_dir + r"Code/Evaluate_result/ROC/Custom/ROC_Human_" + title_name + ".png")

    p, recall, f1_score = show_confMat(arx, label,
                                       cfg.root_dir + r"Code/Evaluate_result/Matrix/Custom"
                                                      r"/MatrixConfusion_Matrix_Human_" + title_name + ".png",
                                       "Human_" + title_name)  # 混淆矩阵
    print("Accuracy:", p)  # 精度
    print("recall:", recall)  # 召回率
    print("f1_score:", f1_score)  # F1 值
    return {"acc": p, "recall": recall, "f1_score": f1_score}


def m_data_loader(train_dir, m_batch_size, ratio, shuffle=True):
    global info
    use_cuda = torch.cuda
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    train_data, valid_data, load_info = load_train_data(train_dir, ratio)
    # test_data = load_test_data(test_dir)
    batch_size = m_batch_size
    train_data = Url_Dataset(train_data)
    valid_data = Url_Dataset(valid_data)
    # test_data = Url_Dataset(test_data)

    train_ = DataLoader(dataset=train_data,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        **kwargs)

    valid_ = DataLoader(dataset=valid_data,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        **kwargs)

    # test_ = DataLoader(dataset=test_data,
    #                    batch_size=batch_size,
    #                    shuffle=shuffle,
    #                    **kwargs)

    info += load_info

    return train_, valid_, len(train_data), len(valid_data)


def test_data_loader(shuffle=False):
    use_cuda = torch.cuda
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    test_data = load_test_data(cfg.custom_test_data_dir)

    batch_size = len(test_data)
    test_data = Url_Dataset(test_data)

    test_ = DataLoader(dataset=test_data,
                       batch_size=batch_size,
                       shuffle=shuffle,
                       **kwargs)

    return test_, batch_size


def m_train(model, m_epoch, m_optimizer, m_train_loader, m_criterion, m_x):
    model.train()
    train_loss, train_acc, = 0, 0
    for img, target in m_train_loader:
        scheduler = torch.optim.lr_scheduler.StepLR(m_optimizer, step_size=2, gamma=0.98)
        img = img.unsqueeze(1).to(cfg.device)

        target = target.clone().detach().to(cfg.device)
        output_t = model(img)
        output_t = output_t.squeeze(1)

        loss_ = m_criterion(output_t, target)
        m_optimizer.zero_grad()
        loss_.backward()
        m_optimizer.step()

        scheduler.step()

        train_loss += loss_ * target.size(0)
        argmax = torch.argmax(output_t, 1)
        train_acc += (argmax == target).sum()

    print('epoch is {}, train_loss is {}'.format(m_epoch, train_loss / m_x))
    print("train_Accuracy is: {}".format(torch.true_divide(train_acc, m_x)))
    train_info = "epoch is " + m_epoch.__str__() + ":\r\n"
    train_info += "train_loss <--------> " + (train_loss / m_x).__str__() + "\r\n"
    train_info += "train_acc  <--------> " + (torch.true_divide(train_acc, m_x)).__str__() + "\r\n"
    return train_loss/m_x, torch.true_divide(train_acc, m_x), train_info


def m_val(model, m_val_loader, m_criterion, m_y):
    global m_best_Acc
    with torch.no_grad():

        val_loss, val_acc = 0, 0
        for space, lab in m_val_loader:

            space = space.unsqueeze(1).to(cfg.device)
            lab = lab.clone().detach() .to(cfg.device)
            output_v = model(space)
            output_v = output_v.squeeze(1)
            loss_v = m_criterion(output_v, lab)

            val_loss += loss_v * lab.size(0)
            argmax = torch.argmax(output_v, 1)
            val_acc += (argmax == lab).sum()

            torch.cuda.empty_cache()

        print('our best_acc is {}'.format(m_best_Acc))
        print("the val_Accuracy is: {}".format(torch.true_divide(val_acc, m_y)))
        val_info = ">>current best accuracy: " + torch.true_divide(val_acc, m_y).__str__() + "\r\n"
        val_info += ">>the val_Accuracy is:  " + torch.true_divide(val_acc, m_y).__str__() + "\r\n"
        val_acc = torch.true_divide(val_acc, m_y)

        if val_acc > m_best_Acc:
            print('Validation accuracy increased ......')
            val_info += "Validation accuracy increased ......\r\n"
            m_best_Acc = val_acc

        return val_loss/m_y, val_acc, val_info


def m_application(model, m_test_loader, m_batch):
    m_test_acc = 0
    with torch.no_grad():
        for data_set, label in m_test_loader:
            data_set = data_set.unsqueeze(1).to(cfg.device)
            label = label.clone().detach().to(cfg.device)
            output = model(data_set)
            output = output.squeeze(1)
            argmax = torch.argmax(output, 1)
            print(output)
            m_test_acc += (argmax == label).sum()
            torch.cuda.empty_cache()

        output = output.cpu().numpy()
        y_pred = np.true_divide(output[:, 1], output.sum(axis=1))
        print(y_pred)
        print(label)
        print(argmax)
        print("the test_Accuracy is: {}".format(torch.true_divide(m_test_acc, m_batch)))
        return torch.true_divide(m_test_acc, m_batch), output, label.cpu().numpy(), y_pred, argmax.cpu().numpy()


def get_current_info():
    global info
    global stage
    return {"stage": stage, "info": info}


def get_loss_image_dir(model_type):
    if model_type == 0:
        title_name = "Res_CnnNet"
    elif model_type == 1:
        title_name = "Lstm_Attention_CNN"
    elif model_type == 2:
        title_name = "Gru"
    elif model_type == 3:
        title_name = "BiGru"
    elif model_type == 4:
        title_name = "Gru_Attention"
    elif model_type == 5:
        title_name = "Gru_MultiHeadedAttention"
    else:
        title_name = "Res_CnnNet"
    path = cfg.root_dir + r"Code/Evaluate_result/LOSS/Custom/loss_and_acc_" + title_name + ".png"
    return path


def get_matrix_image_dir(model_type):
    if model_type == 0:
        title_name = "Res_CnnNet"
    elif model_type == 1:
        title_name = "Lstm_Attention_CNN"
    elif model_type == 2:
        title_name = "Gru"
    elif model_type == 3:
        title_name = "BiGru"
    elif model_type == 4:
        title_name = "Gru_Attention"
    elif model_type == 5:
        title_name = "Gru_MultiHeadedAttention"
    else:
        title_name = "Res_CnnNet"
    path = cfg.root_dir + r"Code/Evaluate_result/Matrix/Custom/MatrixConfusion_Matrix_Human_" + title_name + ".png"
    return path


def get_roc_image_dir(model_type):
    if model_type == 0:
        title_name = "Res_CnnNet"
    elif model_type == 1:
        title_name = "Lstm_Attention_CNN"
    elif model_type == 2:
        title_name = "Gru"
    elif model_type == 3:
        title_name = "BiGru"
    elif model_type == 4:
        title_name = "Gru_Attention"
    elif model_type == 5:
        title_name = "Gru_MultiHeadedAttention"
    else:
        title_name = "Res_CnnNet"
    path = cfg.root_dir + r"Code/Evaluate_result/ROC/Custom/ROC_Human_" + title_name + ".png"
    return path


def zip_file(type):
    if type == 0:
        zip_name = cfg.root_dir + r"Code/Evaluate_result/LOSS/loss.zip"
        read_dir = cfg.root_dir + r"Code/Evaluate_result/LOSS/Custom/"
    elif type == 1:
        zip_name = cfg.root_dir + r"Code/Evaluate_result/Matrix/matrix.zip"
        read_dir = cfg.root_dir + r"Code/Evaluate_result/Matrix/Custom/"
    else:
        zip_name = cfg.root_dir + r"Code/Evaluate_result/ROC/roc.zip"
        read_dir = cfg.root_dir + r"Code/Evaluate_result/ROC/Custom/"
    files = os.listdir(read_dir)
    file_list = []
    for file_one in files:
        file_list.append(read_dir + file_one)
    print(files)
    zp = zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED)
    for file_full, file_name in zip(file_list, files):
        zp.write(file_full, file_name)
    zp.close()
    return zip_name


if __name__ == '__main__':
    # params = {'ratio': 0.7978, 'criterion': 0, 'optimizer': 0, 'lr': 0.001, 'batch_size': 128, 'shuffle': True,
    #           'patience': 24, 'delta': 0.005, 'epoch': 200}
    # train_new_model(cfg.data_dir, 3, params)
    rna = 'ACGU'
    rna2 = 'CGUU'
    print(rna.encode())
    rna_1 = binascii.hexlify(rna.encode())
    rna2_1 = binascii.hexlify(rna2.encode())
    print(rna_1)
    out = [int(rna_1[i:i + 2], 16) for i in range(0, len(rna_1), 2)]  # 编码
    out2 = [int(rna_1[i:i + 2], 16) for i in range(0, len(rna2_1), 2)]  # 编码
    rna_list = [out, out2]
    print(rna_list)
    rna_list = np.array(rna_list)
    print(rna_list)
    new_list = (rna_list - np.min(rna_list)) / (np.max(rna_list) - (np.min(rna_list)))
    print(new_list)
