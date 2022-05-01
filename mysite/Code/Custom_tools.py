import torch
import random
import numpy as np
import torch.nn as nn
from sklearn import metrics
from itertools import cycle
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def setup_seed(seed=1):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def auc_show(target, pre, title_savename, n_class):
    classes = []
    for i in range(n_class):
        classes.append(i)
    target = label_binarize(target, classes=classes)
    print(target)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_class):
        fpr[i], tpr[i], _ = roc_curve(target[:, i], pre[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(target.ravel(), pre.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_class)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_class):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_class
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    lw = 2
    plt.figure()
    # plt.plot(fpr["micro"], tpr["micro"],
    #          label='micro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["micro"]),
    #          color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_class), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{}'.format(title_savename))
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(r"\Evaluate_result\ROC\{}.png".format(title_savename))
    # plt.show()
    plt.close()


def show_confMat(Pre_result, Target, out_dir, set_name:str):
    """
    混淆矩阵绘制
    :param  Pre_result:预测结果,
    :param  Target: 真实标签
    :return: 精度、召回率、F1值
    """
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import recall_score, precision_score, f1_score

    p = precision_score(Target, Pre_result, average='macro')
    recall = recall_score(Target, Pre_result, average='macro')
    f1_score = f1_score(Target, Pre_result, average='macro')

    classes = list(set(Target.tolist()))
    classes.sort()
    confusion = confusion_matrix(Target, Pre_result)
    # plt.imshow(confusion, cmap=plt.cm.Blues)
    plt.imshow(confusion, interpolation='nearest', cmap=plt.get_cmap('Oranges'))
    # Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn,
    # BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2,
    indices = range(len(confusion))
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    plt.colorbar()
    plt.title('Confusion_Matrix_' + set_name)
    plt.xlabel('Pre result')
    plt.ylabel('True label')
    for first_index in range(len(confusion)):
        for second_index in range(len(confusion[first_index])):
            plt.text(first_index, second_index, confusion[first_index][second_index], ha='center')
    plt.tight_layout()
    plt.savefig(out_dir)
    # plt.show()
    plt.close()
    return p, recall, f1_score


def plot_curve(epoch_list, train_loss, train_acc, valid_acc, A, B, out_dir:str):
    """
     绘制训练和验证集的loss曲线/acc曲线
     :param epoch_list: 迭代次数
     :param train_loss: 训练损失
     # :param test_acc:   训练测试精度
     :param train_acc:  训练精度
     :param valid_acc:  验证精度
     :param out_dir:    保存路径
     :return:
     """
    # epoch = [epoch for epoch in range(len(epoch_list))]
    epoch = epoch_list
    plt.subplot(2, 1, 1)

    plt.plot(epoch, train_acc, label="Train_acc_{}".format(B))
    plt.plot(epoch, valid_acc, label="Valid_acc_{}".format(B))

    # plt.plot(epoch, test_acc, label="Test_acc_{}".format(B))
    # import time
    # now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    plt.title('Acc and Loss')
    plt.xlabel('Training ({}) Acc of model vs .-epochs/{}'.format(A, B))
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.subplot(2, 1, 2)
    print(train_loss)
    loss_list = []
    for train_loss_one in train_loss:
        loss_list.append(train_loss_one.cpu().detach().numpy())

    plt.plot(epoch, loss_list, label="Train_loss_{}".format(B))
    plt.xlabel('Training ({}) Loss of model vs .-epochs/{}'.format(A, B))
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(str(out_dir))
    # plt.show()
    plt.close()


class EarlyStopping():
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, weight, patience=12, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.weight = weight

    def __call__(self, val_loss, model):

        es_info = ""
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter ----- > : {self.counter} out of {self.patience}')
            es_info = "EarlyStopping counter --------> " +\
                      self.counter.__str__() + " out of " + self.patience.__str__() + "\r\n"
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        return es_info

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ----> ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.weight)	    # 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss


def Custom_auc(label, target, name, out_dir: str):

    fpr, tpr, thresholds = metrics.roc_curve(label, target)
    roc_auc = metrics.auc(fpr, tpr)  # auc为Roc曲线下的面积
    print("auc-- : ", roc_auc)

    plt.plot(fpr, tpr, 'b', label='Resnet(AUC = %0.2f)' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate')  # 横坐标是fpr
    plt.ylabel('True Positive Rate')  # 纵坐标是tpr
    plt.title('{}'.format(name))
    plt.tight_layout()
    plt.savefig(out_dir)
    # plt.show()
    plt.close()


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def linear_combination(self, x, y, epsilon):
        loss = epsilon * x + (1 - epsilon) * y
        return loss

    def reduce_loss(self, loss, reduction='mean'):
        return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        smooth_loss = self.linear_combination(loss / n, nll, self.epsilon)
        return smooth_loss




