import numpy as np
import binascii
import pandas as pd
from config import cfg


path = cfg.root_dir + r"Code/data/Humen/HumanTrain.csv"

data = pd.read_csv(path, engine="python")
data_train = np.array(data)


def Feat_conver(array):
    lie = []
    for str in array[:, 2]:
        hexst = binascii.hexlify(str.encode())
        str = [int(hexst[i:i + 2], 16) for i in range(0, len(hexst), 2)]  # 编码
        lie.append(str)
    lie = np.array(lie)
    target = array[:, 1:2]
    array = np.hstack((lie, target))  # 堆叠
    return array


z = Feat_conver(data_train)
print(z)