
class DefaultConfigs(object):

    import torch
    root_dir = r"D:/circRNA/Deep_Circ/mysite/"
    data_dir = root_dir + r"Code/data/Humen/HumanTrain.csv"
    test_data_dir = root_dir + r"Code/data/Humen/HumanTest.csv"
    # custom_train_data_dir = root_dir + r"Code/data/Humen/Custom/HumanTrain.csv"     # 修改这个更换网页上训练集
    # custom_test_data_dir = root_dir + r"Code/data/Humen/Custom/HumanTest.csv"       # 修改这个更换网页上测试集
    custom_train_data_dir = root_dir + r"Code/data/Mouse/MouseTrain.csv"
    custom_test_data_dir = root_dir + r"Code/data/Mouse/MouseTest.csv"
    weight = root_dir + r"Code/models/Parameter/human_weight_Res_CnnNet.pt"
    weight0 = root_dir + r"Code/models/Parameter/human_weight_Res_CnnNet.pt"
    weight1 = root_dir + r"Code/models/Parameter/human_weight_Lstm_Attention_CNN.pt"
    weight2 = root_dir + r"Code/models/Parameter/human_weight_Gru.pt"
    weight3 = root_dir + r"Code/models/Parameter/human_weight_BiGru.pt"
    weight4 = root_dir + r"Code/models/Parameter/human_weight_Gru_Attention.pt"
    weight5 = root_dir + r"Code/models/Parameter/human_weight_Gru_MultiHeadedAttention.pt"
    weight100 = root_dir + r"Code/models/Custom_Parameter/custom_weight_Res_CnnNet.pt"
    weight101 = root_dir + r"Code/models/Custom_Parameter/custom_weight_Lstm_Attention_CNN.pt"
    weight102 = root_dir + r"Code/models/Custom_Parameter/custom_weight_Gru.pt"
    weight103 = root_dir + r"Code/models/Custom_Parameter/custom_weight_BiGru.pt"
    weight104 = root_dir + r"Code/models/Custom_Parameter/custom_weight_Gru_Attention.pt"
    weight105 = root_dir + r"Code/models/Custom_Parameter/custom_weight_Gru_MultiHeadedAttention.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    patience = 24
    Class_No = 2
    batch_size = 128
    num_layers = 3
    lr = 0.001
    input_size = 200
    hidden_size_Head = 768
    hidden_size = 128
    head = 12


cfg = DefaultConfigs()

