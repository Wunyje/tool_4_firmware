import config
import pickle
import tensorflow as tf
import os
# import glob
import csv
import networkx as nx
import numpy as np


def generate_all_func_dict():  # 生成方程字典，初步生成数据集，其中有CFG、DFG与块特征
    all_func_dict = {}
    for program in config.T0_PORGRAM_ARR: 
        p_count = 0
        for a_c_o in os.listdir(config.FEA_DIR + os.sep + program):  # 指定不同架构-编译器-优化选项
            a_c_o_count = 0
            for version in os.listdir(config.FEA_DIR + os.sep + program + os.sep + a_c_o):
                v_count = 0
                fea_dir = config.FEA_DIR + os.sep + program + os.sep + a_c_o + os.sep + version
                if not os.path.exists(fea_dir):
                    continue
                function_list_csv = open(fea_dir + os.sep + "functions_list_fea.csv", "r")
                # ？在哪生成，后续根据此函数名列表生成对应特征
                for line in csv.reader(function_list_csv):  # 根据方程列表
                    cfg = read_cfg(line[0], fea_dir)     # 读相应cfg
                    dfg = read_dfg(line[0], fea_dir)     # 读相应dfg
                    node_size = len(cfg)        # 读相应块？节点？个数
                    if node_size < config.min_nodes_threshold:  # 块？节点？个数小于0，没有块？只有一个块？
                        continue  # 跳过此次循环，执行下一次循环
                    if all_func_dict.get(line[0]) == None:  # 该key是否存在
                        all_func_dict[line[0]] = []  # 不存在则创建该key，设对应value为空
                    feature = read_feature(line[0], fea_dir, node_size)
                    p_count = p_count + 1
                    a_c_o_count = a_c_o_count + 1
                    v_count = v_count + 1
                    assert len(cfg.nodes) == len(dfg.nodes) == feature.shape[0], \
                        "binary:%s func:%s cfg:%d dfg:%d and feature:%d_matrix's shape not consistent!"\
                        %(fea_dir, line[0], len(cfg.nodes), len(dfg.nodes), feature.shape[0])
                    g = (cfg, dfg, feature)  # 得到最终cfg+dfg+块特征
                    all_func_dict[line[0]].append(g)
                function_list_csv.close()
                print(version + " :" + str(v_count))
            print(a_c_o + " :" + str(a_c_o_count) + '\n')
        print(program + " :" + str(p_count) + '\n')
    return all_func_dict


def read_cfg(funcname, fea_dir):  # 控制流图
    cfg_path = fea_dir + os.sep + funcname + "_cfg.txt"
    cfg = nx.read_adjlist(cfg_path)
    return cfg


def read_dfg(funcname, fea_dir):  # 数据流图
    dfg_path = fea_dir + os.sep + funcname + "_dfg.txt"
    dfg = nx.read_adjlist(dfg_path)
    return dfg


def read_feature(funcname, fea_dir, nodes_num):  #
    feat_matrix = np.zeros(shape=(nodes_num, config.vulseeker_feature_size), dtype=np.int)
    feature_path = fea_dir + os.sep + funcname + "_fea.csv"
    f = open(feature_path, "r")
    for i, line in enumerate(csv.reader(f)):
        feat_matrix[i, :] = line[8:8+config.vulseeker_feature_size]
    f.close()
    return feat_matrix


def dataset_split(all_function_dict):
    all_func_num = len(all_function_dict)
    train_func_num = int(all_func_num * 0.8)  # 80%作训练集
    test_func_num = int(all_func_num * 0.1)  # 10%作测试集，10%作验证集

    train_name = np.random.choice(list(all_function_dict.keys()), size=train_func_num, replace=False)
    # 从给定的1维数组中随机采样，此处是从方程字典key中随机采样名称
    train_func = {}
    for func in train_name:
        train_func[func] = all_function_dict[func]
        all_function_dict.pop(func)  # 得到key对应内容并删除
    with open(config.DATASET_DIR+ os.sep +"train", "wb") as f:
        pickle.dump(train_func, f)

    test_func = {}
    test_name = np.random.choice(list(all_function_dict.keys()), size=test_func_num, replace=False)
    for func in test_name:
        test_func[func] = all_function_dict[func]
        all_function_dict.pop(func)
    with open(config.DATASET_DIR+ os.sep + "test", "wb") as f:
        pickle.dump(test_func, f)

    valid_func = all_function_dict
    valid_fun_num = len(all_function_dict)
    with open(config.DATASET_DIR+ os.sep + "valid", "wb") as f:
        pickle.dump(valid_func, f)

    print("train dataset's num =%s ,valid dataset's num=%s , test dataset's num =%s"
          % (train_func_num, valid_fun_num, test_func_num))
####################################################################
def dataset_generation(type = "train"):
    data = tf.data.Dataset.from_generator(generate_pairs, \
                                        output_types=(tf.float32, tf.float32, tf.float32, \
                                                    tf.float32, tf.float32, tf.float32, tf.float32), args=[type])
    data = data.repeat()
    data = data.shuffle(buffer_size=config.Buffer_Size) # 1000
    data = data.batch(batch_size=config.mini_batch) # 10
    data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE) # 根据可用的CPU动态设置并行调用的数量
    return data


def generate_pairs(type = b"train"):
    assert type == b"train" or type == b"test" or type == b"valid", "dataset type error!"
    filepath = config.DATASET_DIR + os.sep + type.decode()  # 前面已经分好了训练集、测试集和验证集，在训练时生成
    with open(filepath, "rb") as f:
        func_dict = pickle.load(f)
    funcname_list = list(func_dict.keys())
    length = len(funcname_list)  # 函数总个数
    for funcname in funcname_list:
        func_list = func_dict[funcname]  # 同名函数？
        if len(func_list) < 2:
            continue
        for i, func in enumerate(func_list):  # func_list
            # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中
            cfg, dfg, feat_matrix = func  # 方程字典中的value
            cfg, dfg, feat_matrix = zero_padded_adjmat(cfg, config.max_nodes), \
                                    zero_padded_adjmat(dfg, config.max_nodes), \
                                    zero_padded_featmat(feat_matrix, config.max_nodes)  #
            for j in range(2):
                # 生成正例
                if j == 0:
                    index = np.random.randint(low=0, high=len(func_list))
                    # 返回一个随机整型数
                    while index == i:
                        index = np.random.randint(low=0, high=len(func_list))
                    func_1 = func_list[index]
                    cfg_1, dfg_1, feat_matrix_1 = func_1
                    cfg_1, dfg_1, feat_matrix_1 = zero_padded_adjmat(cfg_1, config.max_nodes), \
                                                  zero_padded_adjmat(dfg_1, config.max_nodes), \
                                                  zero_padded_featmat(feat_matrix_1, config.max_nodes)
                    pair = (cfg, dfg, feat_matrix, cfg_1, dfg_1, feat_matrix_1, 1)  # 1为标签
                # 生成负例
                else:
                    index = np.random.randint(low=0, high=length)
                    while funcname_list[index] == funcname:
                        index = np.random.randint(low=0, high=length)
                    g2_index = np.random.randint(low=0, high=len(func_dict[funcname_list[index]]))
                    func_2 = func_dict[funcname_list[index]][g2_index]
                    cfg_2, dfg_2, feat_matrix_2 = func_2
                    cfg_2, dfg_2, feat_matrix_2 = zero_padded_adjmat(cfg_2, config.max_nodes), \
                                                  zero_padded_adjmat(dfg_2, config.max_nodes), \
                                                  zero_padded_featmat(feat_matrix_2, config.max_nodes)
                    pair = (cfg, dfg, feat_matrix, cfg_2, dfg_2, feat_matrix_2, -1)
                yield pair  # 生成器


def zero_padded_adjmat(graph, size):
    unpadded = adjmat(graph)  # 提取邻接矩阵
    padded = np.zeros((size, size))

    # 怎么放进padded
    if len(graph) > size:
        padded = unpadded[0:size, 0:size]
    else:
        padded[0:unpadded.shape[0], 0:unpadded.shape[1]] = unpadded
    return padded


def adjmat(gr):
    return nx.adjacency_matrix(gr).toarray().astype('float32')


# 生成特征矩阵
def zero_padded_featmat(feat_matrix, size):
    padded = np.zeros(shape=(size, config.vulseeker_feature_size))  # n*8
    nodes = feat_matrix.shape[0]
    if nodes > size:
        padded = feat_matrix[0:size, :]
    else:
        padded[0:nodes, :] = feat_matrix
    return padded

def generate_dataset():
    func_dict = generate_all_func_dict()
    dataset_split(func_dict)

if __name__ == '__main__':
    generate_dataset()
