## !/usr/bin/python
# -*- coding: UTF-8 -*-
import os

IDA32_DIR = "I:\IDA7.2\ida.exe"
IDA64_DIR = "I:\IDA7.2\ida64.exe"

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))  # The path of the current file
CODE_DIR = ROOT_DIR
ORIGIN_DIR = ROOT_DIR + os.sep + "0_Origins"   
FEA_DIR = ROOT_DIR + os.sep + "1_Features"            # The root path of  the feature file
DATASET_DIR = ROOT_DIR + os.sep + "2_Dataset"
VUL_FUN_DIR = ROOT_DIR + os.sep + "4_Vul_Fun"
RES_DIR = ROOT_DIR + os.sep + "5_Res"

# ========================================
# 生成Generation
# ========================================
# if convert all binary files into disassembly files
G0_PORGRAM_ARR=["openssl","search_program"]#"openssl","coreutils","busybox","CVE-2015-1791"

# if extract feature file
G1_PORGRAM_ARR=["openssl"]  #  all the project name"openssl",,"busybox","coreutils","CVE-2015-1791"
G1_PORGRAM_ARR=["search_program"]  #  all the project name
G1_CVE_OPENSSL_FUN_LIST = {'ssl3_get_new_session_ticket':'CVE-2015-1791', 'OBJ_obj2txt':'CVE-2014-3508'}

# ========================================
# 训练Train
# ========================================
# if generate train dataset
T0_PORGRAM_ARR = ["openssl","busybox"]#"openssl","coreutils","busybox",
TRAIN_DATASET_NUM = 50000

# if start to train
T1_VULSEEKER_MODEL_TO_SAVE = ROOT_DIR + os.sep + "3_Output"+ os.sep + "vulseeker_model_weight"
T1_VULSEEKER_FIGURE_TO_SAVE = ROOT_DIR + os.sep + "3_Output"

vulseeker_feature_size = 8
train_func_num = 44067
valid_fun_num = 5509
test_func_num = 5508
Buffer_Size = 4000
mini_batch = 100
base_batch = 10
ratio = mini_batch//base_batch
learning_rate  = ratio*0.0001
decay_steps = 10 # 衰减步长
decay_rate = 0.96 # 衰减率

epochs  = 100
train_step_per_epoch = train_func_num//mini_batch
valid_step_per_epoch = valid_fun_num//mini_batch
test_step_per_epoch = test_func_num//mini_batch
### 超参数
T = 5
embedding_size = 64
embedding_depth = 2
norm_flag = 0
norm_layer_flag = 0

### some details about dataset generation
max_nodes = 500
min_nodes_threshold = 0
# ========================================
# 搜索Search
# ========================================





## VulSeeker





