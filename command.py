## !/usr/bin/python
# -*- coding: UTF-8 -*-
import config
import os
import subprocess
import glob

### 生成
[G_GENERATE,G0_GEN_IDB_FILE,G1_GEN_FEA ]= [1,1,1]
if G_GENERATE:
    if G0_GEN_IDB_FILE:
        print("G0. 将二进制文件反汇编")
        print("="*20)
        subprocess.call(["python", config.CODE_DIR+ os.sep + "g0_gen_idb_file.py"])
    if G1_GEN_FEA:
        print("G1. 生成特征文件")
        print("="*20)
        if not os.path.exists(config.FEA_DIR):
            os.mkdir(config.FEA_DIR)

        for program in config.G1_PORGRAM_ARR:
            tempdir = config.FEA_DIR + os.sep + str(program)
            if not os.path.exists(tempdir):
                os.mkdir(tempdir)

            for version in os.listdir(config.ORIGIN_DIR + os.sep + program):
                curFeaDir = config.FEA_DIR + str(os.sep) + str(program) + str(os.sep) + str(version)
                curBinDir = config.ORIGIN_DIR + str(os.sep) + str(program) + str(os.sep) + str(version)
                if not os.path.exists(curFeaDir):
                    os.mkdir(curFeaDir)
                filters = glob.glob(curBinDir + os.sep + "*.idb")
                filters = filters + (glob.glob(curBinDir + os.sep + "*.i64"))

                for i in filters:
                    if i.endswith("idb"):
                        print( config.IDA32_DIR+" -S\""+config.CODE_DIR+ os.sep + "g1_gen_features.py "+curFeaDir+"  "+i +"  "+ str(program) +"  "+ str(version) +"\"  "+i+"\n\n")
                        os.popen(config.IDA32_DIR+"   -S\""+config.CODE_DIR+ os.sep + "g1_gen_features.py "+curFeaDir+" "+i +"  "+ str(program) +"  "+ str(version)  + "\"  "+i)
                        # subprocess.call(IDA32_DIR+" -S\""+CODE_DIR+"\\2_gen_features.py "+curFeaDir+" "+i +"  "+ str(program) +"  "+ str(version)  + "\"  "+i )
                    else:
                        print( config.IDA64_DIR+" -S\""+config.CODE_DIR+ os.sep + "g1_gen_features.py "+curFeaDir+"  "+i +"  "+ str(program) +"  "+ str(version) +"\"  "+i+"\n\n")
                        os.popen(config.IDA64_DIR+" -S\""+config.CODE_DIR+ os.sep + "g1_gen_features.py "+curFeaDir+" "+i +"  "+ str(program) +"  "+ str(version)  + "\"  "+i)
                        # subprocess.call(IDA64_DIR+" -S\""+CODE_DIR+ os.sep + "2_gen_features.py "+curFeaDir+" "+i +"  "+ str(program) +"  "+ str(version)  + "\"  "+i)

        print( "G2. process dump file")
        print("="*20)
        pro = subprocess.Popen(["python", config.CODE_DIR + os.sep + "g2_remove_duplicate.py"])
        return_code=pro.wait()
### 训练
[T_TRAIN,T0_GEN_DATASET,T1_TRAIN_VULSEEKER_MODEL] = [0,1,1]
if T_TRAIN:
    if T0_GEN_DATASET:
        from t0_train_data_gen import generate_dataset
        print("T0. 生成训练数据集")
        print("="*20)
        generate_dataset()

    if T1_TRAIN_VULSEEKER_MODEL:
        from t1_model_train import train,test
        print("T1. 开始训练模型")
        print("="*20)
        test_flag = 0
        if not test_flag:
            train()
        else:
            test()
### 搜索
[S_SEARCH,S0_GEN_VUL_DATA,S1_RUN_FOR_SEARCH,S2_RES_ANALYSIS] = [1,0,1,1]
if S_SEARCH:
    if S0_GEN_VUL_DATA:
        from s0_gen_vul_data import gen_vul_sheet
        print("S0. 生成漏洞函数表")
        print("="*20)
        gen_vul_sheet(vul_program = 'openssl')
    if S1_RUN_FOR_SEARCH:
        from s1_run_for_search import run_for_search
        print("S1. 扫描漏洞,输出表格")
        print("="*20)
        target_firmware_dir = os.path.join(config.ORIGIN_DIR, 'search_program')
        target_firmware_list = glob.glob(os.path.join(target_firmware_dir,'*'))
        target_firmware_names = [x.split(os.sep)[-1] for x in target_firmware_list]
        for target_firmware in target_firmware_names:
            run_for_search(target_firmware = target_firmware, vul_fun_program = 'openssl', vul_fun_version =  '1.0.1')
            if S2_RES_ANALYSIS:
                from s2_res_analysis import res_analysis
                print("S2. 精简扫描结果表格")
                print("="*20)
                res_analysis(target_firmware = target_firmware, sim_threshold = 0.90)