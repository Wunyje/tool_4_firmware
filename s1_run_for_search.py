import tensorflow as tf
from t0_train_data_gen import read_cfg, read_dfg, read_feature, \
    zero_padded_adjmat, zero_padded_featmat
import pandas as pd
import numpy as np
import glob
import config
import os

def emputy_res_frame(target_firmware = '*', vul_fun_program = '*', vul_fun_version =  '*'):
    _, target_firmware_fun_list, vul_fea_path_select, vul_fun_list = \
            get_fun_list_and_fea(target_firmware = target_firmware, vul_fun_program = vul_fun_program, vul_fun_version =  vul_fun_version)
    
    column_names = ['目标函数名', '漏洞库函数版本']
    for cve_number in vul_fun_list['CVE-number']:  #
            vul_fun = vul_fun_list['Pure Vulnerability Function'][vul_fun_list['CVE-number'] == str(cve_number)]
            vul_fun = vul_fun_list[vul_fun_list['CVE-number'] == str(cve_number)]['Pure Vulnerability Function']
            vul_fun_name = [x for x in vul_fun][0]
            column_names.append((str(cve_number)+':\n'+vul_fun_name))
    res_frame = pd.DataFrame(columns=column_names)
    cve_fun_num = len(vul_fun_list)

    row_num = 0
    target_fun_col = res_frame['目标函数名']
    vul_fun_ver_col = res_frame['漏洞库函数版本']

    for target_fun in target_firmware_fun_list.iloc[:,0]: 
        for fea_dir in vul_fea_path_select:
            if not os.path.exists(fea_dir):
                continue
            else:
                target_fun_col.loc[row_num] = target_fun
                vul_fun_ver_col.loc[row_num] = '_'.join([fea_dir.split(os.sep)[-2],fea_dir.split(os.sep)[-1]])
                row_num = row_num+ 1
    res_frame['目标函数名'] = target_fun_col
    res_frame['漏洞库函数版本'] = vul_fun_ver_col
    return res_frame, cve_fun_num

def get_fun_list_and_fea(target_firmware = '*', vul_fun_program = '*', vul_fun_version =  '*'):
    # 目标固件函数特征路和径函数列表
    target_firmware_fea_path = os.path.join(config.FEA_DIR, 'search_program', target_firmware)
    assert os.path.exists(target_firmware_fea_path),('固件'+target_firmware+'不存在')
    target_firmware_fun_list = pd.DataFrame()
    target_firmware_fun_list_path = glob.glob(os.path.join(target_firmware_fea_path, 'functions_list*' ))
    target_firmware_fun_list = pd.read_csv(target_firmware_fun_list_path[0])

    # 目标固件架构信息
    arch_info_path = target_firmware_fea_path.replace(config.FEA_DIR, config.ORIGIN_DIR)
    arch_info_path = os.path.join(arch_info_path, '*.txt')
    arch_info = ((glob.glob(arch_info_path)[0]).split(os.sep)[-1]).split('.')[0] 

    # 漏洞函数特征
    vul_fea_path_exist = glob.glob(os.path.join(config.FEA_DIR,vul_fun_program,'*','*'))
    vul_fea_path_select = [x for x in vul_fea_path_exist if arch_info in x]
    assert len(vul_fea_path_select)>0,(vul_fun_program+'中无'+arch_info+'架构')
    vul_fea_path_select = [x for x in vul_fea_path_select if vul_fun_version in x]
    assert len(vul_fea_path_select)>0,(vul_fun_program+'中无'+vul_fun_version+'版本')
    
    # 漏洞函数列表
    vul_fun_lists_path = os.path.join(config.VUL_FUN_DIR,vul_fun_program,vul_fun_version,'cve_fun.csv')
    vul_fun_list =  pd.read_csv(vul_fun_lists_path)
    return target_firmware_fea_path, target_firmware_fun_list, vul_fea_path_select, vul_fun_list

def create_generate_targetfun_pairs_generator(target_firmware = '*', vul_fun_program = '*', vul_fun_version =  '*'):
    def generate_targetfun_pairs_generator():   
        target_firmware_fea_path, target_firmware_fun_list, vul_fea_path_select, vul_fun_list = \
            get_fun_list_and_fea(target_firmware = target_firmware, vul_fun_program = vul_fun_program, vul_fun_version =  vul_fun_version)

        for target_fun in target_firmware_fun_list.iloc[:,0]:        
            cfg_target_fun, dfg_target_fun, feat_matrix_target_fun = \
                                    generate_a_func_fea(target_fun, target_firmware_fea_path)
            cfg_target_fun, dfg_target_fun, feat_matrix_target_fun = \
                                    zero_padded_adjmat(cfg_target_fun, config.max_nodes), \
                                    zero_padded_adjmat(dfg_target_fun, config.max_nodes), \
                                    zero_padded_featmat(feat_matrix_target_fun, config.max_nodes) 
            for fea_dir in vul_fea_path_select:  
                for vul_fun in vul_fun_list['Pure Vulnerability Function']:  
                    vul_fun_fea_path = os.path.join(fea_dir, str(vul_fun)+'_fea.csv')
                    if not os.path.exists(vul_fun_fea_path):
                        cfg_vul_fun, dfg_vul_fun, feat_matrix_vul_fun = \
                                            np.zeros((config.max_nodes,config.max_nodes)), \
                                            np.zeros((config.max_nodes,config.max_nodes)), \
                                            zero_padded_featmat(np.zeros(shape=(1,1)), config.max_nodes)
                    else:
                        cfg_vul_fun, dfg_vul_fun, feat_matrix_vul_fun = generate_a_func_fea(str(vul_fun), fea_dir)
                        cfg_vul_fun, dfg_vul_fun, feat_matrix_vul_fun = \
                                                zero_padded_adjmat(cfg_vul_fun, config.max_nodes), \
                                                zero_padded_adjmat(dfg_vul_fun, config.max_nodes), \
                                                zero_padded_featmat(feat_matrix_vul_fun, config.max_nodes)
                    pair = (cfg_target_fun, dfg_target_fun, feat_matrix_target_fun, \
                            cfg_vul_fun, dfg_vul_fun, feat_matrix_vul_fun, 1)
                    yield pair
    return generate_targetfun_pairs_generator

def generate_a_func_fea(funcname, fea_dir):
    cfg = read_cfg(funcname, fea_dir)     # 读相应cfg
    dfg = read_dfg(funcname, fea_dir)     # 读相应dfg
    node_size = len(cfg)        
    if node_size < config.min_nodes_threshold:  
        return  
    feature = read_feature(funcname, fea_dir, node_size)
    assert len(cfg.nodes) == len(dfg.nodes) == feature.shape[0], \
        "binary:%s func:%s cfg:%d dfg:%d and feature:%d_matrix's shape not consistent!"\
        %(fea_dir, funcname, len(cfg.nodes), len(dfg.nodes), feature.shape[0])
    g = (cfg, dfg, feature)  # 得到最终cfg+dfg+块特征
    return g

def generation_4_model(target_firmware = '*', vul_fun_program = '*', vul_fun_version =  '*'):
    generate_targetfun_pairs_generator = create_generate_targetfun_pairs_generator(target_firmware, vul_fun_program, vul_fun_version)
    data = tf.data.Dataset.from_generator(generate_targetfun_pairs_generator, \
                                        output_types=(tf.float32, tf.float32, tf.float32, tf.float32, \
                                                        tf.float32, tf.float32, tf.float32))
    data = data.batch(batch_size=config.mini_batch) # 10
    data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE) # 根据可用的CPU动态设置并行调用的数量
    return data


def calcu_sim(model, g1_cfg_adjmat, g1_dfg_adjmat, g1_featmat, g2_cfg_adjmat, g2_dfg_adjmat, g2_featmat):
    input = (g1_cfg_adjmat, g1_dfg_adjmat, g1_featmat, g2_cfg_adjmat, g2_dfg_adjmat, g2_featmat)
    sim, g1_embedding, g2_embedding = model(input) # sim_score
    if tf.reduce_max(sim) > 1 or tf.reduce_min(sim) < -1:
        sim = sim * 0.999  
    return sim, g1_embedding, g2_embedding


def run_for_search(target_firmware = '*', vul_fun_program = '*', vul_fun_version =  '*'):
    dbg_base = 1000
    dbg_count = 0
    res_frame_DF, cve_fun_num = emputy_res_frame(target_firmware = target_firmware,\
                                    vul_fun_program = vul_fun_program,\
                                    vul_fun_version = vul_fun_version)
    config.mini_batch = cve_fun_num
    config.base_batch = cve_fun_num

    res_frame_res_DF = res_frame_DF.filter(regex='CVE')
    res_frame_res_DF.insert(column='max', loc=len(res_frame_res_DF.columns), value=0)

    target_fun_col = res_frame_DF['目标函数名']
    vul_fun_ver_col = res_frame_DF['漏洞库函数版本']
    res_frame_DF = res_frame_DF.drop(index=res_frame_DF.index)
    
    model = tf.keras.models.load_model(config.T1_VULSEEKER_MODEL_TO_SAVE)
    model_search_dataset = generation_4_model(target_firmware = target_firmware,\
                                              vul_fun_program = vul_fun_program,\
                                              vul_fun_version = vul_fun_version )
    count_num = 0
    res_sav_path = os.path.join(config.RES_DIR,'search_program',target_firmware)
    if not os.path.exists(res_sav_path):
        os.makedirs(res_sav_path)
    for g1_cfg_adjmat,g1_dfg_adjmat,g1_featmat,g2_cfg_adjmat,g2_dfg_adjmat,g2_featmat,y in model_search_dataset:
        sim_res, _, _ = calcu_sim(model, g1_cfg_adjmat,g1_dfg_adjmat,g1_featmat,g2_cfg_adjmat,g2_dfg_adjmat,g2_featmat)
        sim_res = sim_res.numpy()
        y  = (y.numpy()+1)/2
        if sim_res[0] or sim_res[1]:
            sim_res = (sim_res+1)/2
            res_frame_res_DF.loc[count_num] = np.append(sim_res,sim_res.max())
            count_num = count_num + 1
            if (count_num%dbg_base == 0) and (count_num):
                dbg_count = dbg_count + 1
                midres_frame_DF = pd.concat((target_fun_col,vul_fun_ver_col,res_frame_res_DF),axis=1)
                midres_frame_DF.to_excel(os.path.join(res_sav_path, 'midres_frame'+ str(dbg_count)) +'.xlsx', index=False)
                print('输出 midres_frame'+str(dbg_count) +'.xlsx')
    final_res_frame_DF = pd.concat((target_fun_col,vul_fun_ver_col,res_frame_res_DF),axis=1)
    final_res_frame_DF.to_excel(os.path.join(res_sav_path,'res_frame.xlsx'), index=False)
    print('输出 res_frame.xlsx')
    return 

if __name__ == "__main__":
    target_firmware_dir = os.path.join(config.ORIGIN_DIR, 'search_program')
    target_firmware_list = glob.glob(os.path.join(target_firmware_dir,'*'))
    target_firmware_names = [x.split(os.sep)[-1] for x in target_firmware_list]
    for target_firmware in target_firmware_names:
        run_for_search(target_firmware = target_firmware, vul_fun_program = 'openssl', vul_fun_version =  '1.0.1')
