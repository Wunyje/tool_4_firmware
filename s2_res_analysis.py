import pandas as pd
import config
import os
import glob

def res_analysis(target_firmware = '*', sim_threshold = 0.95):
    search_res_path = os.path.join(config.RES_DIR, 'search_program', target_firmware)
    search_res = pd.read_excel(search_res_path + os.sep + 'res_frame.xlsx',sheet_name=0)
    search_res_DF = pd.DataFrame(search_res)
    search_res_DF.rename(columns={'max':'相似值'},inplace=True)

    vul_fun_col = search_res_DF.filter(regex='CVE').idxmax(axis=1) 
    vul_fun_col.rename(index='漏洞库函数', inplace=True)
    max_sim_col= search_res_DF['相似值']
    target_fun_col = search_res_DF['目标函数名']
    vul_fun_ver_col = search_res_DF['漏洞库函数版本']

    rows_max_value_DF = pd.concat((target_fun_col,vul_fun_col , vul_fun_ver_col, max_sim_col),axis=1)
    
    select_cols_max_value_DF = pd.DataFrame(columns=rows_max_value_DF.columns)
    number_2_div = rows_max_value_DF['目标函数名'].value_counts()
    for i in range(0, len(rows_max_value_DF), number_2_div[0]):
        group = rows_max_value_DF.iloc[i:i+number_2_div[0]]

        cols_max_value = group['相似值'].max(axis=0) 

        cols_max_name_DF = group[cols_max_value == group['相似值']]['漏洞库函数']
        cols_max_name = [x for x in cols_max_name_DF][0]

        cols_max_ver_DF = group[cols_max_value == group['相似值']]['漏洞库函数版本']
        cols_max_ver = [x for x in cols_max_ver_DF][0]
        
        cols_target_name = [x for x in group['目标函数名']][0]

        select_cols_max_value_DF.loc[i] = [cols_target_name, cols_max_name,cols_max_ver, cols_max_value]
    select_cols_max_value_DF.sort_values(by='相似值', ascending = 0, inplace = True)
    select_cols_max_value_DF.to_excel(search_res_path + os.sep + 'reduced_res.xlsx', index=False)
    select_cols_max_value_DF.reset_index(drop=True, inplace=True)

    sim_g_mask = select_cols_max_value_DF['相似值'] >= sim_threshold
    sim_g_DF = select_cols_max_value_DF[sim_g_mask]
    sim_g_DF.to_excel(search_res_path + os.sep + 'reduced_res_beyond_%f.xlsx'%sim_threshold, index=False)
    return 0 

if __name__ == '__main__':
    target_firmware_dir = os.path.join(config.ORIGIN_DIR, 'search_program')
    target_firmware_list = glob.glob(os.path.join(target_firmware_dir,'*'))
    target_firmware_names = [x.split(os.sep)[-1] for x in target_firmware_list]
    for target_firmware in target_firmware_names:
        res_analysis(target_firmware = target_firmware, sim_threshold = 0.90)

