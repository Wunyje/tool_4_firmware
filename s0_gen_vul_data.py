import pandas as pd
import re
import os
import glob
import config

def gen_vul_sheet(vul_program = '*', version = ['*']):
    vul_program_paths = glob.glob(config.VUL_FUN_DIR + os.sep +  vul_program  + os.sep +'*.xlsx')
    assert len(vul_program_paths) > 0,(vul_program + '库漏洞信息不存在')
    for vul_program_name in glob.glob(config.VUL_FUN_DIR + os.sep +  vul_program):
        for table_path in glob.glob(vul_program_name+ os.sep + '*.xlsx'):
            # Read in the OpenSSL CVE data from an Excel file
            filename = os.path.join(table_path)
            cve_data = pd.read_excel(filename, sheet_name=None)

            # Concatenate data from all years into a single DataFrame
            cve_DF = pd.concat(cve_data.values(), ignore_index=True, sort=False)
            cve_DF.fillna('NA', inplace=True)

            # Select rows that contain function names
            function_name_mask = cve_DF.iloc[:,1].str.contains('_|:|function') # Vulnerability Function
            cve_fun_DF = cve_DF[function_name_mask]

            # Select rows that are relevant to OpenSSL versions
            if not version == ['*']:
                version_mask = cve_fun_DF.iloc[:,2].str.contains('|'.join(version)) # Affected Version
                cve_fun_DF = cve_fun_DF[version_mask]

            # Read in the list of function names from a CSV file
            loc_fun_lists = glob.glob(vul_program_name.replace(config.VUL_FUN_DIR, config.FEA_DIR)\
                                        + os.sep +'*'+ os.sep + '*'+ os.sep + 'functions_list*')
            for v in version:
                if not v == '*':
                    loc_fun_list_select = [x for x in loc_fun_lists if v in x]
                else:
                    loc_fun_list_select = loc_fun_lists
                assert (len(loc_fun_list_select) > 0),('\n'+\
                    vul_program_name.split(os.sep)[-1]+'-' + v +'无文件')
                
                # select biggest file
                file_sizes = [os.path.getsize(file_path) for file_path in loc_fun_list_select]
                max_index = file_sizes.index(max(file_sizes))
                max_file_path = loc_fun_list_select[max_index]

                fun_list = pd.read_csv(max_file_path, header=None)
                fun_name_list = fun_list[0]

                # Select rows that match a function name in the list
                function_name_regex = re.compile(r'(\w+_\w+)+')
                cve_fun_DF['Pure Vulnerability Function'] = cve_fun_DF.iloc[:,1].str.extract(function_name_regex)[0]
                mask = cve_fun_DF['Pure Vulnerability Function'].isin(fun_name_list.astype(str))
                real_cve_fun_DF = cve_fun_DF[mask]

                # Remove duplicate rows and unnecessary columns
                real_cve_fun_DF.drop_duplicates(subset='CVE-number', inplace=True)
                cve_fun_DF = real_cve_fun_DF[['CVE-number', 'Pure Vulnerability Function']]

                if not v == '*':
                    sheet_save_path = os.path.join(vul_program_name, v)
                else:
                    sheet_save_path = vul_program_name
                if not os.path.exists(sheet_save_path):
                    os.mkdir(sheet_save_path)
                cve_fun_DF.to_csv(sheet_save_path + os.sep +  'cve_fun.csv', index=False)
    return 0

if __name__ == '__main__':
    gen_vul_sheet(vul_program = 'openssl')#,version = ['1.0.1','1.0.2'])
