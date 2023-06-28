##!/usr/bin/python
# -*- coding: UTF-8 -*-
# 将所有的.o文件利用IDA 进行反汇编，得到idb文件？
# .o文件即为二进制格式存储

import config
import os
import subprocess
import glob

for program in config.G0_PORGRAM_ARR:  # openssl
    paths = glob.glob(config.ORIGIN_DIR + str(os.sep) + program + "\\*\\*")  # 0_Libs/openssl
    for file_path in paths:
        if file_path.endswith(".idb") or file_path.endswith(".asm") or file_path.endswith(".i64"):
            # 反汇编文件只保留 .idb、.asm、.i64 三种文件格式
            continue
        if file_path.endswith(".id0") or file_path.endswith(".id1") or file_path.endswith(".id2")\
                or file_path.endswith(".til") or file_path.endswith(".nam") or file_path.endswith(".txt"):
            os.remove(file_path)
        else:
            if os.path.exists(file_path):
                message = os.popen('file '+file_path).read()  #
                if "32" in message or "i386" in message:
                    # if "32" in file_path  or "i386" in file_path :
                    print(config.IDA32_DIR + " -B \"" + file_path+"\"")
                    subprocess.call(config.IDA32_DIR + " -B \"" + file_path+"\"")  # 调用IDA32，最后存储在相应文件路径
                else:
                    print(config.IDA64_DIR + " -B \"" + file_path+"\"")
                    subprocess.call(config.IDA64_DIR + " -B \"" + file_path+"\"")  # 调用IDA64，最后存储在相应文件路径

