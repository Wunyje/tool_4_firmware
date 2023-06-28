# 固件漏洞扫描工具

此固件漏洞扫描工具基于图嵌入模型，模型经样本训练获得二进制相似性检测能力，利用该能力比较目标固件中函数与漏洞库函数的相似性，实现漏洞扫描功能。

## I.使用说明

### 1、构建脚本运行环境和IDA Pro运行环境

#### 脚本运行环境
##### 配置环境
```
conda create -n vulseeker-tf22 python=3.8.0

conda activate vulseeker-tf22

conda install networkx matplotlib numpy scikit-learn cudatoolkit=10.1 cudnn=7.6

pip install tensorflow-gpu==2.3.0

pip uninstall scipy==1.4.1

conda install scipy=1.8.0
```
##### 卸载环境
```
conda remove -n vulseeker-tf22 --all
```
#### IDA pro环境
```
IDA Pro 7.2

Python 2.7：安装在IDA7.2//下

miasm：将反汇编后固件转换为LLVM IR
```
### 2、复制miasm2文件夹到IDA Pro的python目录下

作为第三方库供`IDA Python`调用

### 3、控制并运行`command.py`

`commnd.py` 主要划分为生成，训练和扫描三部分。生成是后两者的基础，训练是扫描的基础。此工具中已包含模型权重，可跳过训练直接开始扫描。

#### 生成：

`g0_gen_idb_file.py`:调用IDA Python反汇编`0_Origins/`中的二进制文件，生成IDA数据库文件，对象包括`0_Origins/`下开源库，如`openssl`，`busybox`，以及目标固件`0_Origins/search_program/`；

`g1_gen_features.py`:调用IDA Python抽取IDA数据库文件信息，生成控制流图.txt文件(cfg, control flow graph)，数据流图.txt文件(dfg, data flow graph)和特征向量.csv文件，统称特征文件，保存到`1_Features/`下对应文件夹中，以便后续将开源库特征文件输入模型进行训练，或将目标固件输入模型进行扫描

`g2_remove_duplicate.py`:移除特征文件重复项

#### 训练

`t0_train_data_gen.py`:读取`1_Features/*`下特征文件，生成训练数据集保存到`2_Dataset/`

`t1_model_train.py`:读取训练集进行训练，训练结果与权重保存至`3_Output/`

#### 扫描

`s0_gen_vul_data.py`:根据`4_Vul_Fun/*/`下开源库cve漏洞信息，结合`1_Features/*/functions_list_*.csv`中已生成函数信息，匹配指定版本的开源库漏洞函数，并生成漏洞函数表格后保存至对应版本的开源库文件夹`4_Vul_Fun/*/*/`。

`s1_run_for_search.py`:加载`3_Output/`下训练好的模型权重，扫描指定固件中的漏洞。根据所指定的`4_Vul_Fun/*/*/`下漏洞函数表格所述信息，提取`1_Features/*`下漏洞函数特征，与`1_Features/search_program/`下固件函数构成函数对，输入模型计算相似值。最终输出所有固件函数与所有漏洞函数（不同版本、架构）相似值表格，存放至`5_Res`对应目录下

`s2_res_analysis.py`:精简上述结果表格，筛选相似度大于指定阈值的函数对。

#### 其他

`config.py`:包含重要路径参数以及训练参数

`config_for_feature.py`:包含生成特征时所需的指令参数

`utils.py`:包含调用IDA Python的常用API

## II.现有不足

### 1、扫描效率问题

由于设计原因，每次扫描要计算包括固件中所有函数与所有编译器、优化选项下漏洞函数的相似值。从开始计算到结果输出需要30多分钟。

### 2、扫描架构受限

特征文件生成需要用到`miasm`模块，而该模块支持架构有限，不支持mips64架构，更遑论车芯架构。但可自行修改编写。

### 3、运行环境不统一

脚本运行环境与IDA Python运行环境不统一。

## III.其他事项：

### 1、生成开源库的二进制文件

自行搜集开源库的对应版本工程，交叉编译后存放至对应`0_Origins/`对应目录下

### 2、搜集开源库cve漏洞信息

本工具中已有的openssl漏洞信息表格`4_Vun_FUn/openssl/openssl-cve.xlsx`，利用`ChatGPT`生成。具体而言，是在cve漏洞网站通过工具获取开源库历年cve漏洞描述文本，使`ChatGPT`按要求提取文本信息，输出表格。要求文字如下：

`Identify the cve number in this paragraph and the description after each cve number, and extract the name of the vulnerability function and the affected Busybox/Openssl version according to the description. If there is no previous content, enter no content. At last, output  CVE-number and corresponding name of the vulnerability function and affected Busybox/Openssl version in table form.：
`

### 3、扫描固件加密模块功能实现

根据已有的相关工作，扫描加密模块亦可用图嵌入网络计算相似度进行，扫描过程也与漏洞扫描类似，差别在于需搜集对应固件架构下的加密样本函数。

## IV.参考文献：
[1]:GENDA-A Graph Embedded Network Based Detection Approach on encryption algorithm, 2022

[2]:基于语义学习的二进制漏洞代码克隆检测, 2019

[3]:VulSeeker A Semantic Learning Based Vulnerability Seeker for Cross-Platform, 2018

[4]:Scalable code clone search for malware analysis, 2015