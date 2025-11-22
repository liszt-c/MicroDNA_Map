# MicroDNA Map

# 安装
推荐使用 Conda 进行安装，建议 Python 版本为 3.9。
``` 
# 创建python环境
conda create -n microdna python==3.9
conda activate microdna

# 安装部分依赖包
pip install -r requirements.txt

# 手动安装 PyTorch，建议版本 ≥ 1.8，cuda>11.7
# 查看https://pytorch.org/get-started/previous-versions/，选择适合的pytorch版本
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# CNVkit安装，请参考 https://github.com/etal/cnvkit 以确保安装成功
# 提示：测试表明 CNVkit 在 Python 3.9 环境下兼容性最佳
pip install cnvkit
``` 

# 快速测试
``` 
conda activate YOUR_ENV_NAME
cd YOUR_DIR_PATH
python run.py --pattern long_segment
``` 
以上命令将自动处理./identify/long_segment_to_identify/目录下的 .fa文件，计算其中的microDNAs，并将结果输出到同一目录。
若 .fa 文件头部包含正确的染色体位置信息，结果将包含 .bed 定位文件与 .fasta 序列文件。
``` 
# 查看 run.py 的更多选项
python run.py -h
``` 

# 从Fastq开始运行

1. 将hg19.fa参考基因组文件放入./cnvkit_do目录

2. 将配对的*.fastq1和*.fastq2文件放入./cnvkit_do (可通过  fasterq-dump --split-3 *.sra 从 SRA 文件转换得到)

3. 在已安装依赖的 Conda 环境中运行 MicroDNA_Map_batch.py
``` 
python MicroDNA_Map_batch.py
``` 

#### 该计算包含的流程简述：
cnvkit_tool.py 执行步骤：

0. 查询 ./cnvkit_do 中所有 fastq 文件
1. 检查 hg19.fa 的索引，如果不存在则使用 bowtie2 构建
2. 验证 FASTQ 文件完整性
3. 使用 Bowtie2 将 FASTQ 序列比对至参考基因组
4. 调用 CNVkit 计算样本的 CNVs

cnvkit_run.py 执行步骤：

0. 检查上一步文件是否完整
1. 读取 CNVkit 结果文件
2. 使用 samtools 对 CNVs 片段进行切割提取
3. 调用 run.py 进行逐次计算

整理合并结果并清理中间文件

# 模型训练
从 NCBI GEO 数据库中下载 *_RAW 文件，提取 microDNA 序列位置信息<br />
使用 ./preprocessing/count*.py 提取 microDNA 序列<br />
使用 ./preprocessing/cout_other*.py 提取 otherDNA序列 <br />
这些脚本将调用 SamtoolBash.sh 完成序列切割<br />

注意：不同数据集可能存在格式差异，建议根据实际情况手动调整预处理脚本参数。<br />

1. 将数据放入./datasets目录中
2. 在 train.py 中调整训练参数（如迭代次数、学习率策略等）
3. 运行训练脚本: 
``` 
python train.py
``` 
#### 独立测试
将用于测试的测试数据放入./datasets，运行以下命令获取评估指标：
``` 
python verification.py
``` 
随后可通过以下命令绘制 ROC 曲线：
``` 
python ROC_draw.py
``` 

#### 其他文件说明
./Additional_tools 包含生物信息学分析工具，但代码未针对通用性优化，仅供参考。

./other_model 包含实验性模型代码，使用前请仔细阅读代码细节。
