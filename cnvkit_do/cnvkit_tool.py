import os
import sys
from glob import glob
import argparse
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RUN CNVkit analysis pipeline')
    parser.add_argument('--thread', default="12", type=str,
                        help='Thread numbers in format "--thread N"')
    parser.add_argument('--FileName', required=True, type=str,
                        help='File base name, such as SRR22924940')
    parser.add_argument('--reference', default="hg19", type=str,
                        help='Reference genome name')
    parser.add_argument('--limit', default="0.95", type=str,
                        help='Strictness parameter: strict, normal, relax')
    args = parser.parse_args()

    # 使用传入的文件名
    SeqName = args.FileName
    thread_arg = args.thread
    
    print(f"开始处理样本: {SeqName}")
    print(f"使用线程参数: {thread_arg}")
    print(f"参考基因组: {args.reference}")
    
    # 检查并构建索引（如果不存在）
    index_files = glob(f'{args.reference}.*.bt2')
    if not index_files:
        print(f"未找到{args.reference}的bowtie2索引，正在构建...")
        cmd0 = f'bowtie2-build -f {args.reference}.fa {args.reference}'
        print(cmd0)
        result = subprocess.run(cmd0, shell=True)
        if result.returncode != 0:
            print(f"索引构建失败: {cmd0}")
            sys.exit(1)
    
    # 检查fastq文件是否存在
    fastq1 = f"{SeqName}_1.fastq"
    fastq2 = f"{SeqName}_2.fastq"
    
    if not os.path.exists(fastq1) or not os.path.exists(fastq2):
        print(f"错误: 找不到fastq文件 {fastq1} 或 {fastq2}")
        sys.exit(1)
    
    # 比对步骤
    bam_file = f"{SeqName}.bam"
    cmd2 = f'bowtie2 -p {thread_arg} -x {args.reference} -1 {fastq1} -2 {fastq2} | samtools sort -@{thread_arg} -o {bam_file}'
    print(cmd2)
    result = subprocess.run(cmd2, shell=True)
    if result.returncode != 0:
        print(f"比对失败: {cmd2}")
        sys.exit(1)
    
    # CNVkit分析步骤
    # 确保输出目录存在
    os.makedirs('.//out', exist_ok=True)
    
    # 批量处理
    cmd3 = f'cnvkit.py batch -m wgs -r {args.reference}_cnvkit_filtered_ref.cnn -p {thread_arg} -d ./out/ {bam_file}'
    print(cmd3)
    result = subprocess.run(cmd3, shell=True)
    if result.returncode != 0:
        print(f"CNVkit批处理失败: {cmd3}")
        sys.exit(1)
    
    # 分段分析
    cnr_file = f"./out/{SeqName}.cnr"
    if not os.path.exists(cnr_file):
        print(f"错误: 找不到CNR文件 {cnr_file}")
        sys.exit(1)
    
    cmd4 = f'cnvkit.py segment {cnr_file} -p {thread_arg} -m cbs -o ./out/result.cns'
    print(cmd4)
    result = subprocess.run(cmd4, shell=True)
    if result.returncode != 0:
        print(f"分段失败: {cmd4}")
        sys.exit(1)
    
    # 调用结果
    cmd5 = f'cnvkit.py call ./out/result.cns -o ./out/result.call.cns'
    print(cmd5)
    result = subprocess.run(cmd5, shell=True)
    if result.returncode != 0:
        print(f"调用失败: {cmd5}")
        sys.exit(1)
    
    print(f"样本 {SeqName} 处理完成")