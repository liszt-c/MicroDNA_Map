import os
import glob
import shutil
import subprocess

# 清理开关 - 设置为True时删除fastq和bam文件
CLEANUP = True

def clean_directory(directory):
    """清空指定目录中的所有内容"""
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"删除 {file_path} 失败: {e}")

def find_fastq_pairs(directory):
    """查找配对的fastq文件"""
    fastq_files = glob.glob(os.path.join(directory, "SRR*_1.fastq"))
    pairs = []
    
    for fastq1 in fastq_files:
        # 获取基础文件名（去掉_1.fastq部分）
        base = os.path.basename(fastq1).split('_')[0]
        fastq2 = os.path.join(directory, f"{base}_2.fastq")
        
        if os.path.exists(fastq2):
            pairs.append((base, fastq1, fastq2))
        else:
            print(f"警告: 找不到配对文件 {fastq2}")
    
    return pairs

def organize_results(base_name, cnvkit_dir):
    """将结果文件组织到样本名命名的子文件夹中"""
    sample_dir = os.path.join(cnvkit_dir, base_name)
    os.makedirs(sample_dir, exist_ok=True)
    
    # 移动fa目录内容
    fa_dir = os.path.join(cnvkit_dir, "fa")
    if os.path.exists(fa_dir) and os.listdir(fa_dir):
        sample_fa_dir = os.path.join(sample_dir, "fa")
        shutil.move(fa_dir, sample_fa_dir)
        print(f"已将fa目录移动到: {sample_fa_dir}")
    
    # 移动out目录内容
    out_dir = os.path.join(cnvkit_dir, "out")
    if os.path.exists(out_dir) and os.listdir(out_dir):
        sample_out_dir = os.path.join(sample_dir, "out")
        shutil.move(out_dir, sample_out_dir)
        print(f"已将out目录移动到: {sample_out_dir}")
    
    return sample_dir

def delete_intermediate_files(base, fastq1, fastq2, cnvkit_dir):
    """删除中间文件（fastq和bam）"""
    # 删除fastq文件
    for fq in [fastq1, fastq2]:
        if os.path.exists(fq):
            try:
                os.remove(fq)
                print(f"已删除: {fq}")
            except Exception as e:
                print(f"删除 {fq} 失败: {e}")
    
    # 删除bam文件及其索引
    bam_files = [
        os.path.join(cnvkit_dir, f"{base}.bam"),
        os.path.join(cnvkit_dir, f"{base}.bam.bai")
    ]
    for bam in bam_files:
        if os.path.exists(bam):
            try:
                os.remove(bam)
                print(f"已删除: {bam}")
            except Exception as e:
                print(f"删除 {bam} 失败: {e}")

def main():
    # 设置路径
    cnvkit_dir = "cnvkit_do"
    cnvkit_tool = os.path.join(cnvkit_dir, "cnvkit_tool.py")
    cnvkit_run = "cnvkit_run.py"  # 当前目录下的cnvkit_run.py
    
    # 查找所有配对的fastq文件
    fastq_pairs = find_fastq_pairs(cnvkit_dir)
    
    if not fastq_pairs:
        print("未找到配对的fastq文件")
        return
    
    # 处理每个配对的fastq文件
    for base, fastq1, fastq2 in fastq_pairs:
        print(f"\n处理样本: {base}")
        
        # 重建fa和out目录
        fa_dir = os.path.join(cnvkit_dir, "fa")
        out_dir = os.path.join(cnvkit_dir, "out")
        
        # 确保目录存在且为空
        if os.path.exists(fa_dir):
            shutil.rmtree(fa_dir)
        os.makedirs(fa_dir, exist_ok=True)
        
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        
        # 运行cnvkit_tool.py - 在cnvkit_dir子目录中运行
        print("运行cnvkit_tool.py...")
        cmd = f"python cnvkit_tool.py --FileName {base} --thread 24"
        subprocess.run(cmd, shell=True, check=True, cwd=cnvkit_dir)
        
        # 运行cnvkit_run.py - 在主目录中运行
        if os.path.exists(cnvkit_run):
            print("运行cnvkit_run.py...")
            subprocess.run(f"python {cnvkit_run}", shell=True, check=True)
        else:
            print(f"警告: 未找到 {cnvkit_run}，跳过执行")
        
        # 组织结果文件到样本目录
        print("组织结果文件...")
        sample_dir = organize_results(base, cnvkit_dir)
        print(f"结果已保存到: {sample_dir}")
        
        # 清理中间文件（如果开关开启）
        if CLEANUP:
            print("清理中间文件...")
            delete_intermediate_files(base, fastq1, fastq2, cnvkit_dir)
    
    print("\n所有样本处理完成")

if __name__ == "__main__":
    main()