import os
import glob
import shutil
import subprocess

# 清理开关 - 设置为True时删除fastq和bam文件，并保留fa和out目录
# 设置为False时移动fa和out目录到样本文件夹
CLEANUP = True

def get_optimal_threads():
    """获取最优线程数，留出1个核心给系统"""
    total_cpus = os.cpu_count()
    if total_cpus is None:
        return 8  # 如果无法检测，使用8线程
    return max(1, total_cpus - 1)  # 至少保留1个线程

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

def merge_bed_files(base_name, cnvkit_dir):
    """合并fa目录下的所有.bed文件，并根据CLEANUP设置处理中间文件"""
    fa_dir = os.path.join(cnvkit_dir, "fa")
    out_dir = os.path.join(cnvkit_dir, "out")
    
    # 检查fa目录是否存在
    if not os.path.exists(fa_dir):
        print(f"警告: fa目录不存在: {fa_dir}")
        return None
    
    # 获取fa目录中的所有.bed文件
    bed_files = glob.glob(os.path.join(fa_dir, "*.bed"))
    
    if not bed_files:
        print(f"警告: 在 {fa_dir} 中没有找到.bed文件")
        # 根据CLEANUP设置处理空目录
        if CLEANUP:
            # 如果CLEANUP为True，删除空目录
            try:
                shutil.rmtree(fa_dir)
                if os.path.exists(out_dir):
                    shutil.rmtree(out_dir)
            except Exception as e:
                print(f"删除目录失败: {e}")
        else:
            # 如果CLEANUP为False，移动空目录到样本文件夹
            sample_dir = os.path.join(cnvkit_dir, base_name)
            os.makedirs(sample_dir, exist_ok=True)
            try:
                if os.path.exists(fa_dir):
                    sample_fa_dir = os.path.join(sample_dir, "fa")
                    shutil.move(fa_dir, sample_fa_dir)
                if os.path.exists(out_dir):
                    sample_out_dir = os.path.join(sample_dir, "out")
                    shutil.move(out_dir, sample_out_dir)
            except Exception as e:
                print(f"移动目录失败: {e}")
        return None
    
    # 创建样本目录
    sample_dir = os.path.join(cnvkit_dir, base_name)
    os.makedirs(sample_dir, exist_ok=True)
    
    # 创建合并后的bed文件名
    merged_bed_file = os.path.join(sample_dir, f"{base_name}_microDNA.bed")
    
    print(f"合并 {len(bed_files)} 个.bed文件到: {merged_bed_file}")
    
    # 合并所有.bed文件
    with open(merged_bed_file, 'w') as outfile:
        for i, bed_file in enumerate(bed_files):
            # 添加分隔线和文件名作为注释
            # outfile.write(f"# === 来源: {os.path.basename(bed_file)} ===\n")
            
            # 写入文件内容
            with open(bed_file, 'r') as infile:
                outfile.write(infile.read())
            '''
            # 文件间添加空行分隔
            if i < len(bed_files) - 1:
                outfile.write("\n")
            '''

    print(f"已创建合并文件: {merged_bed_file}")
    
    # 根据CLEANUP设置处理中间文件目录
    if CLEANUP:
        # 如果CLEANUP为True，删除中间文件目录
        try:
            shutil.rmtree(fa_dir)
            print(f"已删除fa目录: {fa_dir}")
            
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
                print(f"已删除out目录: {out_dir}")
        except Exception as e:
            print(f"删除目录失败: {e}")
    else:
        # 如果CLEANUP为False，移动中间文件目录到样本文件夹
        try:
            if os.path.exists(fa_dir):
                sample_fa_dir = os.path.join(sample_dir, "fa")
                shutil.move(fa_dir, sample_fa_dir)
                print(f"已将fa目录移动到: {sample_fa_dir}")
            
            if os.path.exists(out_dir):
                sample_out_dir = os.path.join(sample_dir, "out")
                shutil.move(out_dir, sample_out_dir)
                print(f"已将out目录移动到: {sample_out_dir}")
        except Exception as e:
            print(f"移动目录失败: {e}")
    
    return merged_bed_file

def delete_intermediate_files(base, fastq1, fastq2, cnvkit_dir):
    """删除中间文件（fastq和bam）"""
    '''
    # 删除fastq文件
    for fq in [fastq1, fastq2]:
        if os.path.exists(fq):
            try:
                os.remove(fq)
                print(f"已删除: {fq}")
            except Exception as e:
                print(f"删除 {fq} 失败: {e}")
    '''
    
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
    
    # 自动检测最优线程数
    optimal_threads = get_optimal_threads()
    print(f"检测到 {os.cpu_count()} 个CPU核心，使用 {optimal_threads} 个线程")
    print(f"CLEANUP 设置: {CLEANUP} (True=删除fa/out目录, False=移动fa/out目录到样本文件夹)")
    
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
        
        # 运行cnvkit_tool.py - 使用自动检测的线程数
        print("运行cnvkit_tool.py...")
        cmd = f"python cnvkit_tool.py --FileName {base} --thread {optimal_threads}"
        subprocess.run(cmd, shell=True, check=True, cwd=cnvkit_dir)
        
        # 运行cnvkit_run.py - 在主目录中运行
        if os.path.exists(cnvkit_run):
            print("运行cnvkit_run.py...")
            subprocess.run(f"python {cnvkit_run}", shell=True, check=True)
        else:
            print(f"警告: 未找到 {cnvkit_run}，跳过执行")
        
        # 合并bed文件并根据CLEANUP设置处理中间文件
        print("合并bed文件并处理中间文件...")
        merged_bed = merge_bed_files(base, cnvkit_dir)
        if merged_bed:
            print(f"合并后的bed文件已保存到: {merged_bed}")
        else:
            print(f"样本 {base} 没有生成bed文件")
        
        # 清理中间文件（如果开关开启）
        if CLEANUP:
            print("清理中间文件...")
            delete_intermediate_files(base, fastq1, fastq2, cnvkit_dir)
    
    print("\n所有样本处理完成")

if __name__ == "__main__":
    main()