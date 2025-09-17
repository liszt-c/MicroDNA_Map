# V11.8 2025-9-1

'''
###需求
1.输入原始测序文件
文件转化路径：SRR-fastq-bam-callpeaks
SAMtools-bowtie2-MACS2
设置识别阈值，strict，normal，relax

2.对于长序列文件例如fasta文件，注意跳过>

3.手动输入序列

###结构

1.识别核
2.在输入中滑动窗口，每次调用识别核
3.检测输入长度，大于400bp调用滑动窗口，小于400bp使用补充噪音(直接调用datapeocess)直接输出检测结果
4.长片段得出的是(400-10)/10长度的数据，每40个一组再次滑动窗口找到窗口和大于20的区域，越大可能性越高
5.callpeaks结果的提取与预测，将序列位置，预测结果写入表格中

'''
import argparse
import torch
import torchvision
import os
import sys
import numpy
import random
import glob
import torch.nn.functional as nn
torch.set_printoptions(profile="full")
#from ResNet_Attention import ResNetAttention
from ResAttention import ResNetSelfAttention
import datetime
from tqdm import tqdm

# Slide the window for the first time
norm_length = 400
STEP1 = 10
# Slide the window for the second time
WINDOW2 = 10
STEP2 = 3


time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(time)

def dataprocess(seq_clear):
    n = len(seq_clear)-1
    if n <norm_length:                                     ###前后噪音填充
        data = numpy.zeros((4,norm_length), dtype = int)
        fund = int((norm_length - n)/2 )
        for j in range(fund):
            x = random.randint(1,4)
            local = x % 4
            data[local,j] = 1
        for i in range(n):
            if (seq_clear[i]=='A' or seq_clear[i]=='a'):
                data[0,fund+i]  = 1
            if (seq_clear[i]=='T' or seq_clear[i]=='t'):
                data[1,fund+i]  = 1
            if (seq_clear[i]=='C' or seq_clear[i]=='c'):
                data[2,fund+i]  = 1
            if (seq_clear[i]=='G' or seq_clear[i]=='g'):
                data[3,fund+i]  = 1
        for ww in range(norm_length - n - fund):
            x = random.randint(1,4)
            local = x % 4
            data[local,fund+n+ww] = 1
    if n >= norm_length:
        data = numpy.zeros((4,n), dtype = int)
        for i in range(n):
            if (seq_clear[i]=='A' or seq_clear[i]=='a'):
                data[0,i]  = 1
            if (seq_clear[i]=='T' or seq_clear[i]=='t'):
                data[1,i]  = 1
            if (seq_clear[i]=='C' or seq_clear[i]=='c'):
                data[2,i]  = 1
            if (seq_clear[i]=='G' or seq_clear[i]=='g'):
                data[3,i]  = 1
    return data,n


def dataprocess2(file_name):    # V11.7
    #print(file_name)
    with open(file_name, "r", encoding='utf-8') as DNA_seq:
        seqs = DNA_seq.readlines()
    
    ### 解析头部信息
    header = ""
    seq_clear = ''
    for seq in seqs:
        if seq.strip().startswith(">"):
            header = seq.strip()
        else:
            seq_clear += seq.strip()
    
    ### 解析染色体位置信息
    chrom_info = {}
    if ":" in header and "-" in header:
        try:
            chrom_part, pos_part = header.split(":", 1)
            chrom = chrom_part.replace(">", "").strip()
            start_pos, end_pos = pos_part.split("-")
            chrom_info = {
                "chrom": chrom,
                "start": int(start_pos),
                "end": int(end_pos)
            }
            print(f"检测到染色体位置信息: {chrom}:{start_pos}-{end_pos}")
        except Exception as e:
            print(f"无法解析头部信息: {header}, 错误: {e}")
    
    ### 处理序列数据
    seq_clear = seq_clear.replace(" ", "").replace("\n", "")
    data, n = dataprocess(seq_clear)
    return data, n, seq_clear, chrom_info



def Recognition_kernel(input_seq,model): 
    ###dataprocess输出是(4,400)，需要变为(1,4,400),torch.unsqueeze用于升维
    input_seq = torch.unsqueeze(input_seq, dim=0)
    #print(input_seq)
    if torch.cuda.is_available():
        input_seq = input_seq.cuda()
    outputs = model(input_seq)
    #print(outputs)
    result = torch.nn.functional.softmax(outputs,dim = -1)
    #print(result,'\n')
    ###(1,0)是eccDNA,(0,1)是otherDNA
    A = result[0,0].item()
    B = result[0,1].item()
    return A-B,A



def forecast(DNA_matrix,n,model):
    if (n<=400):
        inpute = torch.from_numpy(numpy.asarray(DNA_matrix)).float()
        result,prob = Recognition_kernel(inpute,model)
    if(n>400):              ###以20bp为窗口滑动,最后一个窗口滑动距离小于20bp
        sum = 0
        inpute = DNA_matrix[:,n-400:n]
        inpute = torch.from_numpy(numpy.asarray(inpute)).float()
        outputs,probability = Recognition_kernel(inpute,model)
        max = outputs
        prob = probability
        if(n-400>20):
            for i in range(int((n-400)/20)):    #int向下取整
                inpute = DNA_matrix[:,(i*20):(400+i*20)]
                inpute = torch.from_numpy(numpy.asarray(inpute)).float()
                outputs,probability = Recognition_kernel(inpute,model)
                sum = sum + outputs
                max = outputs
                prob = probability
                if(outputs>max):
                    max = outputs
                    prob = probability
        result = max
    return result,prob



def fasta_base(file_name,model):
    DNA_matrix,n,_ = dataprocess2(file_name)      #(4,n)矩阵
    result,prob = forecast(DNA_matrix,n,model)
    return result,prob


def manual_base(seq,model):
    DNA_matrix,n = dataprocess(seq)      #(4,n)矩阵
    result,prob = forecast(DNA_matrix,n,model)
    return result,prob




def long_segment(file_name, model, limit, batch_size=256, max_segment_length=1000000, min_region_length=150):   # v11.7 添加染色体位置支持
    limit = float(limit)
    DNA_matrix, n, seq_clear, chrom_info = dataprocess2(file_name)  # 现在返回chrom_info
    print('输入长度：', n)
    if n < 800:
        print('Length less than 800bp, short sequence identification is better!')
        return (0, 0, 0, 0, chrom_info)  # 返回chrom_info
    
    # 计算窗口数量
    num_windows = int((n - 400) / STEP1) + 1
    if (n - 400) % STEP1 != 0:
        num_windows += 1
    
    print(f"准备处理 {num_windows} 个窗口...")
    
    # 初始化结果列表
    sliding_block = []
    
    # 确定序列是否需要分段处理
    segment_length = max_segment_length
    num_segments = max(1, (n + segment_length - 1) // segment_length)
    
    print(f"序列将被分成 {num_segments} 个段进行处理...")
    
    # 处理每个序列段
    for seg_idx in range(num_segments):
        seg_start = seg_idx * segment_length
        seg_end = min((seg_idx + 1) * segment_length + 400, n)  # 添加400bp的重叠区域
        
        print(f"处理段 {seg_idx+1}/{num_segments}: 位置 {seg_start}-{seg_end}")
        
        # 计算当前段的窗口数量
        seg_num_windows = int((seg_end - seg_start - 400) / STEP1) + 1
        if (seg_end - seg_start - 400) % STEP1 != 0:
            seg_num_windows += 1
        
        # 准备当前段的窗口数据
        inputs = []
        for i in tqdm(range(seg_num_windows), desc=f"创建段 {seg_idx+1} 的窗口", unit="window"):
            start = seg_start + i * STEP1
            end = min(start + 400, seg_end)
            window_data = DNA_matrix[:, start:end]
            
            # 如果窗口不足400bp，填充随机噪音
            if end - start < 400:
                padding = numpy.zeros((4, 400 - (end - start)), dtype=int)
                for j in range(padding.shape[1]):
                    x = random.randint(1, 4)
                    padding[x % 4, j] = 1
                window_data = numpy.concatenate([window_data, padding], axis=1)
            
            inputs.append(torch.from_numpy(window_data).float())
        
        # 批量处理当前段的窗口
        seg_num_batches = (seg_num_windows + batch_size - 1) // batch_size
        
        # 添加进度条：模型推理
        for i in tqdm(range(0, seg_num_windows, batch_size), 
                      desc=f"处理段 {seg_idx+1} 的批次", 
                      unit="batch", 
                      total=seg_num_batches):
            batch_end = min(i + batch_size, seg_num_windows)
            batch = torch.stack(inputs[i:batch_end])
            if torch.cuda.is_available():
                batch = batch.cuda()
            
            with torch.no_grad():
                outputs = model(batch)
                # 确保使用浮点数计算概率
                probabilities = torch.nn.functional.softmax(outputs, dim=-1)[:, 0].cpu().numpy().astype(numpy.float32)
            
            sliding_block.extend(probabilities.tolist())
        
        # 释放当前段的内存
        del inputs
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 后续处理部分
    block_length = len(sliding_block)
    # print(sliding_block)
    sliding_block = numpy.array(sliding_block, dtype=numpy.float32)  # 确保使用浮点数
    print(f'处理完成，得到 {block_length} 个概率值')
    
    # 计算第二次滑动窗口的平均值
    num_second_windows = int((block_length - WINDOW2) / STEP2) + 1
    if (block_length - WINDOW2) % STEP2 != 0:
        num_second_windows += 1
    
    print(f"进行第二次滑动窗口处理 ({num_second_windows} 个窗口)...")
    
    summ1 = numpy.zeros(num_second_windows, dtype=numpy.float32)  # 使用浮点数数组
    # 添加第三个进度条：第二次滑动窗口
    for j in tqdm(range(num_second_windows), desc="第二次滑动窗口", unit="window"):
        start = j * STEP2
        end = min(start + WINDOW2, block_length)
        window_sum = sliding_block[start:end].sum()
        window_size = end - start
        # 确保使用浮点数除法
        summ1[j] = float(window_sum) / window_size
        # print(float(window_sum) / window_size)

    # 打印统计信息以帮助调试
    print(f"summ1 统计信息: min={numpy.min(summ1)}, max={numpy.max(summ1)}, mean={numpy.mean(summ1)}")
    
    # 阈值处理 - 使用浮点数比较
    summ2 = numpy.where(summ1 >= limit, 1, 0)

    '''
    # 将二值化数组输出到文件
    numpy.savetxt(f"{file_name}_binary.txt", summ2, fmt='%d')
    print(f"已将二值化数组输出到 {file_name}_binary.txt")
    '''

    # 检测连续区域
    left = []
    right = []
    in_region = False
    
    print("检测连续区域...")
    # 添加第四个进度条：检测连续区域
    for idx, val in tqdm(enumerate(summ2), desc="检测区域", total=len(summ2)):
        if val == 1 and not in_region:
            in_region = True
            left.append(idx)
        elif val == 0 and in_region:
            in_region = False
            right.append(idx)
    
    if in_region:
        right.append(len(summ2) - 1)
    
    # 确保左右边界数量匹配
    if len(left) != len(right):
        print(f"警告：左右边界数量不匹配！左边界数: {len(left)}, 右边界数: {len(right)}")
        # 取最小数量以确保安全
        min_length = min(len(left), len(right))
        left = left[:min_length]
        right = right[:min_length]
    
    # 过滤掉长度小于阈值的区域
    print(f"过滤掉长度小于 {min_region_length}bp 的区域...")
    filtered_left = []
    filtered_right = []
    
    # 计算每个区域的长度（以bp为单位）
    # 注意：每个索引代表30bp（因为第一次滑动窗口步长10bp，第二次滑动窗口步长3个窗口）
    filtered_cout = 0
    for i in range(len(left)):
        region_length = (right[i] - left[i]) * 30  # 转换为bp单位
        
        if region_length >= min_region_length:
            filtered_left.append(left[i])
            filtered_right.append(right[i])
        else:
            filtered_cout = filtered_cout + 1
    print(f" {filtered_cout} 个过小区域被过滤")
    
    left = filtered_left
    right = filtered_right
    quantity = len(left)
    
    print(f"过滤后保留 {quantity} 个候选区域")
    return left, right, quantity, seq_clear, chrom_info  # 返回chrom_info


# v11.7 合并序列文件为FASTA格式
def merge_to_fasta(base_file, quantity):
    # 获取所有候选区域序列文件
    region_files = glob.glob(f"{base_file}_*.txt")
    
    # 如果没有找到文件，直接返回
    if not region_files:
        print(f"警告：没有找到 {base_file}_*.txt 文件")
        return
    
    # 创建合并后的FASTA文件名
    output_file = f"{base_file}_merged.fasta"
    
    # 打开输出文件
    with open(output_file, 'w') as out_f:
        # 按序号排序文件
        sorted_files = sorted(region_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        # 写入每个序列
        for i, file_path in enumerate(sorted_files):
            # 读取序列内容
            with open(file_path, 'r') as in_f:
                sequence = in_f.read().strip()
            
            # 写入FASTA格式
            out_f.write(f">region_{i+1}\n")
            out_f.write(sequence + "\n")
    
    print(f"合并完成：{output_file}")
    
    # 删除临时文件
    for file_path in region_files:
        os.remove(file_path)
    print(f"已删除临时文件!")
    
    return output_file

# V11.7 生成BED文件
def generate_bed_file(base_file, left, right, chrom_info):
    if not chrom_info:
        print("没有染色体位置信息，无法生成BED文件")
        return
    
    bed_file = f"{base_file}.bed"
    with open(bed_file, 'w') as bed_f:
        for i in range(len(left)):
            # 计算染色体上的绝对位置
            chrom_start = chrom_info["start"] + left[i] * 30
            chrom_end = chrom_info["start"] + right[i] * 30
            
            # BED格式是0-based，半开区间 [start, end)
            bed_line = f"{chrom_info['chrom']}\t{chrom_start}\t{chrom_end}\n"
            bed_f.write(bed_line)
    
    print(f"已生成BED文件: {bed_file}")
    return bed_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'RUN')
    parser.add_argument('--pattern', type=str,help='Input Type: short_sequence,long_segment')
    ###短序列可以通过文件输入、手动输入，如果不指明文件或手动输入则使用./fatsa_to_identify文件夹下所有文件
    ###长片段只有一种模式，读取./long_segment_to_identify文件夹下所有文件，同时可以使用阈值
    parser.add_argument('--model',default = "6.pth", type=str,  # "module.pth"
                        help='model name, The model needs to be in the folder ./save')         #导入模型
    parser.add_argument('--file_path',default ='none',type=str,
                        help='If no parameter is provided, all files in the path ./identify/fatsa_to_identify are used.  If it is not in the run folder, you must specify the path')
    parser.add_argument('--manual_input',default ='none',type=str,
                        help='If you are using manual input pattern, you need to enter the sequence with this option.')
    
    parser.add_argument('--limit',default = "0.99",type=str,
                        help='parameter:strict，normal，relax. relax pattern can identify more but with less precision but strict pattern opposite.')
    


    args = parser.parse_args()

    '''
    model_name = './save/'+args.model
    if torch.cuda.is_available()==False:
        model=torch.load(model_name,map_location='cpu')
        print('using cpu！')
    print('Model deployment completed')
    if torch.cuda.is_available():
        print('using cuda！')
        model = torch.load(model_name,map_location='cuda')
    model.eval()
    print(model)
    '''

    model_name = './save/'+args.model
    #model = ResNetAttention()
    model = ResNetSelfAttention()
    if torch.cuda.is_available()==False:
        model.load_state_dict(torch.load(model_name,map_location='cpu'))
        print('using cpu！')
    print('Model deployment completed')
    if torch.cuda.is_available():
        print('using cuda！')
        model = model.cuda()
        model.load_state_dict(torch.load(model_name,map_location='cuda'))
    model.eval()
    #print(model)


###If no files specified, all files will be used
    if(args.pattern=='short_sequence'):
        if(args.file_path == 'none'):
            print('use all fatsa_files!')
            all_DNA_path = glob.glob(r'./identify/fatsa_to_identify/*.fa')
            file1 = open("./identify/result_out.txt",'w')
            for DNA in all_DNA_path:
                result_,prob_ = fasta_base(DNA,model)
                #写入表格中
                write_line =DNA+'   '+str(result_)+'    '+str(prob_)
                file1.writelines(write_line+'\n')
                print(DNA)
            file1.close()

        if(args.file_path != 'none'):
###manual_input
            if(args.manual_input!='none'):
                seq = args.manual_input
                print('Use manual input!')
                result_,prob_ = manual_base(seq,model)
                print('The probability that input is ecc: ',prob_)
###Specified file
            else:
                print('use the fasta file:',args.file_path)
                result_,prob_ = fasta_base(args.file_path,model)
                print('The probability that ',args.file_path,' is ecc: ',prob_)

###对于长片段的识别
    '''
    1.使用两层滑动窗口，400bp窗口、10bp步长滑动后，得到(seq_lenth-400)/10的可能性序列
    2.设置阈值，高于n(0-100%)
    3.从滑动结果入手，再次滑动窗口，以10bp滑动
        找到第一个大于n-0.1的窗口，滑动5bp如果仍然大于阈值就计数
        直到连续两个窗口小于阈值

    细节：
    1.只有基于文件，一个fa文件创建一个表格来输出结果
    2.读取文件，先向量化再切片，输入(4,n)的矩阵，使用seq[:,x:x+400]切片
    3.复用dataprocess()进行向量化，复用forecast()函数进行400bp识别
    '''

    if(args.pattern=='long_segment'):
        print('long segment pattern!')
        print('use all fatsa_files in ./long_segment_to_identify!')
        if(args.file_path == 'none'):
            print('use all fatsa_files!')
            all_DNA_path = glob.glob(r'./identify/long_segment_to_identify/*.fa')
        else:
            all_DNA_path = glob.glob(str(args.file_path)+'/*.fa')
        for DNA in all_DNA_path:
            left, right, quantity, seq_clear, chrom_info = long_segment(DNA, model, args.limit)
            if quantity == 0:
                continue
            file2 = open("{}.txt".format(DNA), 'w')
            debug = '-1'
            # 写入表格中
            write_line1 = DNA + '   ' + 'The quantity of MicroDNA: ' + str(quantity)
            file2.writelines(write_line1 + '\n')
            print(write_line1)
            
            # V11.7 生成BED文件（如果有染色体位置信息）
            if chrom_info:
                generate_bed_file(DNA, left, right, chrom_info)
            
            # V11.6修复：使用 range(quantity) 而不是 range(quantity+1)
            for wsoe in range(quantity):
                if right[wsoe] != -1 and str(left[wsoe]) != debug:
                    write_line2 = 'relative location:' + '    ' + str(left[wsoe] * 30) + '  ' + str(right[wsoe] * 30)
                    debug = str(left[wsoe])
                    file2.writelines(write_line2 + '\n')

                    fileseq = open("{}_{}.txt".format(DNA, str(wsoe + 1)), 'w')
                    write_line_seq = str(seq_clear[left[wsoe] * 30:right[wsoe] * 30])
                    fileseq.writelines(write_line_seq + '\n')
                    fileseq.close()

            print(DNA)
            file2.close()

            # 合并所有候选区域序列到一个FASTA文件
            merge_to_fasta(DNA, quantity)         
            
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(time)
