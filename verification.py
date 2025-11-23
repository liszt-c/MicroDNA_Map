#
#1.读取模型和参数
#2.调用dataloader读取数据、标签
#3.读出来的数据进模型跑，像test中的一样

'''
1.dataprocess、dataloader原样，文件夹中只有eccDNA，测试准确率
2.取训练模块中测试集的部分代码，作为验证
'''
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from dataloader import data_for_run, dataloader  # 导入完整数据加载器
import numpy as np
from torch.optim import lr_scheduler
import argparse
import datetime
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
# 设置matplotlib为非交互式后端，不显示图形
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

BATCH_SIZE = 3072
version = 6
# 引入概率判断阈值
THRESHOLD = 0.5  

###部署模型
from ResAttention import ResNetSelfAttention
model = ResNetSelfAttention()
#from ResNet_Attention_v126 import ResNetAttention
#model = ResNetAttention()
model_name = './save/'+str(version)+'.pth'

if __name__=='__main__':
    # 加载模型参数
    if torch.cuda.is_available():
        print('using cuda！')
        model = model.cuda()
        model.load_state_dict(torch.load(model_name, map_location='cuda'))
    else:
        print('using cpu！')
        model.load_state_dict(torch.load(model_name, map_location='cpu'))
    print('Model deployment completed')
    print(f'使用概率阈值: {THRESHOLD} (≥{THRESHOLD}为eccDNA, <{THRESHOLD}为otherDNA)')
    
    model.eval()

    # 修改：使用完整数据集而不是测试集
    print("使用完整数据集进行评估...")
    full_dataloader = dataloader(BATCH_SIZE)  # 获取完整数据集的DataLoader
    
    # 检查数据集大小
    print(f"完整数据集样本数: {len(full_dataloader.dataset)}")
    print(f"批次数量: {len(full_dataloader)}")

    # 初始化变量用于收集所有预测结果和真实标签
    all_predictions = []  # 基于阈值的预测结果
    all_predictions_argmax = []  # 基于argmax的预测结果（保留原始方法）
    all_labels = []
    all_probabilities = []  # 用于AUC计算
    
    # 基于阈值的统计
    total_accuracy_o_threshold = 0
    total_accuracy_ecc_threshold = 0
    total_num_labels_ecc = 0
    total_num_labels_o = 0
    
    # 基于argmax的统计（保留）
    total_accuracy_o_argmax = 0
    total_accuracy_ecc_argmax = 0
    
    all_legth = 0
    
    with torch.no_grad():
        # 遍历完整数据集的DataLoader
        for batch_idx, (test_DNAs, test_labels) in enumerate(full_dataloader):
            # 确保标签是浮点型
            test_labels = test_labels.float()
            
            if torch.cuda.is_available():
                test_DNAs = test_DNAs.cuda()
                test_labels = test_labels.cuda()
            
            outputs = model(test_DNAs)
            
            # 收集预测结果和真实标签
            probabilities = torch.softmax(outputs, dim=1)  # 获取概率
            predictions_argmax = outputs.argmax(dim=1)  # 基于argmax的预测
            
            # 基于阈值的预测：eccDNA概率 >= THRESHOLD 则预测为eccDNA(0)，否则为otherDNA(1)
            # 注意：probabilities[:, 0]是eccDNA的概率
            eccDNA_probs = probabilities[:, 0]  # eccDNA的概率
            predictions_threshold = torch.where(eccDNA_probs >= THRESHOLD, 
                                              torch.tensor(0).cuda() if torch.cuda.is_available() else torch.tensor(0),
                                              torch.tensor(1).cuda() if torch.cuda.is_available() else torch.tensor(1))
            
            all_predictions.extend(predictions_threshold.cpu().numpy())  # 使用基于阈值的预测
            all_predictions_argmax.extend(predictions_argmax.cpu().numpy())  # 保留argmax预测
            all_labels.extend(test_labels.argmax(dim=1).cpu().numpy())
            
            # 使用otherDNA的概率作为正类（用于AUC计算）
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())

            # 计算准确率（基于阈值）
            cout = 0
            acc_ecc_threshold = 0
            acc_o_threshold = 0
            acc_ecc_argmax = 0
            acc_o_argmax = 0
            num = 0
            num_o = 0
            
            for i, t in enumerate(test_labels.argmax(1)):
                cout = cout + 1
                t_ecc = t.item()
                if t_ecc == 0:     # 标签是eccDNA
                    num = num + 1
                    # 基于阈值的判断
                    if predictions_threshold[i].item() == 0:
                        acc_ecc_threshold = acc_ecc_threshold + 1
                    # 基于argmax的判断（保留）
                    if predictions_argmax[i].item() == 0:
                        acc_ecc_argmax = acc_ecc_argmax + 1
                else:               # 标签是otherDNA
                    num_o = num_o + 1 
                    # 基于阈值的判断
                    if predictions_threshold[i].item() == 1:
                        acc_o_threshold = acc_o_threshold + 1
                    # 基于argmax的判断（保留）
                    if predictions_argmax[i].item() == 1:
                        acc_o_argmax = acc_o_argmax + 1
            
            all_legth = all_legth + cout
            total_accuracy_ecc_threshold = total_accuracy_ecc_threshold + acc_ecc_threshold
            total_accuracy_o_threshold = total_accuracy_o_threshold + acc_o_threshold
            total_accuracy_ecc_argmax = total_accuracy_ecc_argmax + acc_ecc_argmax
            total_accuracy_o_argmax = total_accuracy_o_argmax + acc_o_argmax
            total_num_labels_ecc = total_num_labels_ecc + num
            total_num_labels_o = total_num_labels_o + num_o

            if batch_idx % 10 == 0:  # 每10个batch打印一次进度
                time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"处理进度: {batch_idx+1}/{len(full_dataloader)}批次, 已处理样本: {all_legth}, 时间: {time}")
    
    # 转换为numpy数组
    all_predictions = np.array(all_predictions)
    all_predictions_argmax = np.array(all_predictions_argmax)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # 计算eccDNA概率（用于阈值分析）
    eccDNA_probabilities = 1 - all_probabilities  # 因为all_probabilities是otherDNA的概率
    
    # 打印数据集统计信息
    print(f"\n数据集统计:")
    print(f"总样本数: {len(all_labels)}")
    print(f"eccDNA样本数: {np.sum(all_labels == 0)}")  # 注意：标签0对应eccDNA
    print(f"OtherDNA样本数: {np.sum(all_labels == 1)}")  # 标签1对应OtherDNA
    
    # 基于阈值的预测统计
    print(f"\n基于阈值({THRESHOLD})的预测统计:")
    pred_ecc_threshold = np.sum(all_predictions == 0)
    pred_other_threshold = np.sum(all_predictions == 1)
    print(f"预测为eccDNA的样本数: {pred_ecc_threshold}")
    print(f"预测为otherDNA的样本数: {pred_other_threshold}")
    
    # 调试信息 - 打印一些样本的真实标签和预测概率
    print("\nSample predictions with threshold (first 10):")
    for i in range(min(10, len(all_labels))):
        true_label_name = "eccDNA" if all_labels[i] == 0 else "OtherDNA"
        pred_label_name_threshold = "eccDNA" if all_predictions[i] == 0 else "OtherDNA"
        pred_label_name_argmax = "eccDNA" if all_predictions_argmax[i] == 0 else "OtherDNA"
        ecc_prob = eccDNA_probabilities[i]
        print(f"Sample {i}: True={all_labels[i]}({true_label_name}), "
              f"eccDNA概率={ecc_prob:.4f}, "
              f"阈值预测={all_predictions[i]}({pred_label_name_threshold}), "
              f"argmax预测={all_predictions_argmax[i]}({pred_label_name_argmax})")
    
    # 计算各种评估指标（基于阈值）
    # AUC-ROC
    try:
        auc_roc = roc_auc_score(all_labels, all_probabilities)
        print(f"AUC-ROC calculated: {auc_roc:.4f}")
    except Exception as e:
        print(f"Error calculating AUC-ROC: {e}")
        auc_roc = 0.5
    
    # 基于阈值的指标
    f1_threshold = f1_score(all_labels, all_predictions, average='binary')
    precision_threshold = precision_score(all_labels, all_predictions, average='binary')
    recall_threshold = recall_score(all_labels, all_predictions, average='binary')
    
    # 基于argmax的指标（保留用于比较）
    f1_argmax = f1_score(all_labels, all_predictions_argmax, average='binary')
    precision_argmax = precision_score(all_labels, all_predictions_argmax, average='binary')
    recall_argmax = recall_score(all_labels, all_predictions_argmax, average='binary')
    
    # 混淆矩阵（基于阈值）
    cm_threshold = confusion_matrix(all_labels, all_predictions)
    
    if cm_threshold.shape == (2, 2):
        tn_threshold, fp_threshold, fn_threshold, tp_threshold = cm_threshold.ravel()
    else:
        tn_threshold, fp_threshold, fn_threshold, tp_threshold = 0, 0, 0, 0
    
    # 特异性（基于阈值）
    specificity_threshold = tn_threshold / (tn_threshold + fp_threshold) if (tn_threshold + fp_threshold) > 0 else 0
    
    # 计算并输出最终准确率（基于阈值）
    if (total_num_labels_ecc + total_num_labels_o) > 0:
        total_accuracy_threshold = (total_accuracy_ecc_threshold + total_accuracy_o_threshold) / (total_num_labels_ecc + total_num_labels_o)
        line1 = f"Full dataset accuracy (阈值={THRESHOLD}): {total_accuracy_threshold:.4f}"
        
        # 基于argmax的准确率（保留）
        total_accuracy_argmax = (total_accuracy_ecc_argmax + total_accuracy_o_argmax) / (total_num_labels_ecc + total_num_labels_o)
        line1_argmax = f"Full dataset accuracy (argmax): {total_accuracy_argmax:.4f}"
    else:
        line1 = "Full dataset accuracy: No samples processed"
        line1_argmax = line1
    
    if total_num_labels_ecc > 0:
        ecc_accuracy_threshold = total_accuracy_ecc_threshold / total_num_labels_ecc
        line2 = f"Accuracy of eccDNA (阈值={THRESHOLD}): {ecc_accuracy_threshold:.4f}"
        
        ecc_accuracy_argmax = total_accuracy_ecc_argmax / total_num_labels_ecc
        line2_argmax = f"Accuracy of eccDNA (argmax): {ecc_accuracy_argmax:.4f}"
    else:
        line2 = "Accuracy of eccDNA: No eccDNA samples"
        line2_argmax = line2
    
    # 添加其他统计信息
    line3 = "Total samples: {}".format(total_num_labels_ecc + total_num_labels_o)
    line4 = "eccDNA samples: {}, OtherDNA samples: {}".format(total_num_labels_ecc, total_num_labels_o)
    line5 = "Correctly classified eccDNA: {}, Correctly classified OtherDNA: {}".format(
        total_accuracy_ecc_threshold, total_accuracy_o_threshold)
    
    # 新增指标（基于阈值）
    line6 = "AUC-ROC: {:.4f}".format(auc_roc)
    line7 = f"F1 Score (阈值={THRESHOLD}): {f1_threshold:.4f}"
    line8 = f"Precision (阈值={THRESHOLD}): {precision_threshold:.4f}"
    line9 = f"Recall (阈值={THRESHOLD}): {recall_threshold:.4f}"
    line10 = f"Specificity (阈值={THRESHOLD}): {specificity_threshold:.4f}"
    line11 = f"Confusion Matrix (阈值={THRESHOLD}):\n[[TN:{tn_threshold}, FP:{fp_threshold}]\n [FN:{fn_threshold}, TP:{tp_threshold}]]"
    
    # 添加argmax指标对比
    line7_argmax = f"F1 Score (argmax): {f1_argmax:.4f}"
    line8_argmax = f"Precision (argmax): {precision_argmax:.4f}"
    line9_argmax = f"Recall (argmax): {recall_argmax:.4f}"
    
    # 打印所有结果
    print("\n" + "="*60)
    print("评估结果 (基于阈值分类):")
    print("="*60)
    print(line1)
    print(line1_argmax)  # 显示argmax结果对比
    print(line2)
    print(line2_argmax)  # 显示argmax结果对比
    print(line3)
    print(line4)
    print(line5)
    print(line6)
    print(line7)
    print(line7_argmax)  # 显示argmax结果对比
    print(line8)
    print(line8_argmax)  # 显示argmax结果对比
    print(line9)
    print(line9_argmax)  # 显示argmax结果对比
    print(line10)
    print(line11)
    
    # 绘制ROC曲线并保存到文件
    try:
        fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
        
        # 如果AUC很低，尝试反转概率
        if auc_roc < 0.5:
            print("AUC-ROC is less than 0.5, trying with inverted probabilities...")
            all_probabilities_inv = 1 - all_probabilities
            try:
                auc_roc_inv = roc_auc_score(all_labels, all_probabilities_inv)
                print(f"Inverted AUC-ROC: {auc_roc_inv:.4f}")
                if auc_roc_inv > auc_roc:
                    auc_roc = auc_roc_inv
                    all_probabilities = all_probabilities_inv
                    fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
            except Exception as e:
                print(f"Error with inverted probabilities: {e}")
        
        # 绘制主图表
        plt.figure(figsize=(12, 10))
        
        # ROC曲线
        plt.subplot(2, 2, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_roc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        # 在ROC曲线上标记阈值点
        # 找到最接近阈值的点
        threshold_idx = np.argmin(np.abs(tpr - THRESHOLD))
        if threshold_idx < len(fpr):
            plt.scatter(fpr[threshold_idx], tpr[threshold_idx], color='red', s=50, 
                       label=f'Threshold {THRESHOLD}')
            plt.annotate(f'Threshold {THRESHOLD}', 
                        xy=(fpr[threshold_idx], tpr[threshold_idx]), 
                        xytext=(fpr[threshold_idx]+0.1, tpr[threshold_idx]-0.1),
                        arrowprops=dict(arrowstyle='->', color='red'))
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve (Threshold={THRESHOLD})')
        plt.legend(loc="lower right")
        
        # 绘制PR曲线
        precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probabilities)
        pr_auc = auc(recall_curve, precision_curve)
        plt.subplot(2, 2, 2)
        plt.plot(recall_curve, precision_curve, color='blue', lw=2, label='PR curve (area = %0.2f)' % pr_auc)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve (Threshold={THRESHOLD})')
        plt.legend(loc="lower left")
        
        # 绘制混淆矩阵热图
        plt.subplot(2, 2, 3)
        plt.imshow(cm_threshold, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix (Threshold={THRESHOLD})')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['eccDNA', 'OtherDNA'])
        plt.yticks(tick_marks, ['eccDNA', 'OtherDNA'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # 在热图中添加数值
        thresh = cm_threshold.max() / 2.
        for i, j in np.ndindex(cm_threshold.shape):
            plt.text(j, i, format(cm_threshold[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm_threshold[i, j] > thresh else "black")
        
        # 添加详细分类报告
        plt.subplot(2, 2, 4)
        plt.axis('off')
        report_threshold = classification_report(all_labels, all_predictions, target_names=['eccDNA', 'OtherDNA'])
        plt.text(0, 0.5, report_threshold, fontfamily='monospace', fontsize=9, verticalalignment='center')
        plt.title(f'Classification Report (Threshold={THRESHOLD})')
        
        plt.tight_layout()
        plt.savefig(f'./full_dataset_evaluation_metrics_threshold{THRESHOLD}_v{version}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error plotting curves: {e}")
    
    # 保存结果到文件
    output_filename = f'./full_dataset_acc_threshold{THRESHOLD}_v{version}.txt'
    file2 = open(output_filename, 'w')
    file2.writelines(f"完整数据集评估结果 (阈值={THRESHOLD})\n")
    file2.writelines("="*60 + "\n")
    file2.writelines(line1+'\n')
    file2.writelines(line1_argmax+'\n')
    file2.writelines(line2+'\n')
    file2.writelines(line2_argmax+'\n')
    file2.writelines(line3+'\n')
    file2.writelines(line4+'\n')
    file2.writelines(line5+'\n')
    file2.writelines(line6+'\n')
    file2.writelines(line7+'\n')
    file2.writelines(line7_argmax+'\n')
    file2.writelines(line8+'\n')
    file2.writelines(line8_argmax+'\n')
    file2.writelines(line9+'\n')
    file2.writelines(line9_argmax+'\n')
    file2.writelines(line10+'\n')
    file2.writelines(line11+'\n')
    file2.writelines("\nClassification Report (Threshold):\n" + report_threshold)
    
    # 添加argmax的分类报告
    report_argmax = classification_report(all_labels, all_predictions_argmax, target_names=['eccDNA', 'OtherDNA'])
    file2.writelines("\n\nClassification Report (Argmax):\n" + report_argmax)
    file2.close()
    
    print(f"\n评估完成. 结果已保存到:")
    print(f"- {output_filename}")
    print(f"- full_dataset_evaluation_metrics_threshold{THRESHOLD}_v{version}.png")