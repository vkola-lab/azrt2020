import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import interp
import os
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import roc_curve, auc, confusion_matrix
from glob import glob
from utils import read_txt

def read_score_label(file):
    score, label = [], []
    for line in open(file, 'r'):
        line = line.strip('\n')
        line = line.split('__')
        score.append([float(line[0]), float(line[1])])
        label.append(int(line[2]))
    return score, label

def softmax(a):
    a0, a1 = a
    return np.exp(a1) / (np.exp(a1) + np.exp(a0))

def roc_util(label, pred):
    score = [softmax(a) for a in pred]
    fpr, tpr, thresholds = roc_curve(label, score)
    mean_fpr = np.linspace(0, 1, 100)
    interp_tpr = interp(mean_fpr, fpr, tpr)
    area = auc(fpr, tpr)
    return mean_fpr, interp_tpr, area

def get_metrics(matrix):
    tn, fn, fp, tp = matrix[0][0]+0.00000001, matrix[0][1]+0.00000001, matrix[1][0]+0.00000001, matrix[1][1]+0.00000001
    accuracy = (tn + tp) / (tp + tn + fn + fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    F1 = 2 * precision * recall / (precision + recall)
    MCC = (tp*tn-fp*fn) / ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5
    return accuracy, precision, recall, F1, MCC

def ROC_plot(path):
    TPR = []
    AUC = []
    name = path.split('/')[-2]
    files = glob(path + '*.txt')
    for file in files:
        pred, label = read_score_label(file)
        fpr, tpr, area = roc_util(label, pred)
        TPR.append(tpr)
        AUC.append(area)

    TPR = np.array(TPR)
    TPR_std = np.std(TPR, axis=0)

    mean_tpr = np.mean(TPR, axis=0)

    plt.figure(figsize=(10, 10))
    plt.plot(fpr, mean_tpr, color='seagreen',
             label=r'(AUC=%0.3f $\pm$ %0.3f)' % (auc(fpr, mean_tpr), np.std(AUC)),
             lw=2, alpha=.8)
    tprs_upper = np.minimum(mean_tpr + TPR_std, 1)
    tprs_lower = np.maximum(mean_tpr - TPR_std, 0)
    plt.fill_between(fpr, tprs_lower, tprs_upper, color='seagreen', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', alpha=.8)
    plt.xlim([-0.005, 1.005])
    plt.ylim([-0.005, 1.005])
    legend_properties = {'weight':'bold', 'size':23}
    plt.xlabel('False Positive Rate', fontweight='bold', fontsize = 23)
    plt.ylabel('True Positive Rate', fontweight='bold', fontsize = 23)
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize = 20, fontweight="bold")
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize = 20, fontweight="bold")
    plt.legend(prop=legend_properties)
    plt.savefig('{}{}.png'.format(path, name))
    plt.close()

def bold_axs_stick(axs, fontsize):
    for tick in axs.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')
    for tick in axs.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')

def GAN_test_plot(out_dir, id, img1, output, img3, info):
    img15 = img1
    imgp = output
    img3 = img3
    plt.set_cmap("gray")
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    fig, axs = plt.subplots(3, 3, figsize=(20,15))
    side_a = 100
    side_b = 160
    axs[0, 0].imshow(img15[:, :, 105].T, vmin=-1, vmax=2.5)
    axs[0, 0].set_title('1.5T', fontsize=25)
    axs[0, 0].axis('off')
    axs[1, 0].imshow(img3[:, :, 105].T, vmin=-1, vmax=2.5)
    axs[1, 0].set_title('3T', fontsize=25)
    axs[1, 0].axis('off')
    axs[2, 0].imshow(imgp[:, :, 105].T, vmin=-1, vmax=2.5)
    axs[2, 0].set_title('1.5T+', fontsize=25)
    axs[2, 0].axis('off')

    axs[0, 1].imshow(img15[side_a:side_b, side_a:side_b, 105].T, vmin=-1, vmax=2.5)
    axs[0, 1].set_title('1.5T zoom in', fontsize=25)
    axs[0, 1].axis('off')
    axs[1, 1].imshow(img3[side_a:side_b, side_a:side_b, 105].T, vmin=-1, vmax=2.5)
    axs[1, 1].set_title('3T zoom in', fontsize=25)
    axs[1, 1].axis('off')
    axs[2, 1].imshow(imgp[side_a:side_b, side_a:side_b, 105].T, vmin=-1, vmax=2.5)
    axs[2, 1].set_title('1.5T+ zoom in', fontsize=25)
    axs[2, 1].axis('off')

    axs[0, 2].hist(img15[side_a:side_b, side_a:side_b, 105].T.flatten(), bins=50, range=(0, 1.8))
    bold_axs_stick(axs[0, 2], 16)
    axs[0, 2].set_xticks([0, 0.5, 1, 1.5])
    axs[0, 2].set_yticks([0, 100, 200, 300])
    axs[0, 2].set_title('1.5T voxel histogram', fontsize=25)
    axs[1, 2].hist(img3[side_a:side_b, side_a:side_b, 105].T.flatten(), bins=50, range=(0, 1.8))
    bold_axs_stick(axs[1, 2], 16)
    axs[1, 2].set_xticks([0, 0.5, 1, 1.5])
    axs[1, 2].set_yticks([0, 100, 200, 300])
    axs[1, 2].set_title('3T voxel histogram', fontsize=25)
    axs[2, 2].hist(imgp[side_a:side_b, side_a:side_b, 105].T.flatten(), bins=50, range=(0, 1.8))
    bold_axs_stick(axs[1, 2], 16)
    axs[2, 2].set_xticks([0, 0.5, 1, 1.5])
    axs[2, 2].set_yticks([0, 100, 200, 300])
    axs[2, 2].set_title('1.5T+ voxel histogram', fontsize=25)
    plt.savefig(out_dir+info + '#{}.png'.format(id), dpi=150)
    plt.close()

def MRI_slice_plot(out_dir, mri_low, mri_high):
    img15 = np.load(mri_low)
    img3 = np.load(mri_high)
    id = mri_low.split('/')[-1][5:15]
    plt.set_cmap("gray")
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    fig, axs = plt.subplots(2, 3, figsize=(20,12))
    side_a = 100
    side_b = 160
    axs[0, 0].imshow(img15[:, :, 105].T, vmin=-1, vmax=2.5)
    axs[0, 0].set_title('1.5T', fontsize=25)
    axs[0, 0].axis('off')
    axs[1, 0].imshow(img3[:, :, 105].T, vmin=-1, vmax=2.5)
    axs[1, 0].set_title('3T', fontsize=25)
    axs[1, 0].axis('off')

    axs[0, 1].imshow(img15[side_a:side_b, side_a:side_b, 105].T, vmin=-1, vmax=2.5)
    axs[0, 1].set_title('1.5T zoom in', fontsize=25)
    axs[1, 1].imshow(img3[side_a:side_b, side_a:side_b, 105].T, vmin=-1, vmax=2.5)
    axs[1, 1].set_title('3T zoom in', fontsize=25)
    axs[0, 1].axis('off')
    axs[1, 1].axis('off')

    axs[0, 2].hist(img15[side_a:side_b, side_a:side_b, 105].T.flatten(), bins=50, range=(0, 1.8))
    bold_axs_stick(axs[0, 2], 16)
    axs[0, 2].set_xticks([0, 0.5, 1, 1.5])
    axs[0, 2].set_yticks([0, 100, 200, 300])
    axs[0, 2].set_title('1.5T voxel histogram', fontsize=25)
    axs[1, 2].hist(img3[side_a:side_b, side_a:side_b, 105].T.flatten(), bins=50, range=(0, 1.8))
    bold_axs_stick(axs[1, 2], 16)
    axs[1, 2].set_xticks([0, 0.5, 1, 1.5])
    axs[1, 2].set_yticks([0, 100, 200, 300])
    axs[1, 2].set_title('3T voxel histogram', fontsize=25)
    plt.savefig(out_dir + '{}.png'.format(id), dpi=150)
    plt.close()


def all_mri_plot(path, txt_low, txt_high):
    low_list = read_txt('../lookuptxt/', txt_low)
    high_list = read_txt('../lookuptxt/', txt_high)
    for i in range(len(low_list)):
        low_mri, high_mri = path + low_list[i], path + high_list[i]
        MRI_slice_plot('../mri_image/', low_mri, high_mri)
        break







if __name__ == "__main__":
    # ROC_plot('../model_checkpoint/ADNI_3T_NL_ADNI_3T_AD_balance0/')
    all_mri_plot('/data/datasets/ADNI_NoBack/', 'ADNI_1.5T_GAN_AD.txt', 'ADNI_3T_AD.txt')
