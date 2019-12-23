import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import interp
import os
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import roc_curve, auc, confusion_matrix
from glob import glob


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

if __name__ == "__main__":
    ROC_plot('../model_checkpoint/ADNI_3T_NL_ADNI_3T_AD_balance0/')

