from utils_stat import get_roc_info, get_pr_info, load_neurologist_data, calc_neurologist_statistics, read_raw_score
import matplotlib.pyplot as plt
from utils_plot import plot_curve, plot_legend, plot_neorologist
from time import time
from matrix_stat import confusion_matrix, stat_metric
import collections
from tabulate import tabulate
import os
from matplotlib.patches import Rectangle
from scipy import stats
import csv
import numpy as np

def p_val(o, g):
    t, p = stats.ttest_ind(o, g, equal_var = False)
    # print(o, g, p)
    return p

def plot_legend(axes, crv_lgd_hdl, crv_info, neo_lgd_hdl, set1, set2):
    m_name = list(crv_lgd_hdl.keys())
    ds_name = list(crv_lgd_hdl[m_name[0]].keys())

    hdl = collections.defaultdict(list)
    val = collections.defaultdict(list)

    if neo_lgd_hdl:
        for ds in neo_lgd_hdl:
            hdl[ds] += neo_lgd_hdl[ds]
            val[ds] += ['Neurologist', 'Avg. Neurologist']

    convert = {'fcn':"1.5T", 'fcn_gan':"1.5T*", 'fcn_aug':'1.5T Aug'}

    for ds in ds_name:
        for m in m_name:
            # print(m, ds)
            hdl[ds].append(crv_lgd_hdl[m][ds])
            val[ds].append('{}: {:.3f}$\pm${:.3f}'.format(convert[m], crv_info[m][ds]['auc_mean'], crv_info[m][ds]['auc_std']))
        extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
        val[ds].append('p-value: {:.4f}'.format(p_val(set1[ds], set2[ds])))

        axes[ds].legend(hdl[ds]+[extra], val[ds],
                        facecolor='w', prop={"weight":'bold', "size":17},  # frameon=False,
                        bbox_to_anchor=(0.04, 0.04, 0.5, 0.5),
                        loc='lower left')

def roc_plot_perfrom_table(txt_file=None, mode=['fcn', 'fcn_gan']):
    roc_info, pr_info = {}, {}
    aucs, apss = {}, {}
    for m in mode:
        roc_info[m], pr_info[m], aucs[m], apss[m] = {}, {}, {}, {}
        for ds in ['test', 'AIBL', 'NACC']:
            Scores, Labels = [], []
            for exp_idx in range(1):
                for repe_idx in range(5):
                    labels, scores = read_raw_score('checkpoint_dir/{}_exp{}/raw_score_{}_{}.txt'.format(m, exp_idx, ds, repe_idx))
                    Scores.append(scores)
                    Labels.append(labels)
            # scores = np.array(Scores).mean(axis=0)
            # labels = Labels[0]
            # filename = '{}_{}_mean'.format(m, ds)
            # with open(filename+'.csv', 'w') as f1:
            #     wr = csv.writer(f1)
            #     wr.writerows([[s] for s in scores])
            # with open(filename+'_l.csv', 'w') as f2:
            #     wr = csv.writer(f2)
            #     wr.writerows([[l] for l in labels])
            #     # f.write(' '.join(map(str,scores))+'\n'+' '.join(map(str,labels)))

            roc_info[m][ds], aucs[m][ds] = get_roc_info(Labels, Scores)
            pr_info[m][ds], apss[m][ds] = get_pr_info(Labels, Scores)

    plt.style.use('fivethirtyeight')
    plt.rcParams['axes.facecolor'] = 'w'
    plt.rcParams['figure.facecolor'] = 'w'
    plt.rcParams['savefig.facecolor'] = 'w'

    convert = {'fcn':"1.5T", 'fcn_gan':"1.5T*", 'fcn_aug':'1.5T Aug'}

    # roc plot
    fig, axes_ = plt.subplots(1, 3, figsize=[18, 6], dpi=100)
    axes = dict(zip(['test', 'AIBL', 'NACC'], axes_))
    lines = ['solid', '-.', '-']
    hdl_crv = {m:{} for m in mode}
    for i, ds in enumerate(['test', 'AIBL', 'NACC']):
        title = ds
        i += 1
        for j, m in enumerate(mode):
            hdl_crv[m][ds] = plot_curve(curve='roc', **roc_info[m][ds], ax=axes[ds],
                                    **{'color': 'C{}'.format(i), 'hatch': '//////', 'alpha': .4, 'line': lines[j],
                                        'title': title})

    plot_legend(axes=axes, crv_lgd_hdl=hdl_crv, crv_info=roc_info, neo_lgd_hdl=None, set1=aucs['fcn'], set2=aucs['fcn_gan'])
    fig.savefig('./plot/roc.png', dpi=300)

    # pr plot
    fig, axes_ = plt.subplots(1, 3, figsize=[18, 6], dpi=100)
    axes = dict(zip(['test', 'AIBL', 'NACC'], axes_))
    hdl_crv = {m: {} for m in mode}
    for i, ds in enumerate(['test', 'AIBL', 'NACC']):
        title = ds
        i += 1
        for j, m in enumerate(mode):
            hdl_crv[m][ds] = plot_curve(curve='pr', **pr_info[m][ds], ax=axes[ds],
                                    **{'color': 'C{}'.format(i), 'hatch': '//////', 'alpha': .4, 'line': lines[j],
                                        'title': title})

    plot_legend(axes=axes, crv_lgd_hdl=hdl_crv, crv_info=pr_info, neo_lgd_hdl=None, set1=apss['fcn'], set2=apss['fcn_gan'])
    fig.savefig('./plot/pr.png', dpi=300)

    table = collections.defaultdict(dict)

    for i, ds in enumerate(['valid', 'test', 'AIBL', 'NACC']):
        for m in mode:
            Matrix = []
            for exp_idx in range(1):
                for repe_idx in range(5):
                    labels, scores = read_raw_score('checkpoint_dir/{}_exp{}/raw_score_{}_{}.txt'.format(m, exp_idx, ds, repe_idx))
                    Matrix.append(confusion_matrix(labels, scores))
            accu_m, accu_s, sens_m, sens_s, spec_m, spec_s, f1_m, f1_s, mcc_m, mcc_s = stat_metric(Matrix)
            table[ds][m] = ['{0:.4f}+/-{1:.4f}'.format(accu_m, accu_s),
                            '{0:.4f}+/-{1:.4f}'.format(sens_m, sens_s),
                            '{0:.4f}+/-{1:.4f}'.format(spec_m, spec_s),
                            '{0:.4f}+/-{1:.4f}'.format(f1_m, f1_s),
                            '{0:.4f}+/-{1:.4f}'.format(mcc_m, mcc_s)]

    cnn_table = {}
    for m in mode:
        print('################################################################ ' + m)
        cnn_table[m] = [[ds]+table[ds][m] for ds in ['valid', 'test', 'AIBL', 'NACC']]
        print(tabulate(cnn_table[m],
        headers=['dataset', 'accuracy', 'sensitivity', 'specificity', 'F-1', 'MCC']))

    if txt_file:
        with open(txt_file, 'a') as f:
            line = tabulate(cnn_table['fcn_gan'], headers=['dataset', 'accuracy', 'sensitivity', 'specificity', 'F-1', 'MCC'])
            f.write(str(line) + '\n')

if __name__ == "__main__":
    roc_plot_perfrom_table()
