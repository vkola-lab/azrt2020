from utils_stat import get_roc_info, get_pr_info, load_neurologist_data, calc_neurologist_statistics, read_raw_score
import matplotlib.pyplot as plt
from utils_plot import plot_curve, plot_legend, plot_neorologist
from time import time
from matrix_stat import confusion_matrix, stat_metric
import collections
from tabulate import tabulate
import os

def plot_legend(axes, crv_lgd_hdl, crv_info, neo_lgd_hdl):
    m_name = list(crv_lgd_hdl.keys())
    ds_name = list(crv_lgd_hdl[m_name[0]].keys())

    hdl = collections.defaultdict(list)
    val = collections.defaultdict(list)

    if neo_lgd_hdl:
        for ds in neo_lgd_hdl:
            hdl[ds] += neo_lgd_hdl[ds]
            val[ds] += ['Neurologist', 'Avg. Neurologist']

    convert = {'cnn':"1.5T", 'cnnp':"1.5T+"}

    for ds in ds_name:
        for m in m_name:
            hdl[ds].append(crv_lgd_hdl[m][ds])
            val[ds].append('{}: {:.3f}$\pm${:.3f}'.format(convert[m], crv_info[m][ds]['auc_mean'], crv_info[m][ds]['auc_std']))

        axes[ds].legend(hdl[ds], val[ds],
                        facecolor='w', prop={"weight":'bold', "size":17},  # frameon=False,
                        bbox_to_anchor=(0.04, 0.04, 0.5, 0.5),
                        loc='lower left')

def roc_plot_perfrom_table(txt_file=None):
    roc_info, pr_info = {}, {}
    for m in ['cnn', 'cnnp']:
        roc_info[m], pr_info[m] = {}, {}
        for ds in ['test', 'AIBL', 'NACC']:
            Scores, Labels = [], []
            for exp_idx in range(1):
                for repe_idx in range(5):
                    labels, scores = read_raw_score('checkpoint_dir/{}_exp{}/raw_score_{}_{}.txt'.format(m, exp_idx, ds, repe_idx))
                    Scores.append(scores)
                    Labels.append(labels)
            roc_info[m][ds] = get_roc_info(Labels, Scores)
            pr_info[m][ds] = get_pr_info(Labels, Scores)

    plt.style.use('fivethirtyeight')
    plt.rcParams['axes.facecolor'] = 'w'
    plt.rcParams['figure.facecolor'] = 'w'
    plt.rcParams['savefig.facecolor'] = 'w'

    convert = {'cnn':"1.5T", 'cnnp':"1.5T+"}

    # roc plot
    fig, axes_ = plt.subplots(1, 3, figsize=[18, 6], dpi=100)
    axes = dict(zip(['test', 'AIBL', 'NACC'], axes_))
    hdl_crv = {'cnn': {}, 'cnnp': {}}
    for i, ds in enumerate(['test', 'AIBL', 'NACC']):
        title = ds
        i += 1
        hdl_crv['cnn'][ds] = plot_curve(curve='roc', **roc_info['cnn'][ds], ax=axes[ds],
                                    **{'color': 'C{}'.format(i), 'hatch': '//////', 'alpha': .4, 'line': 'solid',
                                        'title': title})
        hdl_crv['cnnp'][ds] = plot_curve(curve='roc', **roc_info['cnnp'][ds], ax=axes[ds],
                                    **{'color': 'C{}'.format(i), 'hatch': '....', 'alpha': .4, 'line': 'dashed',
                                        'title': title})
    plot_legend(axes=axes, crv_lgd_hdl=hdl_crv, crv_info=roc_info, neo_lgd_hdl=None)
    fig.savefig('./plot/roc.tif', dpi=50)

    # pr plot
    fig, axes_ = plt.subplots(1, 3, figsize=[18, 6], dpi=100)
    axes = dict(zip(['test', 'AIBL', 'NACC'], axes_))
    hdl_crv = {'cnn': {}, 'cnnp': {}}
    for i, ds in enumerate(['test', 'AIBL', 'NACC']):
        title = ds
        i += 1
        hdl_crv['cnn'][ds] = plot_curve(curve='pr', **pr_info['cnn'][ds], ax=axes[ds],
                                    **{'color': 'C{}'.format(i), 'hatch': '//////', 'alpha': .4, 'line': 'solid',
                                        'title': title})
        hdl_crv['cnnp'][ds] = plot_curve(curve='pr', **pr_info['cnnp'][ds], ax=axes[ds],
                                    **{'color': 'C{}'.format(i), 'hatch': '....', 'alpha': .4, 'line': 'dashed',
                                        'title': title})
    plot_legend(axes=axes, crv_lgd_hdl=hdl_crv, crv_info=pr_info, neo_lgd_hdl=None)
    fig.savefig('./plot/pr.tif', dpi=50)

    table = collections.defaultdict(dict)

    for i, ds in enumerate(['valid', 'test', 'AIBL', 'NACC']):
        for m in ['cnn', 'cnnp']:
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


    print('################################################################ 1.5T')
    cnn_table = [[ds]+table[ds]['cnn'] for ds in ['valid', 'test', 'AIBL', 'NACC']]
    print(tabulate(cnn_table,
    headers=['dataset', 'accuracy', 'sensitivity', 'specificity', 'F-1', 'MCC']))

    print('################################################################ 1.5T*')
    cnnp_table = [[ds]+table[ds]['cnnp'] for ds in ['valid', 'test', 'AIBL', 'NACC']]
    print(tabulate(cnnp_table,
    headers=['dataset', 'accuracy', 'sensitivity', 'specificity', 'F-1', 'MCC']))

    if txt_file:
        with open(txt_file, 'a') as f:
            line = tabulate(cnnp_table, headers=['dataset', 'accuracy', 'sensitivity', 'specificity', 'F-1', 'MCC'])
            f.write(str(line) + '\n')
