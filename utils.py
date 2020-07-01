import numpy as np
import torch
import matlab
import csv
import os
import shutil
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matlab.engine
from numpy import random
import json
from skimage import img_as_float
from skimage.metrics import structural_similarity as ssim
import sys
import time
from scipy import stats
from sklearn.metrics import precision_recall_fscore_support
from scipy.interpolate import interp1d
import pandas as pd

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

'''
ori = [[0.93382353, 0.81742044, 0.80555556, 0.88481675,],
       [0.94607843, 0.82077052, 0.76851852, 0.88219895,],
       [0.94117647, 0.84589615, 0.85185185, 0.81413613,],
       [0.93382353, 0.78224456, 0.75,       0.84816754,],
       [0.94362745, 0.74874372, 0.74074074, 0.83246073,]]

gen = [[0.95588235, 0.86097152, 0.73148148, 0.87958115,],
       [0.94607843, 0.80234506, 0.64814815, 0.82984293,],
       [0.95098039, 0.83249581, 0.87037037, 0.82984293,],
       [0.94362745, 0.77721943, 0.80555556, 0.85602094,],
       [0.94852941, 0.85929648, 0.7962963,  0.86649215,]]

ori = np.asarray(ori)
gen = np.asarray(gen)

for i in range(4):
    o = ori[:, i]
    g = gen[:, i]
    # print(o)
    # print(g)
    print('1.5  mean:', np.mean(o), 'std:', np.std(o))
    print('1.5+ mean:', np.mean(g), 'std:', np.std(g))
    t, p = stats.ttest_ind(o, g, equal_var = False)
    print('p_value:', p)
#'''


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed

def iqa_tensor(tensor, eng, filename, metric, target):
    # if not os.path.isdir(target):
    #     os.mkdir(target)
    out = []
    if metric == 'brisque':
        func = eng.brisque
    elif metric == 'niqe':
        func = eng.niqe
    elif metric == 'piqe':
        func = eng.piqe
    elif metric == 'CNR':
        return CNR(tensor)
    elif metric == 'SNR':
        return SNR(tensor)


    for side in range(len(tensor.shape)):
        vals = []
        for slice_idx in [80]:
            if side == 0:
                img = tensor[slice_idx, :, :]
            elif side == 1:
                img = tensor[:, slice_idx, :]
            else:
                img = tensor[:, :, slice_idx]
            img = matlab.double(img.tolist())
            vals += [func(img)]
        out += vals
        break
    val_avg = sum(out) / len(out)
    #np.save(target+filename+'$'+metric, out)
    # return np.asarray(out)
    return val_avg


def SNR(tensor):
    # return the signal to noise ratio
    for slice_idx in [80]:
        img = tensor[slice_idx, :, :]
        m = interp1d([np.min(img),np.max(img)],[0,255])
        img = m(img)
        val = signaltonoise(img, axis=None)
    return float(val)

def CNR(tensor):
    # return the signal to noise ratio
    for slice_idx in [80]:
        img = tensor[slice_idx, :, :] #shape 217, 181
        # print(img.shape)
        m = interp1d([np.min(img),np.max(img)],[0,255])
        img = m(img)
        roi1, roi2 = img[90:120, 80:110], img[110:140, 110:140]
        return np.abs(np.mean(roi1) - np.mean(roi2)) / np.sqrt(np.square(np.std(roi1))+np.square(np.std(roi2)))

def SSIM(tensor1, tensor2, zoom=False):
    ssim_list = []
    for slice_idx in [80]:
        if zoom:
            side_a = slice_idx
            side_b = slice_idx+60
            img1, img2 = tensor1[side_a:side_b, side_a:side_b, 105], tensor2[side_a:side_b, side_a:side_b, 105]
        else:
            img1, img2 = tensor1[slice_idx, :, :], tensor2[slice_idx, :, :]
        img1 = img_as_float(img1)
        img2 = img_as_float(img2)
        ssim_val = ssim(img1, img2)
        if ssim_val != ssim_val:
            print('\n\n Error @ SSIM')
            sys.exit()
        ssim_list.append(ssim_val)
    ssim_avg = sum(ssim_list) / len(ssim_list)
    return ssim_avg

def immse(tensor1, tensor2, zoom, eng):
    vals = []
    for slice_idx in [50, 80, 110]:
        if zoom:
            side_a = slice_idx
            side_b = slice_idx+60
            img1, img2 = tensor1[side_a:side_b, side_a:side_b, 105], tensor2[side_a:side_b, side_a:side_b, 105]
        else:
            img1, img2 = tensor1[slice_idx, :, :], tensor2[slice_idx, :, :]
        img1, img2 = matlab.double(img1.tolist()), matlab.double(img2.tolist())
        val = eng.immse(img1, img2)
        vals.append(val)
    val_avg = sum(vals) / len(vals)
    return val_avg

def psnr(tensor1, tensor2, zoom, eng):
    #all are single slice!
    vals = []
    for slice_idx in [50, 80, 110]:
        if zoom:
            side_a = slice_idx
            side_b = slice_idx+60
            img1, img2 = tensor1[side_a:side_b, side_a:side_b, 105], tensor2[side_a:side_b, side_a:side_b, 105]
        else:
            img1, img2 = tensor1[slice_idx, :, :], tensor2[slice_idx, :, :]
        img1, img2 = matlab.double(img1.tolist()), matlab.double(img2.tolist())
        val = eng.psnr(img1, img2)
        vals.append(val)
    val_avg = sum(vals) / len(vals)
    return val_avg

def brisque(tensor, zoom, eng):
    vals = []
    for slice_idx in [50, 80, 110]:
        if zoom:
            side_a = slice_idx
            side_b = slice_idx+60
            img = tensor[side_a:side_b, side_a:side_b, 105]
        else:
            img = tensor[slice_idx, :, :]
        img = matlab.double(img.tolist())
        val = eng.brisque(img)
        vals.append(val)
    val_avg = sum(vals) / len(vals)
    return val_avg

def niqe(tensor, zoom, eng):
    vals = []
    for slice_idx in [50, 80, 110]:
        if zoom:
            side_a = slice_idx
            side_b = slice_idx+60
            img = tensor[side_a:side_b, side_a:side_b, 105]
        else:
            img = tensor[slice_idx, :, :]
        img = matlab.double(img.tolist())
        val = eng.niqe(img)
        vals.append(val)
    val_avg = sum(vals) / len(vals)
    return val_avg

def piqe(tensor, zoom, eng):
    vals = []
    for slice_idx in [50, 80, 110]:
        if zoom:
            side_a = slice_idx
            side_b = slice_idx+60
            img = tensor[side_a:side_b, side_a:side_b, 105]
        else:
            img = tensor[slice_idx, :, :]
        img = matlab.double(img.tolist())
        val = eng.piqe(img)
        vals.append(val)
    val_avg = sum(vals) / len(vals)
    return val_avg

def report(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred)
    #print(p[0], r[0], f[0], s[0])
    #print(p[1], r[1], f[1], s[1])
    #out = classification_report(y_true, y_pred)
    #print(out)
    return (p[0], r[0], f[0], s[0])

def p_val(o, g):
    t, p = stats.ttest_ind(o, g, equal_var = True)
    return p

def read_json(config_file):
    with open(config_file) as config_buffer:
        config = json.loads(config_buffer.read())
    return config

def remove_dup(list1, list2):
    # remove the 82 cases (list2) from 417 cases (list1)
    # currently specifically written for this single case
    # will return a list, where each element corresponding to non-82 cases in 417
    # i.e.: if [0,1,2,3,4,5], and 2 is the case of 82, then will return [0,1,3,4,5]
    idxs = list(range(len(list1)))
    list1 = [i[:22] for i in list1]
    list2 = [i[:22] for i in list2]
    for item in list2:
        if item in list1:
            idxs.remove(list1.index(item))
    #print(len(idxs), len(list1))
    return idxs

def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    filenames = [a[0] for a in your_list[1:]]
    labels = [0 if a[1]=='NL' else 1 for a in your_list[1:]]
    return filenames, labels

def save_list(txt_dir, txt_name, file):
    with open(txt_dir + txt_name, 'w') as f:
        f.write(str(file))

def load_list(txt_dir, txt_name):
    with open(txt_dir + txt_name, 'r') as f:
        return eval(f.readline())

def load_txt(txt_dir, txt_name):
    List = []
    with open(txt_dir + txt_name, 'r') as f:
        for line in f:
            List.append(line.strip('\n').replace('.nii', '.npy'))
    return List

def train_valid_test_index_list(): #define our training indices
    valid_index = [i for i in range(257, 337)]
    train_index = [i for i in range(257)]
    test_index = [i for i in range(337, 417)]
    return train_index, valid_index, test_index

def padding(tensor, win_size=23):
    A = np.ones((tensor.shape[0]+2*win_size, tensor.shape[1]+2*win_size, tensor.shape[2]+2*win_size)) * tensor[-1,-1,-1]
    A[win_size:win_size+tensor.shape[0], win_size:win_size+tensor.shape[1], win_size:win_size+tensor.shape[2]] = tensor
    return A.astype(np.float32)

def get_input_variable(index_list, Data_dir, Data_list, stage):
    array_list = []
    if stage == 'train':
        patch_locs = [[random.randint(0, 134), random.randint(0, 170), random.randint(0, 134)] for _ in range(len(index_list))]
        for i, index in enumerate(index_list):
            x, y, z = patch_locs[i]
            data = np.load(Data_dir + Data_list[index])
            patch = data[x:x+47, y:y+47, z:z+47]
            array_list.append(np.expand_dims(patch, axis = 0))

    elif stage == 'valid':
        patch_locs = [[25, 90, 30], [115, 90, 30], [67, 90, 90], [67, 45, 60], [67, 135, 60]]
        data = np.load(Data_dir + Data_list[index_list[0]])
        for i, loc in enumerate(patch_locs):
            x, y, z = loc
            patch = data[x:x+47, y:y+47, z:z+47]
            array_list.append(np.expand_dims(patch, axis = 0))

    elif stage == 'test':
        for i, index in enumerate(index_list):
            data = np.load(Data_dir + Data_list[index])
            data = padding(data)
            array_list.append(np.expand_dims(data, axis = 0))

    return Variable(torch.FloatTensor(np.stack(array_list, axis = 0))).cuda()

def get_labels(index_list, Label_list, stage):
    if stage in ['train', 'test']:
        label_list = [Label_list[i] for i in index_list]
        label_list = [0 if a=='NL' else 1 for a in label_list]
    elif stage == 'valid':
        label = 0 if Label_list[index_list[0]]=='NL' else 1
        label_list = [label]*5
    label_list = np.asarray(label_list)
    return Variable(torch.LongTensor(label_list)).cuda()

def write_raw_score(f, preds, labels):
    preds = preds.data.cpu().numpy()
    labels = labels.data.cpu().numpy()
    for index, pred in enumerate(preds):
        label = str(labels[index])
        pred = "__".join(map(str, list(pred)))
        f.write(pred + '__' + label + '\n')

def get_confusion_matrix(preds, labels):
    labels = labels.data.cpu().numpy()
    preds = preds.data.cpu().numpy()
    matrix = [[0, 0], [0, 0]]
    for index, pred in enumerate(preds):
        if np.amax(pred) == pred[0]:
            if labels[index] == 0:
                matrix[0][0] += 1
            if labels[index] == 1:
                matrix[0][1] += 1
        elif np.amax(pred) == pred[1]:
            if labels[index] == 0:
                matrix[1][0] += 1
            if labels[index] == 1:
                matrix[1][1] += 1
    return matrix

def matrix_sum(A, B): # sum two confusion matrices
    return [[A[0][0]+B[0][0], A[0][1]+B[0][1]],
            [A[1][0]+B[1][0], A[1][1]+B[1][1]]]

def get_accu(matrix): # calculate accuracy from confusion matrix
    return float(matrix[0][0] + matrix[1][1])/ float(sum(matrix[0]) + sum(matrix[1]))

def softmax(x1, x2):
    return np.exp(x2) / (np.exp(x1) + np.exp(x2))

def get_AD_risk(raw):
    x1, x2 = raw[0, :, :, :], raw[1, :, :, :]
    risk = np.exp(x2) / (np.exp(x1) + np.exp(x2))
    return risk

def get_ROI(train_MCC, roi_threshold):
    roi = np.load(train_MCC)
    roi = roi > roi_threshold
    for i in range(roi.shape[0]):
        for j in range(roi.shape[1]):
            for k in range(roi.shape[2]):
                if i%3!=0 or j%2!=0 or k%3!=0:
                    roi[i,j,k] = False
    return roi

def pr_interp(rc_, rc, pr):
    pr_ = np.zeros_like(rc_)
    locs = np.searchsorted(rc, rc_)
    for idx, loc in enumerate(locs):
        l = loc - 1
        r = loc
        r1 = rc[l] if l > -1 else 0
        r2 = rc[r] if r < len(rc) else 1
        p1 = pr[l] if l > -1 else 1
        p2 = pr[r] if r < len(rc) else 0
        t1 = (1-p2)*r2/p2/(r2-r1) if p2*(r2-r1) > 1e-16 else (1-p2)*r2/1e-16
        t2 = (1-p1)*r1/p1/(r2-r1) if p1*(r2-r1) > 1e-16 else (1-p1)*r1/1e-16
        t3 = (1-p1)*r1/p1 if p1 > 1e-16 else (1-p1)*r1/1e-16
        a = 1 + t1 - t2
        b = t3 - t1*r1 + t2*r1
        pr_[idx] = rc_[idx]/(a*rc_[idx]+b)
    return pr_

def read_txt(path, txt_file):
    content = []
    with open(path + txt_file, 'r') as f:
        for line in f:
            content.append(line.strip('\n'))
    return content

def DPM_statistics(DPMs, Labels):
    shape = DPMs[0].shape[1:]
    voxel_number = shape[0] * shape[1] * shape[2]
    TP, FP, TN, FN = np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape)
    for label, DPM in zip(Labels, DPMs):
        risk_map = get_AD_risk(DPM)
        if label == 0:
            TN += (risk_map < 0.5).astype(np.int)
            FP += (risk_map >= 0.5).astype(np.int)
        elif label == 1:
            TP += (risk_map >= 0.5).astype(np.int)
            FN += (risk_map < 0.5).astype(np.int)
    tn = float("{0:.2f}".format(np.sum(TN) / voxel_number))
    fn = float("{0:.2f}".format(np.sum(FN) / voxel_number))
    tp = float("{0:.2f}".format(np.sum(TP) / voxel_number))
    fp = float("{0:.2f}".format(np.sum(FP) / voxel_number))
    matrix = [[tn, fn], [fp, tp]]
    count = len(Labels)
    TP, TN, FP, FN = TP.astype(np.float)/count, TN.astype(np.float)/count, FP.astype(np.float)/count, FN.astype(np.float)/count
    ACCU = TP + TN
    F1 = 2*TP/(2*TP+FP+FN)
    MCC = (TP*TN-FP*FN)/(np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))+0.00000001*np.ones(shape))
    return matrix, ACCU, F1, MCC

def read_csv_complete(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    filenames, labels, demors = [], [], []
    for line in your_list:
        try:
            demor = list(map(float, line[2:5]))
            gender = [0, 1] if demor[1] == 1 else [1, 0]
            demor = [(demor[0]-70.0)/10.0] + gender + [(demor[2]-27)/2]
            # demor = [demor[0]] + gender + demor[2:]
        except:
            continue
        filenames.append(line[0])
        label = 0 if line[1]=='NL' else 1
        labels.append(label)
        demors.append(demor)
    return filenames, labels, demors

# def combine_csv(f1, f2):
#     with open(f1, 'r') as f:
#         reader = csv.reader(f)
#         csv1 = list(reader)
#     with open(f2, 'r') as f:
#         reader = csv.reader(f)
#         csv2 = list(reader)
#     filenames, labels, demors = [], [], []
#     for line in csv1:
#         print(line)
#         break
#         # try:
#         #     demor = list(map(float, line[2:5]))
#         #     gender = [0, 1] if demor[1] == 1 else [1, 0]
#         #     demor = [(demor[0]-70.0)/10.0] + gender + [(demor[2]-27)/2]
#         #     # demor = [demor[0]] + gender + demor[2:]
#         # except:
#         #     continue
#         # filenames.append(line[0])
#         # label = 0 if line[1]=='NL' else 1
#         # labels.append(label)
#         # demors.append(demor)
#     print(csv2[0])
#     df1 = pd.read_csv(f1)
#     df2 = pd.read_csv(f2)
#     # df3 = pd.read_csv(f3)
#     df4 = df1.merge(df2, 'left', on='ID')
#     print(df1, df2, df4)
#     print(df2['ID']=='051_S_1123')
#     print(df2.loc[df2['ID']=='051_S_1123'])
#     # print(df1['ID']==df2['ID'])
#
#     return


if __name__ == "__main__":
    # test(item=3, ittt='a', whole=3)
    combine_csv('ADNI_GAN_iqa_ANOVA.csv', '/home/xzhou/fcn2020/gan2020/lookupcsv/exp0/train_15T_scanner.csv')
