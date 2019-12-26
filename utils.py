import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from numpy import random
import json



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
    raw = raw[0, :, :, :, :].cpu()
    a, x, y, z = raw.shape
    risk = np.zeros((x, y, z))
    for i in range(x):
        for j in range(y):
            for k in range(z):
                risk[i, j, k] = softmax(raw[0, i, j, k], raw[1, i, j, k])
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

def test(**super_kwargs):
    print(super_kwargs['whole'])
    del super_kwargs['whole']
    test_2(**super_kwargs)

def test_2(item, ittt, df=1):
    print(item, ittt, df)

if __name__ == "__main__":
    test(item=3, ittt='a', whole=3)

