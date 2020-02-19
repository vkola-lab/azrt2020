from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
from utils import read_txt
import random
import copy

"""
to do list:
    1. to make the dataloader work, create shuffled filename.txt and label.txt
    2. save different version for MRI scans: clip or background remove
    3. for those 82 cases, 1.5T has inconsistency between data/datasets/ADNI and /data/MRI_GAN/
    4. in filename.txt maybe put complete path of data, no need to assign data_dir
"""

class Data(Dataset):
    """
    txt files ./lookuptxt/*.txt complete path of MRIs
    MRI with clip and backremove: /data/datasets/ADNI_NoBack/*.npy
    """
    def __init__(self, Data_dir, class1, class2, stage, ratio=(0.6, 0.2, 0.2), seed=1000, shuffle=True):
        random.seed(seed)
        self.Data_dir = Data_dir
        Data_list0 = read_txt('./lookuptxt/', class1 + '.txt')
        Data_list1 = read_txt('./lookuptxt/', class2 + '.txt')
        self.Data_list = Data_list0 + Data_list1
        self.Label_list = [0]*len(Data_list0) + [1]*len(Data_list1)
        self.stage = stage
        self.length = len(self.Data_list)
        idxs = list(range(self.length))
        if shuffle:
            random.shuffle(idxs)
        split1, split2 = int(self.length*ratio[0]), int(self.length*(ratio[0]+ratio[1]))
        if self.stage == 'train':
            self.index_list = idxs[:split1]
        elif self.stage == 'valid':
            self.index_list = idxs[split1:split2]
        elif self.stage == 'test':
            self.index_list = idxs[split2:]
        elif self.stage == 'all':
            self.index_list = idxs
        else:
            raise ValueError('invalid stage setting')

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        index = self.index_list[idx]
        label = self.Label_list[index]
        data = np.load(self.Data_dir + self.Data_list[index]).astype(np.float32)
        data = np.expand_dims(data, axis=0)
        return data, label

    def get_sample_weights(self):
        labels = []
        for idx in self.index_list:
            labels.append(self.Label_list[idx])
        weights = []
        count, count0, count1 = float(len(labels)), float(labels.count(0)), float(labels.count(1))
        weights = [count/count0 if i == 0 else count/count1 for i in labels]
        return weights, count0 / count1


class PatchGenerator:
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def random_sample(self, data1, data2):
        """sample random patch from numpy array data"""
        X, Y, Z = data1.shape
        x = random.randint(0, X-self.patch_size)
        y = random.randint(0, Y-self.patch_size)
        z = random.randint(0, Z-self.patch_size)
        return data1[x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size], \
               data2[x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size]

    def fixed_sample(self, data):
        """sample patch from fixed locations"""
        patches = []
        patch_locs = [[25, 90, 30], [115, 90, 30], [67, 90, 90], [67, 45, 60], [67, 135, 60]]
        for i, loc in enumerate(patch_locs):
            x, y, z = loc
            patch = data[x:x+47, y:y+47, z:z+47]
            patches.append(np.expand_dims(patch, axis = 0))
        return patches


class GAN_Data:

    """
    stage=train   load pairwise 1.5T and 3T patch sampled from same location from same patient
    stage=valid   use a specific validation loss to save model checkpoint
    stage=test    leave some portion as testing set to directly compare 1.5T, 1.5T+ and 3T

    Todo;
    1. selectively pick good quality 3T image based on non-reference image quality
    2. G model evaluation, what validation loss to use? maybe SSIM
    3.
    """

    def __init__(self, Data_dir, stage, ratio=(0.6, 0.2, 0.2), seed=1000):
        random.seed(seed)
        self.Data_dir = Data_dir
        Data_list0 = read_txt('./lookuptxt/', 'ADNI_1.5T_GAN_NL.txt')
        Data_list1 = read_txt('./lookuptxt/', 'ADNI_1.5T_GAN_MCI.txt')
        Data_list2 = read_txt('./lookuptxt/', 'ADNI_1.5T_GAN_AD.txt')
        Data_list3 = read_txt('./lookuptxt/', 'ADNI_3T_NL.txt')
        Data_list4 = read_txt('./lookuptxt/', 'ADNI_3T_MCI.txt')
        Data_list5 = read_txt('./lookuptxt/', 'ADNI_3T_AD.txt')
        self.Data_list_lo = Data_list0 + Data_list1 + Data_list2
        self.Data_list_hi = Data_list3 + Data_list4 + Data_list5
        self.Label_list = [0]*len(Data_list0) + [2]*len(Data_list1) + [1]*len(Data_list2)
        self.stage = stage
        self.length = len(self.Data_list_lo)
        self.patchsampler = PatchGenerator(patch_size = 47)
        idxs = list(range(self.length))
        random.shuffle(idxs)
        split1, split2 = int(self.length*ratio[0]), int(self.length*(ratio[0]+ratio[1]))
        if self.stage == 'train':
            self.index_list = idxs[:split1]
        elif self.stage == 'valid':
            self.index_list = idxs[split1:split2]
        elif self.stage == 'test':
            self.index_list = idxs[split2:]
        elif self.stage == 'all':
            self.index_list = idxs
        else:
            raise ValueError('invalid stage setting')

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        index = self.index_list[idx]
        data_lo = np.load(self.Data_dir + self.Data_list_lo[index]).astype(np.float32)
        data_hi = np.load(self.Data_dir + self.Data_list_hi[index]).astype(np.float32)
        if self.stage == 'train':
            patch_lo, patch_hi = self.patchsampler.random_sample(data_lo, data_hi)
            return np.expand_dims(patch_lo, axis=0), np.expand_dims(patch_hi, axis=0), self.Label_list[index]
        else:
            return np.expand_dims(data_lo, axis=0), np.expand_dims(data_hi, axis=0), self.Label_list[index]


if __name__ == "__main__":
    dataset = GAN_Data(Data_dir='/data/datasets/ADNI_NoBack/', stage='test')
    dataloader = DataLoader(dataset, batch_size=10)
    for scan1, scan2, label in dataloader:
        print(scan1.shape, scan2.shape, label.shape)
        numpy_label = label.numpy()
        index = torch.LongTensor(np.argwhere(numpy_label!=2).squeeze())
        print(label, index)
        selected_scan = torch.index_select(scan1, 0, index)
        selected_label = torch.index_select(label, 0, index)
        print(selected_scan.shape, selected_label)

    # dataset = Data(Data_dir='/data/datasets/ADNI_NoBack/', class1='ADNI_1.5T_GAN_NL', class2='ADNI_1.5T_GAN_AD', stage='train')
    # labels = []
    # print(dataset.get_sample_weights())
    # for i in range(len(dataset)):
    #     scan, label = dataset[i]
    #     labels.append(label)
    # print(labels)
    # dataloader = DataLoader(dataset, batch_size=10, shuffle=True, drop_last=True)
    # for scan, label in dataloader:
    #     print(scan.shape, label)
