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
    def __init__(self, Data_dir, class1, class2, stage, ratio=(0.6, 0.2, 0.2), seed=1000):
        random.seed(seed)
        self.Data_dir = Data_dir
        Data_list0 = read_txt('../lookuptxt/', class1 + '.txt')
        Data_list1 = read_txt('../lookuptxt/', class2 + '.txt')
        self.Data_list = Data_list0 + Data_list1
        self.Label_list = [0]*len(Data_list0) + [1]*len(Data_list1)
        self.stage = stage
        self.length = len(self.Data_list)
        idxs = list(range(self.length))
        random.shuffle(idxs)
        split1, split2 = int(self.length*ratio[0]), int(self.length*(ratio[0]+ratio[1]))
        if self.stage == 'train':
            self.index_list = idxs[:split1]
        elif self.stage == 'valid':
            self.index_list = idxs[split1:split2]
        elif self.stage == 'test':
            self.index_list = idxs[split2:]
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

    def random_sample(self, data):
        """sample random patch from numpy array data"""
        X, Y, Z = data.shape
        x = random.randint(0, X-self.patch_size)
        y = random.randint(0, Y-self.patch_size)
        z = random.randint(0, Z-self.patch_size)
        return data[x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size]

    def fixed_sample(self, data):
        """sample patch from fixed locations"""
        patches = []
        patch_locs = [[25, 90, 30], [115, 90, 30], [67, 90, 90], [67, 45, 60], [67, 135, 60]]
        for i, loc in enumerate(patch_locs):
            x, y, z = loc
            patch = data[x:x+47, y:y+47, z:z+47]
            patches.append(np.expand_dims(patch, axis = 0))
        return patches 


class GAN_dataset(Data):
    # def __init__(self, whole, Data_dir, dset_name, magnetic, stage, ratio=(0.6, 0.2, 0.2), seed=1000):
    #     super().__init__(Data_dir, dset_name, magnetic, stage, ratio, seed)
    #     self.whole = whole

    def __init__(self, **kwargs):
        self.whole = kwargs.pop('whole')
        self.patchsampler = PatchGenerator(patch_size = 47)
        super().__init__(**kwargs)

    def __getitem__(self, idx):
        index = self.index_list[idx]
        data = np.load(self.Data_list[index]).astype(np.float32)
        label = self.Label_list[index]
        if not self.whole:
            if self.stage == 'train':
                patch = self.patchsampler.random_sample(data)
                return np.expand_dims(patch, axis=0), label
            elif self.stage == 'valid':
                patches = self.patchsampler.fixed_sample(data)
                data = Variable(torch.FloatTensor(np.stack(patches, axis = 0)))
                label = Variable(torch.LongTensor([label]*5))
                return data, label
        else:
            return np.expand_dims(data, axis=0), label
            
if __name__ == "__main__":
    dataset = Data(Data_dir='/data/datasets/ADNI_NoBack/', class1='ADNI_1.5T_GAN_NL', class2='ADNI_1.5T_GAN_AD', stage='train')
    labels = []
    print(dataset.get_sample_weights())
    for i in range(len(dataset)):
        scan, label = dataset[i]
        labels.append(label)
    print(labels)
    # dataloader = DataLoader(dataset, batch_size=10, shuffle=True, drop_last=True)
    # for scan, label in dataloader:
    #     print(scan.shape, label)



