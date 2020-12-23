from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
from utils import read_txt, read_csv, padding, read_csv_complete, get_AD_risk
import random
import copy
import matplotlib.pyplot as plt

"""
to do list:
    1. to make the dataloader work, create shuffled filename.txt and label.txt
    2. save different version for MRI scans: clip or background remove
    3. for those 82 cases, 1.5T has inconsistency between data/datasets/ADNI and /data/MRI_GAN/
    4. in filename.txt maybe put complete path of data, no need to assign data_dir
"""
class Augment:
    def __init__(self):
        self.contrast_factor = 0.2
        self.bright_factor = 0.4
        self.sig_factor = 0.2

    def change_contrast(self, image):
        ratio = 1 + (random.random() - 0.5)*self.contrast_factor
        return image.mean() + ratio*(image - image.mean())

    def change_brightness(self, image):
        val = (random.random() - 0.5)*self.bright_factor
        return image + val

    def add_noise(self, image):
        sig = random.random() * self.sig_factor
        return np.random.normal(0, sig, image.shape) + image

    def apply(self, image):
        image = self.change_contrast(image)
        image = self.change_brightness(image)
        image = self.add_noise(image)
        return image


class Data(Dataset):
    """
    txt files ./lookuptxt/*.txt complete path of MRIs
    MRI with clip and backremove: /data/datasets/ADNI_NoBack/*.npy
    """
    def __init__(self, Data_dir, class1, class2, stage, ratio=(0.6, 0.2, 0.2), seed=1000, shuffle=True):
        random.seed(seed)
        self.Data_dir = Data_dir

        if 'AIBL' in Data_dir:
            self.Data_list, self.Label_list = read_csv('./lookupcsv/{}.csv'.format('AIBL'))
            self.Data_list = [d+'.npy' for d in self.Data_list]
        else:
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


class CNN_Data(Dataset):
    """
    csv files ./lookuptxt/*.csv contains MRI filenames along with demographic and diagnosis information
    """
    def __init__(self, Data_dir, exp_idx, stage, seed=1000):
        random.seed(seed)
        self.Data_dir = Data_dir
        if stage in ['train', 'valid', 'test', 'valid_patch']:
            self.Data_list, self.Label_list = read_csv('./lookupcsv/exp{}/{}.csv'.format(exp_idx, stage.replace('_patch', '')))
        elif stage in ['ADNI', 'NACC', 'AIBL']:
            self.Data_list, self.Label_list = read_csv('./lookupcsv/{}.csv'.format(stage))

    def __len__(self):
        return len(self.Data_list)

    def get_filenames(self):
        return [i +'.npy' for i in self.Data_list]

    def __getitem__(self, idx):
        label = self.Label_list[idx]
        data = np.load(self.Data_dir + self.Data_list[idx] + '.npy').astype(np.float32)
        data = np.expand_dims(data, axis=0)
        return data, label

    def get_sample_weights(self):
        count, count0, count1 = float(len(self.Label_list)), float(self.Label_list.count(0)), float(self.Label_list.count(1))
        weights = [count / count0 if i == 0 else count / count1 for i in self.Label_list]
        return weights, count0 / count1


class FCN_Data(CNN_Data):
    def __init__(self, Data_dir, exp_idx, stage, transform=None, whole_volume=False, seed=1000, patch_size=47):
        CNN_Data.__init__(self, Data_dir, exp_idx, stage, seed)
        self.stage = stage
        self.transform = transform
        self.whole = whole_volume
        self.patch_size = patch_size
        self.patch_sampler = PatchGenerator(patch_size=self.patch_size)
        self.cache = []

    def __getitem__(self, idx):
        if self.whole:
            data = np.load(self.Data_dir + self.Data_list[idx] + '.npy').astype(np.float32)
            data = np.expand_dims(padding(data, win_size=self.patch_size // 2), axis=0)
            label = self.Label_list[idx]
            return data, label
        elif self.stage == 'valid_patch' and len(self.cache) == len(self.Label_list):
            return self.cache[idx]
        elif self.stage == 'valid_patch':
            label = self.Label_list[idx]
            data = np.load(self.Data_dir + self.Data_list[idx] + '.npy', mmap_mode='r').astype(np.float32)
            array_list = []
            patch_locs = [[25, 90, 30], [115, 90, 30], [67, 90, 90], [67, 45, 60], [67, 135, 60]]
            for i, loc in enumerate(patch_locs):
                x, y, z = loc
                patch = data[x:x+47, y:y+47, z:z+47]
                array_list.append(np.expand_dims(patch, axis = 0))
            data = Variable(torch.FloatTensor(np.stack(array_list, axis = 0)))
            label = Variable(torch.LongTensor([label]*5))
            self.cache.append((data, label))
            return data, label
        elif self.stage == 'train':
            label = self.Label_list[idx]
            data = np.load(self.Data_dir + self.Data_list[idx] + '.npy', mmap_mode='r').astype(np.float32)
            patch = self.patch_sampler.random_sample(data)
            if self.transform:
                patch = self.transform.apply(patch).astype(np.float32)
            patch = np.expand_dims(patch, axis=0)
            return patch, label

class PatchGenerator:
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def random_sample(self, data1, data2=None):
        """sample random patch from numpy array data"""
        X, Y, Z = data1.shape
        x = random.randint(0, X-self.patch_size)
        y = random.randint(0, Y-self.patch_size)
        z = random.randint(0, Z-self.patch_size)
        if data2 is None:
            return data1[x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size]
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


class GAN_Data(Dataset):
    def __init__(self, Data_dir, stage, ratio=(0.6, 0.2, 0.2), seed=1000):
        random.seed(seed)
        self.Data_dir = Data_dir
        Data_list0 = read_txt('./lookuptxt/', 'ADNI_1.5T_GAN_NL.txt')
        Data_list1 = read_txt('./lookuptxt/', 'ADNI_1.5T_GAN_MCI.txt')
        Data_list2 = read_txt('./lookuptxt/', 'ADNI_1.5T_GAN_AD.txt')
        Data_list3 = read_txt('./lookuptxt/', 'ADNI_3T_NL.txt')
        Data_list4 = read_txt('./lookuptxt/', 'ADNI_3T_MCI.txt')
        Data_list5 = read_txt('./lookuptxt/', 'ADNI_3T_AD.txt')
        self.Data_list_lo = Data_list0 + Data_list2 + Data_list1
        self.Data_list_hi = Data_list3 + Data_list5 + Data_list4
        self.Label_list = [0]*len(Data_list0) + [1]*len(Data_list2) + [2]*len(Data_list1)
        self.stage = stage
        self.length = len(self.Data_list_lo)
        self.patchsampler = PatchGenerator(patch_size = 47)
        idxs = list(range(self.length))
        random.shuffle(idxs)
        split1, split2 = int(self.length*ratio[0]), int(self.length*(ratio[0]+ratio[1]))
        if self.stage == 'train_p':
            self.index_list = idxs[:split1]
        elif self.stage == 'train_w':
            self.index_list = idxs[:split1]
        elif self.stage == 'valid':
            self.index_list = idxs[split1:split2]
        elif self.stage == 'test':
            self.index_list = idxs[split2:]
        elif self.stage == 'all':
            self.index_list = idxs
        else:
            raise ValueError('invalid stage setting')

    def get_filenames(self):
        return [self.Data_list_hi[idx] for idx in self.index_list], [self.Data_list_lo[idx] for idx in self.index_list]

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        index = self.index_list[idx]
        data_lo = np.load(self.Data_dir + self.Data_list_lo[index], mmap_mode='r').astype(np.float32)
        data_hi = np.load(self.Data_dir + self.Data_list_hi[index], mmap_mode='r').astype(np.float32)
        if self.stage == 'train_p':
            patch_lo, patch_hi = self.patchsampler.random_sample(data_lo, data_hi)
            return np.expand_dims(patch_lo, axis=0), np.expand_dims(patch_hi, axis=0)
        elif self.stage == 'train_w':
            return np.expand_dims(data_lo[:,:,:], axis=0), self.Label_list[index]
        else:
            return np.expand_dims(data_lo[:,:,:], axis=0), np.expand_dims(data_hi[:,:,:], axis=0), self.Label_list[index]


class MLP_Data(Dataset):
    def __init__(self, Data_dir, exp_idx, stage, roi_threshold, roi_count, choice, seed=1000):
        random.seed(seed)
        self.exp_idx = exp_idx
        self.Data_dir = Data_dir
        self.roi_threshold = roi_threshold
        self.roi_count = roi_count
        if choice == 'count':
            self.select_roi_count()
        else:
            self.select_roi_thres()
        if stage in ['train', 'valid', 'test']:
            self.path = './lookupcsv/exp{}/{}.csv'.format(exp_idx, stage)
        else:
            self.path = './lookupcsv/{}.csv'.format(stage)
        self.Data_list, self.Label_list, self.demor_list = read_csv_complete(self.path)
        self.risk_list = [get_AD_risk(np.load(Data_dir+filename+'.npy'))[self.roi] for filename in self.Data_list]
        self.in_size = self.risk_list[0].shape[0]

    def select_roi_thres(self):
        self.roi = np.load(self.Data_dir + 'train_MCC.npy')
        self.roi = self.roi > self.roi_threshold
        for i in range(self.roi.shape[0]):
            for j in range(self.roi.shape[1]):
                for k in range(self.roi.shape[2]):
                    if i%3!=0 or j%2!=0 or k%3!=0:
                        self.roi[i,j,k] = False

    def select_roi_count(self):
        self.roi = np.load(self.Data_dir + 'train_MCC.npy')
        tmp = []
        for i in range(self.roi.shape[0]):
            for j in range(self.roi.shape[1]):
                for k in range(self.roi.shape[2]):
                    if i%3!=0 or j%2!=0 or k%3!=0: continue
                    tmp.append((self.roi[i,j,k], i, j, k))
        tmp.sort()
        tmp = tmp[-self.roi_count:]
        self.roi = self.roi != self.roi
        for _, i, j, k in tmp:
            self.roi[i,j,k] = True

    def __len__(self):
        return len(self.Data_list)

    def __getitem__(self, idx):
        label = self.Label_list[idx]
        risk = self.risk_list[idx]
        demor = self.demor_list[idx]
        return risk, label, np.asarray(demor).astype(np.float32)

    def get_sample_weights(self):
        count, count0, count1 = float(len(self.Label_list)), float(self.Label_list.count(0)), float(self.Label_list.count(1))
        weights = [count / count0 if i == 0 else count / count1 for i in self.Label_list]
        return weights, count0 / count1



if __name__ == "__main__":
    transform = Augment()
    dataset = FCN_Data(Data_dir='/data/datasets/ADNI_NoBack/', exp_idx=0, stage='train', transform=None)
    dataset = CNN_Data(Data_dir='/data/datasets/ADNI_NoBack/', exp_idx=0, stage='train')
    train_dataloader = DataLoader(dataset, batch_size=1)
    for scan1, label in train_dataloader:
        print(scan1.shape)
        # scan1 = scan1.data.squeeze().numpy()
        # plt.imshow(scan1[20, :, :], cmap='gray', vmin=-1, vmax=2.5)
        # plt.show()


    # dataset = GAN_Data(Data_dir='/data/datasets/ADNI_NoBack/', stage='train_w')
    # sample_weight = dataset.get_sample_weights()
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(sample_weight))
    # train_w_dataloader = DataLoader(dataset, batch_size=3, sampler=sampler)
    # for scan1, scan2, label in train_w_dataloader:
    #     print(label)

    # dataloader = DataLoader(dataset, batch_size=10)
    # for scan1, scan2, label in dataloader:
    #     print(scan1.shape, scan2.shape, label.shape)
    #     numpy_label = label.numpy()
    #     index = torch.LongTensor(np.argwhere(numpy_label!=2).squeeze())
    #     print(label, index)
    #     selected_scan = torch.index_select(scan1, 0, index)
    #     selected_label = torch.index_select(label, 0, index)
    #     print(selected_scan.shape, selected_label)

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
