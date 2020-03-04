import os
from os import path
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matlab.engine
import shutil
from torch.utils.data import Dataset, DataLoader
from dataloader import Data, GAN_Data, CNN_Data, FCN_Data
from models import Vanila_CNN, Vanila_CNN_Lite, _netD, _netG, _FCN
from utils import *
from tqdm import tqdm
from visual import GAN_test_plot
from statistics import mean
from sklearn.metrics import accuracy_score

"""
1. augmentation, small rotation,
2. hyperparameter auto tunning
3. early stopping or set epoch
"""

class CNN_Wrapper:
    def __init__(self, fil_num, drop_rate, seed, batch_size, balanced, Data_dir, exp_idx, model_name, metric):
        self.seed = seed
        self.exp_idx = exp_idx
        self.Data_dir = Data_dir
        self.model_name = model_name
        self.eval_metric = get_accu if metric == 'accuracy' else get_MCC
        self.model = Vanila_CNN_Lite(fil_num=fil_num, drop_rate=drop_rate).cuda()
        self.prepare_dataloader(batch_size, balanced, Data_dir)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.checkpoint_dir = './checkpoint_dir/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

    def train(self, lr, epochs, verbose=0):
        self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, self.imbalanced_ratio])).cuda()
        self.optimal_valid_matrix = [[0, 0], [0, 0]]
        self.optimal_valid_metric = 0
        self.optimal_epoch        = -1
        for self.epoch in range(epochs):
            self.train_model_epoch()
            valid_matrix = self.valid_model_epoch()
            if verbose >= 2:
                print('{}th epoch validation confusion matrix:'.format(self.epoch), valid_matrix, 'eval_metric:', "%.4f" % self.eval_metric(valid_matrix))
            self.save_checkpoint(valid_matrix)
        if verbose >= 2:
            print('Best model saved at the {}th epoch:'.format(self.optimal_epoch), self.optimal_valid_metric, self.optimal_valid_matrix)
        return self.optimal_valid_metric

    def test(self):
        print('testing ... ')
        self.model.load_state_dict(torch.load('{}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch)))
        self.model.train(False)
        with torch.no_grad():
            for stage in ['train', 'valid', 'test', 'AIBL', 'NACC']:
                Data_dir = self.Data_dir
                if stage in ['AIBL', 'NACC']:
                    Data_dir = Data_dir.replace('ADNI', stage)
                data = CNN_Data(Data_dir, self.exp_idx, stage=stage, seed=self.seed)
                dataloader = DataLoader(data, batch_size=10, shuffle=False)
                f = open(self.checkpoint_dir + 'raw_score_{}.txt'.format(stage), 'w')
                matrix = [[0, 0], [0, 0]]
                for idx, (inputs, labels) in enumerate(dataloader):
                    inputs, labels = inputs.cuda(), labels.cuda()
                    preds = self.model(inputs)
                    write_raw_score(f, preds, labels)
                    matrix = matrix_sum(matrix, get_confusion_matrix(preds, labels))
                print(stage + ' confusion matrix ', matrix, ' accuracy ', self.eval_metric(matrix))
                f.close()

    def save_checkpoint(self, valid_matrix):
        if self.eval_metric(valid_matrix) >= self.optimal_valid_metric:
            self.optimal_epoch = self.epoch
            self.optimal_valid_matrix = valid_matrix
            self.optimal_valid_metric = self.eval_metric(valid_matrix)
            for root, Dir, Files in os.walk(self.checkpoint_dir):
                for File in Files:
                    if File.endswith('.pth'):
                        try:
                            os.remove(self.checkpoint_dir + File)
                        except:
                            pass
            torch.save(self.model.state_dict(), '{}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))

    def train_model_epoch(self):
        self.model.train(True)
        for inputs, labels in self.train_dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            self.model.zero_grad()
            preds = self.model(inputs)
            loss = self.criterion(preds, labels)
            loss.backward()
            self.optimizer.step()

    def valid_model_epoch(self):
        with torch.no_grad():
            self.model.train(False)
            valid_matrix = [[0, 0], [0, 0]]
            for inputs, labels in self.valid_dataloader:
                inputs, labels = inputs.cuda(), labels.cuda()
                preds = self.model(inputs)
                valid_matrix = matrix_sum(valid_matrix, get_confusion_matrix(preds, labels))
        return valid_matrix

    def prepare_dataloader(self, batch_size, balanced, Data_dir):
        train_data = CNN_Data(Data_dir, self.exp_idx, stage='train', seed=self.seed)
        valid_data = CNN_Data(Data_dir, self.exp_idx, stage='valid', seed=self.seed)
        test_data  = CNN_Data(Data_dir, self.exp_idx, stage='test', seed=self.seed)
        sample_weight, self.imbalanced_ratio = train_data.get_sample_weights()
        # the following if else blocks represent two ways of handling class imbalance issue
        if balanced == 1:
            sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(sample_weight))
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
            self.imbalanced_ratio = 1
        elif balanced == 0:
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        self.valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
        self.test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

class FCN_Wrapper(CNN_Wrapper):
    def __init__(self, fil_num, drop_rate, seed, batch_size, balanced, Data_dir, exp_idx, model_name, metric, patch_size, lr):
        self.seed = seed
        self.exp_idx = exp_idx
        self.Data_dir = Data_dir
        self.patch_size = patch_size
        self.model_name = model_name
        self.eval_metric = get_accu if metric == 'accuracy' else get_MCC
        self.model = _FCN(num=fil_num, p=drop_rate).cuda()
        self.prepare_dataloader(batch_size, balanced, Data_dir)
        self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, self.imbalanced_ratio])).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.999))
        self.checkpoint_dir = './checkpoint_dir/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.DPMs_dir = './DPMs/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.DPMs_dir):
            os.mkdir(self.DPMs_dir)

    def train(self, epochs):
        self.optimal_valid_matrix = [[0, 0], [0, 0]]
        self.optimal_valid_metric = 0
        self.optimal_epoch        = -1
        for self.epoch in range(epochs):
            self.train_model_epoch()
            if self.epoch % 20 == 0:
                valid_matrix = self.valid_model_epoch()
                print('{}th epoch validation confusion matrix:'.format(self.epoch), valid_matrix, 'eval_metric:', "%.4f" % self.eval_metric(valid_matrix))
                self.save_checkpoint(valid_matrix)
        print('Best model saved at the {}th epoch:'.format(self.optimal_epoch), self.optimal_valid_metric, self.optimal_valid_matrix)
        return self.optimal_valid_metric

    def valid_model_epoch(self):
        self.fcn = self.model.dense_to_conv()
        DPMs, Labels = [], []
        with torch.no_grad():
            self.fcn.train(False)
            for idx, (inputs, labels) in enumerate(self.valid_dataloader):
                inputs, labels = inputs.cuda(), labels.cuda()
                DPM = self.fcn(inputs, stage='inference')
                DPMs.append(DPM.cpu().numpy().squeeze())
                Labels.append(labels)
        valid_matrix, ACCU, F1, MCC = DPM_statistics(DPMs, Labels)
        return valid_matrix

    def prepare_dataloader(self, batch_size, balanced, Data_dir):
        train_data = FCN_Data(Data_dir, self.exp_idx, stage='train', seed=self.seed, patch_size=self.patch_size)
        valid_data = FCN_Data(Data_dir, self.exp_idx, stage='valid', seed=self.seed, patch_size=self.patch_size)
        test_data  = FCN_Data(Data_dir, self.exp_idx, stage='test', seed=self.seed, patch_size=self.patch_size)
        sample_weight, self.imbalanced_ratio = train_data.get_sample_weights()
        if balanced == 1:
            sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(sample_weight))
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
            self.imbalanced_ratio = 1
        elif balanced == 0:
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        self.valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False)
        self.test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
        self.valid_patch_dataloader = FCN_Data(Data_dir, self.exp_idx, stage='valid_patch', seed=self.seed, patch_size=self.patch_size)

    def test_and_generate_DPMs(self):
        print('testing and generating DPMs ... ')
        self.model.load_state_dict(torch.load('{}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch)))
        self.fcn = self.model.dense_to_conv()
        self.fcn.train(False)
        with torch.no_grad():
            for stage in ['train', 'valid', 'test', 'AIBL', 'NACC', 'FHS']:
                Data_dir = self.Data_dir
                if stage in ['AIBL', 'NACC', 'FHS']:
                    Data_dir = Data_dir.replace('ADNI', stage)
                data = FCN_Data(Data_dir, self.exp_idx, stage=stage, whole_volume=True, seed=self.seed, patch_size=self.patch_size)
                filenames = data.Data_list
                dataloader = DataLoader(data, batch_size=1, shuffle=False)
                DPMs, Labels = [], []
                for idx, (inputs, labels) in enumerate(dataloader):
                    inputs, labels = inputs.cuda(), labels.cuda()
                    DPM = self.fcn(inputs, stage='inference').cpu().numpy().squeeze()
                    np.save(self.DPMs_dir + filenames[idx] + '.npy', DPM)
                    DPMs.append(DPM)
                    Labels.append(labels)
                matrix, ACCU, F1, MCC = DPM_statistics(DPMs, Labels) 
                np.save(self.DPMs_dir + '{}_MCC.npy'.format(stage), MCC)
                np.save(self.DPMs_dir + '{}_F1.npy'.format(stage),  F1)
                np.save(self.DPMs_dir + '{}_ACCU.npy'.format(stage), ACCU)  
                print(stage + ' confusion matrix ', matrix, ' accuracy ', self.eval_metric(matrix))
        print('DPM generation is done')

class GAN:
    # 1. Larger input for classifier
    #       Already done (√)
    # 2. Larger validation set
    #       Comes from the CNN's validation set+train set(?), instead of the 82 cases (√)
    # 3. Train CNN in process & use Trained CNN (CNN*) directly (√?)
    def __init__(self, config, seed):
        self.seed = seed
        self.config = read_json(config)
        self.netG = _netG(self.config["G_fil_num"]).cuda()
        self.netD = _netD(self.config["D_fil_num"]).cuda()
        self.cnn = self.initial_CNN('./cnn_config.json', exp_idx=0, epoch=100)
        if self.config["D_pth"]:
            self.netD.load_state_dict(torch.load(self.config["D_pth"]))
        if self.config["G_pth"]:
            self.netD.load_state_dict(torch.load(self.config["G_pth"]))
        self.checkpoint_dir = self.config["checkpoint_dir"]
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.prepare_dataloader()
        self.log_name = self.config["log_name"]
        self.eng = matlab.engine.start_matlab()
        self.save_every_epoch = self.config["save_every_epoch"]

    def prepare_dataloader(self):
        self.train_dataloader = DataLoader(GAN_Data(self.config['Data_dir'], seed=self.seed, stage='train_p'), batch_size=self.config['batch_size_p'], shuffle=True)

    def initial_CNN(self, json_file, exp_idx, epoch):
        cnn_setting = read_json(json_file)['cnn']
        cnn = CNN_Wrapper(fil_num       = cnn_setting['fil_num'],
                        drop_rate       = cnn_setting['drop_rate'],
                        batch_size      = cnn_setting['batch_size'],
                        balanced        = cnn_setting['balanced'],
                        Data_dir        = cnn_setting['Data_dir'],
                        exp_idx         = exp_idx,
                        seed            = 1000,
                        model_name      = 'cnn_gan',
                        metric          = 'accuracy')
        cnn.model.train(True)
        cnn.model.load_state_dict(torch.load('./checkpoint_dir/cnn_exp{}/cnn_{}.pth'.format(exp_idx, epoch)))
        return cnn

    def train(self):
        self.log = open(self.log_name, 'w')
        self.log.close()
        self.G_lr, self.D_lr = self.config["G_lr"], self.config["D_lr"]
        self.optimizer_G = optim.SGD([ {'params': self.netG.conv1.parameters()},
                          {'params': self.netG.conv2.parameters()},
                          {'params': self.netG.conv3.parameters(), 'lr': 0.1*self.G_lr},
                          {'params': self.netG.BN1.parameters(), 'lr':self.G_lr},
                          {'params': self.netG.BN2.parameters(), 'lr':self.G_lr},
                        ], lr=self.G_lr, momentum=0.9)
        self.optimizer_D = optim.Adam(self.netD.parameters(), lr=self.D_lr, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss().cuda()
        self.crossentropy = nn.CrossEntropyLoss().cuda()
        self.valid_optimal_metric, self.optimal_epoch = 0, -1
        for self.epoch in range(self.config['epochs']):
            cond1 = self.epoch<self.config["warm_G_epoch"]
            cond2 = self.config["warm_G_epoch"]<=self.epoch<=self.config["warm_D_epoch"]
            self.train_model_epoch(warmup_G=cond1, warmup_D=cond2)
            if self.epoch % self.save_every_epoch == 0:
                valid_metric = self.valid_model_epoch()
                with open(self.log_name, 'a') as f:
                    f.write('validation accuracy {} \n'.format(valid_metric))
                print('validation accuracy {}'.format(valid_metric))
                self.save_checkpoint(valid_metric)
        return self.valid_optimal_metric

    def generate(self):
        # generate & save 1.5T* images using MRIGAN
        sources = ["/data/datasets/ADNI_NoBack/", "/data/datasets/NACC_NoBack/", "/data/datasets/AIBL_NoBack/"]
        targets = ["./ADNIP_NoBack/", "./NACCP_NoBack/", "./AIBLP_NoBack/"]

        data = []
        data += [Data(sources[0], class1='ADNI_1.5T_NL', class2='ADNI_1.5T_AD', stage='all', shuffle=False)]
        data += [Data(sources[1], class1='NACC_1.5T_NL', class2='NACC_1.5T_AD', stage='all', shuffle=False)]
        data += [Data(sources[2], class1='AIBL_1.5T_NL', class2='AIBL_1.5T_AD', stage='all', shuffle=False)]
        dataloaders = [DataLoader(d, batch_size=1, shuffle=False) for d in data]
        Data_lists = [d.Data_list for d in data]
        # print('Generating 1.5T+ images for datasets: ADNI, NACC, FHS, AIBL')

        with torch.no_grad():
            self.netG.train(False)
            for i in range(len(dataloaders)):
                dataloader = dataloaders[i]
                target = targets[i]
                Data_list = Data_lists[i]
                for j, (input, label) in enumerate(dataloader):
                    input = input.cuda()
                    mask = self.netG(input)
                    output = input + mask
                    if not os.path.isdir(target):
                        os.mkdir(target)
                    np.save(target+Data_list[j], output.data.cpu().numpy().squeeze())

    def train_model_epoch(self, warmup_G=False, warmup_D=False):
        self.netG.train(True)
        self.cnn.model.train(True)
        self.netD.train(True)
        for idx, ((patch_lo, patch_hi), (whole_lo, AD_label)) in enumerate(zip(self.train_dataloader, self.cnn.train_dataloader)):
            # get gradient for D network with fake data
            self.netD.zero_grad()
            patch_lo, patch_hi = patch_lo.cuda(), patch_hi.cuda()
            Mask = self.netG(patch_lo)
            Output = patch_lo + Mask
            Foutput = self.netD(Output.detach())
            label = torch.FloatTensor(Foutput.shape[0])
            Flabel = label.fill_(0).cuda()
            loss_D_F = self.criterion(Foutput, Flabel)
            loss_D_F.backward()
            # get gradient for D network with real data
            Routput = self.netD(patch_hi)
            Rlabel = label.fill_(1).cuda()
            loss_D_R = self.criterion(Routput, Rlabel)
            loss_D_R.backward()
            if not warmup_G:
                self.optimizer_D.step()
            if idx % self.config['D_G_ratio'] != 0:
                continue
            #######################################
            # get gradient for G network
            self.netG.zero_grad()
            Goutput = self.netD(Output)
            Glabel = label.fill_(1).cuda()
            loss_G_GAN = self.criterion(Goutput, Glabel)
            # get gradient for G network with L1 norm between real and fake
            loss_G_dif = torch.mean(torch.abs(Mask))
            AD_loss = torch.tensor(0)
            if warmup_G:
                loss_G = self.config["L1_norm_factor"] * loss_G_dif
            else:
                # forward output to CNN to get AD loss
                whole_lo, AD_label = whole_lo.cuda(), AD_label.cuda()
                for a in range(whole_lo.shape[0]):
                    whole_l, AD_l = whole_lo[a:a+1], AD_label[a:a+1]
                    pred = self.cnn.model(whole_l + self.netG(whole_l))
                    AD_loss = self.config['AD_factor'] * self.crossentropy(pred, AD_l)
                    AD_loss.backward()
                self.cnn.optimizer.step()
                loss_G = self.config["L1_norm_factor"] * loss_G_dif + loss_G_GAN
            loss_G.backward()
            if not warmup_D:
                self.optimizer_G.step()

            if self.epoch % self.save_every_epoch == 0 or (self.epoch % self.save_every_epoch == 0 and self.epoch > self.config["warm_D_epoch"]):
                with open(self.log_name, 'a') as f:
                    out = 'epoch '+str(self.epoch)+': '+('[%d/%d][%d/%d] D(x): %.4f D(G(z)): %.4f / %.4f Mask L1_norm: %.4f loss_G: %.4f loss_D: %.4f loss_AD: %.4f \n'
                          % (self.epoch, self.config['epochs'], idx, len(self.train_dataloader), Routput.data.cpu().mean(),
                             Foutput.data.cpu().mean(), Goutput.data.cpu().mean(), loss_G_dif.data.cpu().mean(), loss_G.data.cpu().sum().item(), (loss_D_R+loss_D_F).data.cpu().sum().item(), AD_loss.data.cpu().sum().item()))
                    f.write(out)
                print('[%d/%d][%d/%d] D(x): %.4f D(G(z)): %.4f / %.4f Mask L1_norm: %.4f loss_G: %.4f loss_D: %.4f loss_AD: %.4f'
                      % (self.epoch, self.config['epochs'], idx, len(self.train_dataloader), Routput.data.cpu().mean(),
                         Foutput.data.cpu().mean(), Goutput.data.cpu().mean(), loss_G_dif.data.cpu().mean(), loss_G.data.cpu().sum().item(), (loss_D_R+loss_D_F).data.cpu().sum().item(), AD_loss.data.cpu().sum().item()))

    def valid_model_epoch(self):
        with torch.no_grad():
            self.cnn.model.train(False)
            valid_matrix = [[0, 0], [0, 0]]
            for idx, (inputs, labels) in enumerate(self.cnn.valid_dataloader):
                inputs, labels = inputs.cuda(), labels.cuda()
                output = self.netG(inputs) + inputs
                if idx == 0:
                    self.gen_output_image(output, self.epoch)
                preds = self.cnn.model(output)
                valid_matrix = matrix_sum(valid_matrix, get_confusion_matrix(preds, labels))
        return get_accu(valid_matrix)

    def gen_output_image(self, tensor, epoch):
        tensor = tensor.data.cpu().numpy()
        if not os.path.exists('./output/'):
            os.mkdir('./output/')
        plt.imshow(tensor[0, 0, :, 100, :], cmap='gray', vmin=-1, vmax=2.5)
        name = '#'.join(self.image_quality(tensor[0, 0, :, 100, :]))
        plt.savefig('./output/{}#{}.png'.format(epoch, name))

    def image_quality(self, img):
        img = matlab.double(img.tolist())
        vals = ['niqe', self.eng.niqe(img), 'piqe', self.eng.piqe(img), 'brisque', self.eng.brisque(img)]
        vals = map(str, vals)
        return vals

    def get_checkpoint_dir(self):
        return self.checkpoint_dir

    def save_checkpoint(self, valid_metric):
        if valid_metric >= self.valid_optimal_metric or True:
            self.optimal_epoch = self.epoch
            self.valid_optimal_metric = valid_metric
            for root, Dir, Files in os.walk(self.checkpoint_dir):
                for File in Files:
                    if File.endswith('.pth'):
                        try:
                            # os.remove(self.checkpoint_dir + File)
                            pass
                        except:
                            pass
            torch.save(self.netG.state_dict(), '{}G_{}.pth'.format(self.checkpoint_dir, self.optimal_epoch))
            torch.save(self.netD.state_dict(), '{}D_{}.pth'.format(self.checkpoint_dir, self.optimal_epoch))
            torch.save(self.cnn.model.state_dict(), '{}cnn_{}.pth'.format(self.checkpoint_dir, self.optimal_epoch))
            # print('model saved in', self.checkpoint_dir)

class FCN_GAN:
    def __init__(self, config, seed):
        self.seed = seed
        self.config = read_json(config)
        self.netG = _netG(self.config["G_fil_num"]).cuda()
        self.netD = _netD(self.config["D_fil_num"]).cuda()
        self.fcn = self.initial_FCN('./cnn_config.json', exp_idx=0)
        if self.config["D_pth"]:
            self.netD.load_state_dict(torch.load(self.config["D_pth"]))
        if self.config["G_pth"]:
            self.netD.load_state_dict(torch.load(self.config["G_pth"]))
        self.checkpoint_dir = self.config["checkpoint_dir"]
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.prepare_dataloader()
        self.log_name = self.config["log_name"]
        self.eng = matlab.engine.start_matlab()
        self.save_every_epoch = self.config["save_every_epoch"]

    def initial_FCN(self, json_file, exp_idx, epoch=None):
        fcn_setting = read_json(json_file)['fcn']
        fcn = FCN_Wrapper(fil_num         = fcn_setting['fil_num'],
                          drop_rate       = fcn_setting['drop_rate'],
                          batch_size      = fcn_setting['batch_size'],
                          balanced        = fcn_setting['balanced'],
                          Data_dir        = fcn_setting['Data_dir'],
                          patch_size      = fcn_setting['patch_size'],
                          lr              = fcn_setting['learning_rate'],
                          exp_idx         = exp_idx,
                          seed            = 1000,
                          model_name      = 'fcn_gan',
                          metric          = 'accuracy')
        fcn.model.train(True)
        if epoch:
            fcn.model.load_state_dict(torch.load('./checkpoint_dir/fcn_exp{}/fcn_{}.pth'.format(exp_idx, epoch)))
        return fcn

    def prepare_dataloader(self):
        self.gan_train_dataloader = DataLoader(GAN_Data(self.config['Data_dir'], seed=self.seed, stage='train_p'), batch_size=self.config['batch_size_p'], shuffle=True, drop_last=True)
        self.AD_train_dataloader = self.fcn.train_dataloader
        self.AD_valid_dataloader = self.fcn.valid_patch_dataloader

    def train(self):
        self.log = open(self.log_name, 'w')
        self.log.close()
        self.G_lr, self.D_lr = self.config["G_lr"], self.config["D_lr"]
        self.optimizer_G = optim.SGD([ {'params': self.netG.conv1.parameters()},
                          {'params': self.netG.conv2.parameters()},
                          {'params': self.netG.conv3.parameters(), 'lr': 0.1*self.G_lr},
                          {'params': self.netG.BN1.parameters(), 'lr':self.G_lr},
                          {'params': self.netG.BN2.parameters(), 'lr':self.G_lr},
                        ], lr=self.G_lr, momentum=0.9)
        self.optimizer_D = optim.Adam(self.netD.parameters(), lr=self.D_lr, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss().cuda()
        self.crossentropy = nn.CrossEntropyLoss().cuda()
        self.valid_optimal_metric, self.optimal_epoch = 0, -1
        for self.epoch in range(self.config['epochs']):
            cond1 = self.epoch<self.config["warm_G_epoch"]
            cond2 = self.config["warm_G_epoch"]<=self.epoch<=self.config["warm_D_epoch"]
            self.train_model_epoch(warmup_G=cond1, warmup_D=cond2)
            if self.epoch % self.save_every_epoch == 0:
                valid_metric = self.valid_model_epoch()
                with open(self.log_name, 'a') as f:
                    f.write('validation accuracy {} \n'.format(valid_metric))
                print('validation accuracy {}'.format(valid_metric))
                self.save_checkpoint(valid_metric)
        return self.valid_optimal_metric

    def train_model_epoch(self, warmup_G=False, warmup_D=False):
        self.netG.train(True)
        self.fcn.model.train(True)
        self.netD.train(True)
        for idx, ((patch_lo, patch_hi), (patch, AD_label)) in enumerate(zip(self.gan_train_dataloader, self.AD_train_dataloader)):
            # get gradient for D network with fake data
            self.netD.zero_grad()
            patch_lo, patch_hi, patch, AD_label = patch_lo.cuda(), patch_hi.cuda(), patch.cuda(), AD_label.cuda()
            Mask = self.netG(patch_lo)
            Output = patch_lo + Mask
            Foutput = self.netD(Output.detach())
            label = torch.FloatTensor(Foutput.shape[0])
            Flabel = label.fill_(0).cuda()
            loss_D_F = self.criterion(Foutput, Flabel)
            loss_D_F.backward()
            # get gradient for D network with real data
            Routput = self.netD(patch_hi)
            Rlabel = label.fill_(1).cuda()
            loss_D_R = self.criterion(Routput, Rlabel)
            loss_D_R.backward()
            if not warmup_G:
                self.optimizer_D.step()
            if idx % self.config['D_G_ratio'] != 0:
                continue
            #######################################
            # get gradient for G network
            self.netG.zero_grad()
            Goutput = self.netD(Output)
            Glabel = label.fill_(1).cuda()
            loss_G_GAN = self.criterion(Goutput, Glabel)
            # get gradient for G network with L1 norm between real and fake
            loss_G_dif = torch.mean(torch.abs(Mask))
            # get gradient for G network and FCN from AD_loss
            pred = self.fcn.model(patch)
            AD_loss = self.fcn.criterion(pred, AD_label)

            if warmup_G:
                loss_G = self.config["L1_norm_factor"] * loss_G_dif
            else:              
                loss_G = self.config["L1_norm_factor"] * loss_G_dif + loss_G_GAN + self.config["L1_norm_factor"] * AD_loss
            loss_G.backward()

            self.fcn.optimizer.step()
            if not warmup_D:
                self.optimizer_G.step()

            if self.epoch % self.save_every_epoch == 0 or (self.epoch % self.save_every_epoch == 0 and self.epoch > self.config["warm_D_epoch"]):
                with open(self.log_name, 'a') as f:
                    out = 'epoch '+str(self.epoch)+': '+('[%d/%d][%d/%d] D(x): %.4f D(G(z)): %.4f / %.4f Mask L1_norm: %.4f loss_G: %.4f loss_D: %.4f loss_AD: %.4f \n'
                          % (self.epoch, self.config['epochs'], idx, len(self.gan_train_dataloader), Routput.data.cpu().mean(),
                             Foutput.data.cpu().mean(), Goutput.data.cpu().mean(), loss_G_dif.data.cpu().mean(), loss_G.data.cpu().sum().item(), (loss_D_R+loss_D_F).data.cpu().sum().item(), AD_loss.data.cpu().sum().item()))
                    f.write(out)
                print('[%d/%d][%d/%d] D(x): %.4f D(G(z)): %.4f / %.4f Mask L1_norm: %.4f loss_G: %.4f loss_D: %.4f loss_AD: %.4f'
                      % (self.epoch, self.config['epochs'], idx, len(self.gan_train_dataloader), Routput.data.cpu().mean(),
                         Foutput.data.cpu().mean(), Goutput.data.cpu().mean(), loss_G_dif.data.cpu().mean(), loss_G.data.cpu().sum().item(), (loss_D_R+loss_D_F).data.cpu().sum().item(), AD_loss.data.cpu().sum().item()))

    def valid_model_epoch(self):
        # forward 5 representative patches into netG and then fcn to get accuracy on validation set
        with torch.no_grad():
            self.fcn.model.train(False)
            self.netG.train(False)
            valid_matrix = [[0, 0], [0, 0]]
            for idx, (patches, labels) in enumerate(self.AD_valid_dataloader):
                patches, labels = patches.cuda(), labels.cuda()
                patches = self.netG(patches) + patches
                if idx == 0:
                    self.gen_output_image(patches, self.epoch)
                preds = self.fcn.model(patches)
                valid_matrix = matrix_sum(valid_matrix, get_confusion_matrix(preds, labels))
        return get_accu(valid_matrix)

    def gen_output_image(self, tensor, epoch):
        tensor = tensor.data.cpu().numpy()
        if not os.path.exists('./output/'):
            os.mkdir('./output/')
        plt.imshow(tensor[0, 0, :, 20, :], cmap='gray', vmin=-1, vmax=2.5)
        name = '#'.join(self.image_quality(tensor[0, 0, :, 20, :]))
        plt.savefig('./output/{}#{}.png'.format(epoch, name))

    def image_quality(self, img):
        img = matlab.double(img.tolist())
        vals = ['niqe', self.eng.niqe(img), 'piqe', self.eng.piqe(img), 'brisque', self.eng.brisque(img)]
        vals = map(str, vals)
        return vals

    def save_checkpoint(self, valid_metric):
        if valid_metric >= self.valid_optimal_metric or True:
            self.optimal_epoch = self.epoch
            self.valid_optimal_metric = valid_metric
            torch.save(self.netG.state_dict(), '{}G_{}.pth'.format(self.checkpoint_dir, self.optimal_epoch))
            torch.save(self.netD.state_dict(), '{}D_{}.pth'.format(self.checkpoint_dir, self.optimal_epoch))
            torch.save(self.fcn.model.state_dict(), '{}fcn_{}.pth'.format(self.checkpoint_dir, self.optimal_epoch))

    def generate(self):
        # generate & save 1.5T* images using MRIGAN
        sources = ["/data/datasets/ADNI_NoBack/", "/data/datasets/NACC_NoBack/", "/data/datasets/AIBL_NoBack/"]
        targets = ["./ADNIP_NoBack/", "./NACCP_NoBack/", "./AIBLP_NoBack/"]

        data = []
        data += [Data(sources[0], class1='ADNI_1.5T_NL', class2='ADNI_1.5T_AD', stage='all', shuffle=False)]
        data += [Data(sources[1], class1='NACC_1.5T_NL', class2='NACC_1.5T_AD', stage='all', shuffle=False)]
        data += [Data(sources[2], class1='AIBL_1.5T_NL', class2='AIBL_1.5T_AD', stage='all', shuffle=False)]
        dataloaders = [DataLoader(d, batch_size=1, shuffle=False) for d in data]
        Data_lists = [d.Data_list for d in data]

        with torch.no_grad():
            self.netG.train(False)
            for i in range(len(dataloaders)):
                dataloader = dataloaders[i]
                target = targets[i]
                Data_list = Data_lists[i]
                for j, (input, label) in enumerate(dataloader):
                    input = input.cuda()
                    mask = self.netG(input)
                    output = input + mask
                    if not os.path.isdir(target):
                        os.mkdir(target)
                    np.save(target+Data_list[j], output.data.cpu().numpy().squeeze())


if __name__ == "__main__":
    gan = FCN_GAN('./gan_config_optimal.json', 0)
    gan.train()
    # gan.optimal_epoch=281
    # gan.test()

"""
train - plot metrics on validation
generate_1.5T+ numpy array
ADNI_test, NACC, AIBL


intermediate:

1.5T+ npy -> niqe npy (3, 10) -> boxplot; table; p-value

"""
