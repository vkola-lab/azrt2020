import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matlab.engine
import shutil
from torch.utils.data import Dataset, DataLoader
from dataloader import Data, GAN_Data, CNN_Data
from models import Vanila_CNN, Vanila_CNN_Lite, _netD, _netG
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
        self.checkpoint_dir = './checkpoint_dir/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

    def train(self, lr, epochs, verbose=0):
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.999))
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


class GAN:
    def __init__(self, config, seed):
        self.seed = seed
        self.config = read_json(config)
        self.netG = _netG(self.config["G_fil_num"]).cuda()
        self.netD = _netD(self.config["D_fil_num"]).cuda()
        self.cnn = self.initial_CNN('./cnn_config.json', exp_idx=0, epoch=190)
        if self.config["D_pth"]:
            self.netD.load_state_dict(torch.load(self.config["D_pth"]))
        if self.config["G_pth"]:
            self.netD.load_state_dict(torch.load(self.config["G_pth"]))
        self.checkpoint_dir = self.config["checkpoint_dir"] + 'gan/'
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.prepare_dataloader()

    def prepare_dataloader(self):
        dataset = GAN_Data(self.config['Data_dir'], seed=self.seed, stage='train_w')
        sample_weight = dataset.get_sample_weights()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(sample_weight))
        self.train_w_dataloader = DataLoader(dataset, batch_size=self.config['batch_size_w'], sampler=sampler)
        self.train_p_dataloader = DataLoader(GAN_Data(self.config['Data_dir'], seed=self.seed, stage='train_p'), batch_size=self.config['batch_size_p'], shuffle=True)
        self.valid_dataloader = DataLoader(GAN_Data(self.config['Data_dir'], seed=self.seed, stage='valid'), batch_size=1, shuffle=False)
        self.test_dataloader = DataLoader(GAN_Data(self.config['Data_dir'], seed=self.seed, stage='test'), batch_size=1, shuffle=False)

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
        cnn.model.train(False)
        cnn.model.load_state_dict(torch.load('./checkpoint_dir/cnn_exp{}/cnn_{}.pth'.format(exp_idx, epoch)))
        return cnn

    def train(self):
        try:
            os.remove('log.txt')
        except:
            pass
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
            if self.epoch % 10 == 0:
                valid_metric = self.valid_model_epoch()
                print('validation accuracy ', valid_metric)
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

        # print('Generation completed!')

    @timeit
    def validate(self, plot=True, zoom=False):
        eng = matlab.engine.start_matlab()
        # print('#########Preparing validation results...#########')
        with torch.no_grad():
            self.netG.train(False)
            # do some plots and post analysis
            iqa_oris = []
            iqa_gens = []
            iqa_3s = []
            out_dir = './outputs/'+str(self.epoch)+'/'
            if os.path.isdir(out_dir):
                shutil.rmtree(out_dir)
            os.mkdir(out_dir)
            for idx, (inputs_lo, inputs_hi, label) in enumerate(self.valid_dataloader):
                inputs_lo, inputs_hi = inputs_lo.cuda(), inputs_hi.cuda()
                Mask = self.netG(inputs_lo)
                output = inputs_lo + Mask

                inputs_lo = inputs_lo.cpu().squeeze().numpy()
                output = output.cpu().squeeze().numpy()
                inputs_hi = inputs_hi.cpu().squeeze().numpy()

                iqa_funcs = {'ssim':SSIM, 'immse':immse, 'psnr':psnr, 'brisque':brisque, 'niqe':niqe, 'piqe':piqe}
                out_string = ''
                for metric in iqa_funcs:
                    if metric == 'ssim':
                        iqa_gen = iqa_funcs[metric](output, inputs_hi, zoom)
                        iqa_ori = iqa_funcs[metric](inputs_lo, inputs_hi, zoom)
                    elif metric == 'immse' or metric == 'psnr':
                        iqa_gen = iqa_funcs[metric](output, inputs_hi, zoom, eng)
                        iqa_ori = iqa_funcs[metric](inputs_lo, inputs_hi, zoom, eng)
                    elif metric == 'brisque' or metric == 'niqe' or metric == 'piqe':
                        iqa_gen = iqa_funcs[metric](output, zoom, eng)
                        iqa_ori = iqa_funcs[metric](inputs_lo, zoom, eng)
                        iqa_3 = iqa_funcs[metric](inputs_hi, zoom, eng)
                        iqa_3s += [iqa_3]
                    else:
                        iqa_gen = None
                        iqa_ori = None
                    iqa_oris += [iqa_ori]
                    iqa_gens += [iqa_gen]
                    out_string += '#'+metric+'#'+str(iqa_gen)
                if plot:
                    GAN_test_plot(out_dir, idx, inputs_lo, output, inputs_hi, out_string)
                print('ssim', 'immse', 'psnr', 'brisque', 'niqe', 'piqe')
                # iqa_oris = np.asarray(iqa_oris)
                # iqa_oris = iqa_oris.reshape(-1, 6)
                print(iqa_oris)
                print('brisque', 'niqe', 'piqe')
                # iqa_3s = np.asarray(iqa_3s)
                # iqa_3s = iqa_oris.reshape(-1, 3)
                print(iqa_3s)
        eng.quit()

    def train_model_epoch(self, warmup_G=False, warmup_D=False):
        self.netG.train(True)
        self.netD.train(True)
        for idx, ((patch_lo, patch_hi), (whole_lo, AD_label)) in enumerate(zip(self.train_p_dataloader, self.train_w_dataloader)):
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
            #######################################
            # get gradient for G network
            self.netG.zero_grad()
            Goutput = self.netD(Output)
            Glabel = label.fill_(1).cuda()
            loss_G_GAN = self.criterion(Goutput, Glabel)
            # get gradient for G network with L1 norm between real and fake
            loss_G_dif = torch.mean(torch.abs(Mask))
            # forward output to CNN to get AD loss
            AD_loss = 0
            whole_lo, AD_label = whole_lo.cuda(), AD_label.cuda()
            for a in range(whole_lo.shape[0]):
                whole_l, AD_l = whole_lo[a:a+1], AD_label[a:a+1]
                pred = self.cnn.model(whole_l + self.netG(whole_l))
                AD_loss += self.crossentropy(pred, AD_l)
            if warmup_G:
                loss_G = self.config["L1_norm_factor"] * loss_G_dif
            else:
                loss_G = self.config["L1_norm_factor"] * loss_G_dif + loss_G_GAN + self.config['AD_factor'] * AD_loss
            loss_G.backward()
            if not warmup_D:
                self.optimizer_G.step()

            if self.epoch % 10 == 0 or (self.epoch % 10 == 0 and self.epoch > self.config["warm_D_epoch"]):
                with open('log.txt', 'a') as f:
                    out = 'epoch '+str(self.epoch)+': '+('[%d/%d][%d/%d] D(x): %.4f D(G(z)): %.4f / %.4f Mask L1_norm: %.4f loss_G: %.4f loss_D: %.4f'
                          % (self.epoch, self.config['epochs'], idx, len(self.train_p_dataloader), Routput.data.cpu().mean(),
                             Foutput.data.cpu().mean(), Goutput.data.cpu().mean(), 1000*loss_G_dif.data.cpu().mean(), loss_G.data.cpu().sum().item(), (loss_D_R+loss_D_F).data.cpu().sum().item()))+'\n'
                    f.write(out)
                print('[%d/%d][%d/%d] D(x): %.4f D(G(z)): %.4f / %.4f Mask L1_norm: %.4f loss_G: %.4f loss_D: %.4f'
                      % (self.epoch, self.config['epochs'], idx, len(self.train_p_dataloader), Routput.data.cpu().mean(),
                         Foutput.data.cpu().mean(), Goutput.data.cpu().mean(), 1000*loss_G_dif.data.cpu().mean(), loss_G.data.cpu().sum().item(), (loss_D_R+loss_D_F).data.cpu().sum().item()))

    def valid_model_epoch(self):
        with torch.no_grad():
            self.cnn.model.train(False)
            valid_matrix = [[0, 0], [0, 0]]
            for inputs, inputs_high, labels in self.valid_dataloader:
                inputs, labels = inputs.cuda(), labels.cuda()
                output = self.netG(inputs) + inputs
                preds = self.cnn.model(output)
                valid_matrix = matrix_sum(valid_matrix, get_confusion_matrix(preds, labels))
        return get_accu(valid_matrix)

    def get_checkpoint_dir(self):
        return self.checkpoint_dir

    def save_checkpoint(self, valid_metric):
        if valid_metric >= self.valid_optimal_metric:
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
            # print('model saved in', self.checkpoint_dir)

    def eval_iqa_all(self, metrics=['brisque']):
        # print('Evaluating IQA results on all datasets:')
        eng = matlab.engine.start_matlab()

        data  = []
        names = ['ADNI', 'NACC', 'AIBL']

        sources = ["/data/datasets/ADNI_NoBack/", "/data/datasets/NACC_NoBack/", "/data/datasets/AIBL_NoBack/"]
        data += [Data(sources[0], class1='ADNI_1.5T_NL', class2='ADNI_1.5T_AD', stage='all', shuffle=False)]
        data += [Data(sources[1], class1='NACC_1.5T_NL', class2='NACC_1.5T_AD', stage='all', shuffle=False)]
        data += [Data(sources[2], class1='AIBL_1.5T_NL', class2='AIBL_1.5T_AD', stage='all', shuffle=False)]
        dataloaders = [DataLoader(d, batch_size=1, shuffle=False) for d in data]

        data = []
        targets = ["./ADNIP_NoBack/", "./NACCP_NoBack/", "./AIBLP_NoBack/"]
        data += [Data(targets[0], class1='ADNI_1.5T_NL', class2='ADNI_1.5T_AD', stage='all', shuffle=False)]
        data += [Data(targets[1], class1='NACC_1.5T_NL', class2='NACC_1.5T_AD', stage='all', shuffle=False)]
        data += [Data(targets[2], class1='AIBL_1.5T_NL', class2='AIBL_1.5T_AD', stage='all', shuffle=False)]
        dataloaders_p = [DataLoader(d, batch_size=1, shuffle=False) for d in data]

        Data_lists = [d.Data_list for d in data]

        out_dir = './iqa/'
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.mkdir(out_dir)

        for m in metrics:
            for id in range(len(names)):
                name = names[id]
                Data_list = Data_lists[id]
                dataloader = dataloaders[id]
                dataloader_p = dataloaders_p[id]
                iqa_oris = []
                iqa_gens = []
                for i, (input, _) in enumerate(dataloader):
                    input = input.squeeze().numpy()
                    iqa_tensor(input, eng, Data_list[i], m, out_dir)
                for j, (input_p, _) in enumerate(dataloader_p):
                    input_p = input_p.squeeze().numpy()
                    iqa_tensor(input_p, eng, Data_list[j], m, out_dir)
        eng.quit()


if __name__ == "__main__":
    gan = GAN('./gan_config_optimal.json', 0)
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
