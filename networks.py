import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dataloader import Data, GAN_Data
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

class CNN:
    def __init__(self, config, seed):
        self.seed = seed
        self.config = read_json(config)
        self.model = Vanila_CNN_Lite(fil_num=self.config['fil_num'], drop_rate=self.config['drop_rate']).cuda()
        self.checkpoint_dir = self.config["checkpoint_dir"] + self.config['class1'] + '_' + self.config['class2'] + '_balance{}'.format(self.config['balanced']) + '/'
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        train_data = Data(self.config['Data_dir'], self.config['class1'], self.config['class2'], seed=seed, stage='train')
        sample_weight, self.imbalanced_ratio = train_data.get_sample_weights()

        # the following if else blocks represent two ways of handling class imbalance issue
        if self.config['balanced']:
            # if config['balanced'] == 1, use pytorch sampler to sample data with probability according to the count of each class
            # so that each mini-batch has the same expectation counts of samples from each class
            sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(sample_weight))
            self.train_dataloader = DataLoader(train_data, batch_size=self.config['batch_size'], sampler=sampler)
            self.imbalanced_ratio = 1
        else:
            # if config['balanced'] == 0, sample data from the same probability, but
            # self.imbalanced_ratio will be used in the weighted cross entropy loss to handle imbalanced issue
            self.train_dataloader = DataLoader(train_data, batch_size=self.config['batch_size'], shuffle=True, drop_last=True)

        self.valid_dataloader = DataLoader(Data(self.config['Data_dir'], self.config['class1'], self.config['class2'], seed=seed, stage='valid'),
                                           batch_size=self.config['batch_size'], shuffle=True)
        self.test_dataloader = DataLoader(Data(self.config['Data_dir'], self.config['class1'], self.config['class2'], seed=seed, stage='test'),
                                           batch_size=self.config['batch_size'], shuffle=True)

    def train(self, verbose=2):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['lr'], betas=(0.5, 0.999))
        self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, self.imbalanced_ratio])).cuda()
        self.optimal_valid_matrix = [[0,0],[0,0]]
        self.valid_optimal_accu, self.epoch = 0, -1
        for epoch in range(self.config['epochs']):
            self.train_model_epoch()
            valid_matrix = self.valid_model_epoch()
            if verbose > 1:
                print('This epoch validation confusion matrix:', valid_matrix, 'validation_accuracy:', "%.4f" % get_accu(valid_matrix))
            self.save_checkpoint(valid_matrix, epoch)
        if verbose > 0:
            print('(CNN) Best validation accuracy saved at the {}th epoch:'.format(self.epoch), self.valid_optimal_accu, self.valid_optimal_matrix)
        return self.valid_optimal_accu

    def test(self):
        f = open(self.checkpoint_dir + self.config['class1'] + 'vs' + self.config['class2'] + 'seed{}'.format(self.seed) + '.txt', 'w')
        self.model.load_state_dict(torch.load('{}CNN_{}.pth'.format(self.checkpoint_dir, self.epoch)))
        with torch.no_grad():
            self.model.train(False)
            test_matrix = [[0, 0], [0, 0]]
            for inputs, labels in self.test_dataloader:
                inputs, labels = inputs.cuda(), labels.cuda()
                preds = self.model(inputs)
                write_raw_score(f, preds, labels)
                test_matrix = matrix_sum(test_matrix, get_confusion_matrix(preds, labels))
        print('Test confusion matrix:', test_matrix, 'test_accuracy:', "%.4f" % get_accu(test_matrix))
        f.close()
        return get_accu(test_matrix)

    def get_checkpoint_dir(self):
        return self.checkpoint_dir

    def save_checkpoint(self, valid_matrix, epoch):
        if get_accu(valid_matrix) >= self.valid_optimal_accu:
            self.epoch = epoch
            self.valid_optimal_matrix = valid_matrix
            self.valid_optimal_accu = get_accu(valid_matrix)
            for root, Dir, Files in os.walk(self.checkpoint_dir):
                for File in Files:
                    if File.endswith('.pth'):
                        try:
                            os.remove(self.checkpoint_dir + File)
                        except:
                            pass
            torch.save(self.model.state_dict(), '{}CNN_{}.pth'.format(self.checkpoint_dir, self.epoch))

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


class GAN:
    def __init__(self, config, seed):
        self.seed = seed
        self.config = read_json(config)
        self.netG = _netG(self.config["G_fil_num"]).cuda()
        self.netD = _netD(self.config["D_fil_num"]).cuda()
        self.cnn  = CNN('./cnn_config_1.5T_optimal.json', 0)
        self.cnn.model.train(False)
        self.cnn.epoch=117
        self.cnn.model.load_state_dict(torch.load('{}CNN_{}.pth'.format(self.cnn.checkpoint_dir, self.cnn.epoch)))

        self.checkpoint_dir = self.config["checkpoint_dir"] + 'gan/'
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.train_dataloader = DataLoader(GAN_Data(self.config['Data_dir'], seed=seed, stage='train'),
                                           batch_size=self.config['batch_size'], shuffle=True)
        self.valid_dataloader = DataLoader(GAN_Data(self.config['Data_dir'], seed=seed, stage='valid'),
                                           batch_size=1, shuffle=True)
        self.test_dataloader = DataLoader(GAN_Data(self.config['Data_dir'], seed=seed, stage='test'),
                                           batch_size=1, shuffle=True)

    def train(self):
        self.G_lr, self.D_lr = self.config["G_lr"], self.config["D_lr"]
        self.optimizer_G = optim.SGD([ {'params': self.netG.conv1.parameters()},
                          {'params': self.netG.conv2.parameters()},
                          {'params': self.netG.conv3.parameters(), 'lr': 0.1*self.G_lr},
                          {'params': self.netG.BN1.parameters(), 'lr':self.G_lr},
                          {'params': self.netG.BN2.parameters(), 'lr':self.G_lr},
                        ], lr=self.G_lr, momentum=0.9)
        self.optimizer_D = optim.Adam(self.netD.parameters(), lr=self.D_lr, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss().cuda()
        self.valid_optimal_metric, self.optimal_epoch = 0, -1
        for self.epoch in range(self.config['epochs']):
            self.train_model_epoch()
            valid_metric = self.valid_model_epoch()
            print('This epoch validation metric (SSIM):', valid_metric)
            self.save_checkpoint(valid_metric)
        print('(GAN) Best validation metric at the {}th epoch:'.format(self.optimal_epoch), self.valid_optimal_metric)
        return self.valid_optimal_metric

# numpy_label = label.numpy()
# index = torch.LongTensor(np.argwhere(numpy_label!=2).squeeze())
# print(label, index)
# selected_scan = torch.index_select(scan1, 0, index)
# selected_label = torch.index_select(label, 0, index)
# print(selected_scan.shape, selected_label)

    def train_model_epoch(self):
        self.netG.train(True)
        self.netD.train(True)
        #Error may raise when input is actually smaller than batch size
        #Currently fixed by using real time size. Or can fix by discard small input
        #label = torch.FloatTensor(self.config["batch_size"])
        for idx, (inputs_lo, inputs_hi, label) in enumerate(self.train_dataloader):

            #for GAN later use, no use for now
            # modified = False
            # if modified:
            #     for i in range (len(labels)):
            #         inputs_lo_cnn = []
            #         inputs_hi_cnn = []
            #         labels_cnn = []
            #         inputs_lo_d = []
            #         inputs_hi_d = []
            #         labels_d = []
            #         if labels[i] != 2:
            #             #set consisting AD & NI, gen also get retruns from classifier
            #             inputs_lo_cnn.append(inputs_lo[i])
            #             inputs_hi_cnn.append(inputs_lo[i])
            #             labels_cnn.append(labels[i])
            #         else:
            #             #set consisting MCI, gen get returns from des
            #             inputs_lo_d.append(inputs_lo[i])
            #             inputs_hi_d.append(inputs_lo[i])
            #             labels_d.append(labels[i])


            inputs_lo, inputs_hi = inputs_lo.cuda(), inputs_hi.cuda()
            # get gradient for D network with fake data
            self.netD.zero_grad()
            Mask = self.netG(inputs_lo)
            Output = inputs_lo + Mask
            Foutput = self.netD(Output.detach())
            label = torch.FloatTensor(Foutput.shape[0])
            Flabel = label.fill_(0).cuda()
            loss_D_F = self.criterion(Foutput, Flabel)
            loss_D_F.backward()
            # get gradient for D network with real data
            Routput = self.netD(inputs_hi)
            Rlabel = label.fill_(1).cuda()
            loss_D_R = self.criterion(Routput, Rlabel)
            loss_D_R.backward()
            self.optimizer_D.step()
            # get gradient for G network
            self.netG.zero_grad()
            Goutput = self.netD(Output)
            #######################################
            Glabel = label.fill_(1).cuda()
            loss_G_GAN = self.criterion(Goutput, Glabel)
            # get gradient for G network with L2 norm between real and fake
            loss_G_dif = torch.mean(torch.abs(Mask))
            loss_G = loss_G_GAN + self.config["L1_norm_factor"] * loss_G_dif
            # loss_G.backward(retain_graph=True)
            loss_G.backward(retain_graph=True)
            self.optimizer_G.step()

            if idx % 5 == 0:
                print('[%d/%d][%d/%d] D(x): %.4f D(G(z)): %.4f / %.4f Mask L1_norm: %.4f loss_G: %.4f loss_D: %.4f'
                      % (self.epoch, self.config['epochs'], idx, len(self.train_dataloader), Routput.data.cpu().mean(),
                         Foutput.data.cpu().mean(), Goutput.data.cpu().mean(), 1000*loss_G_dif.data.cpu().mean(), loss_G.data.cpu().sum().item(), (loss_D_R+loss_D_F).data.cpu().sum().item()))

    def valid_model_epoch(self):
        with torch.no_grad():
            self.netG.train(False)
            valid_metric = []
            for inputs_lo, inputs_hi, label in self.valid_dataloader:
                inputs_lo, inputs_hi = inputs_lo.cuda(), inputs_hi
                Mask = self.netG(inputs_lo)
                output = inputs_lo + Mask
                ssim1 = SSIM(inputs_hi.squeeze().numpy(), output.squeeze().cpu().numpy())
                ssim2 = SSIM(output.squeeze().cpu().numpy(), inputs_hi.squeeze().numpy())
                if ssim1 != ssim2:
                    print(ssim1, ssim2)
                    sys.exit()
                valid_metric.append(ssim2)
        return sum(valid_metric) / len(valid_metric)

    # def valid_model_epoch(self):
    #     with torch.no_grad():
    #         self.netG.train(False)
    #         valid_metric = []
    #         for inputs_lo, inputs_hi, label in self.valid_dataloader:
    #             inputs_lo, inputs_hi = inputs_lo.cuda(), inputs_hi.cuda()
    #             Mask = self.netG(inputs_lo)
    #             output = inputs_lo + Mask
    #
    #             #need classifier, 1.5T
    #             #cnn = configuration_1.5T_optimal.json
    #             pred = classifier(output)
    #             valid_metric.append(pred==label)
    #
    #     #return classification accuracy
    #     return sum(valid_metric) / len(valid_metric)

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
                            os.remove(self.checkpoint_dir + File)
                        except:
                            pass
            torch.save(self.netG.state_dict(), '{}G_{}.pth'.format(self.checkpoint_dir, self.optimal_epoch))

    def test(self, plot=True):
        # Need test, partially completed
        print('preparing testing results...')
        self.netG.load_state_dict(torch.load('{}G_{}.pth'.format(self.checkpoint_dir, self.optimal_epoch))) #use optimal_epoch instead of epoch
        with torch.no_grad():
            self.netG.train(False)
            test_matrix = [[0, 0], [0, 0]]
            # do some plots and post analysis
            ssim_oris = []
            ssim_gens = []
            # pred_gens = []
            # pred_oris = []
            # pred_tars = []
            # labels = []
            # preds=[]
            for idx, (inputs_lo, inputs_hi, label) in enumerate(self.test_dataloader):
                if label[0] == 2:
                    continue
                inputs_lo, inputs_hi = inputs_lo.cuda(), inputs_hi.cuda()
                Mask = self.netG(inputs_lo)
                output = inputs_lo + Mask

                # preds += [[self.cnn.model(output).cpu().tolist()[0], self.cnn.model(inputs_lo).cpu().tolist()[0], self.cnn.model(inputs_hi).cpu().tolist()[0]]]
                # pred_gens += [torch.argmax(self.cnn.model(output)).item()]
                # pred_oris += [torch.argmax(self.cnn.model(inputs_lo)).item()]
                # pred_tars += [torch.argmax(self.cnn.model(inputs_hi)).item()]
                # labels += [label]

                #valid_metric.append(pred==label)

                out_dir = '../Test_GAN/'
                inputs_lo = inputs_lo.cpu().squeeze().numpy()
                output = output.cpu().squeeze().numpy()
                inputs_hi = inputs_hi.cpu().squeeze().numpy()

                if plot:
                    GAN_test_plot(out_dir, idx, inputs_lo, output, inputs_hi)
                # Do we need ssim in figure?
                ssim_ori = SSIM(inputs_lo, inputs_hi)
                ssim_oris += [ssim_ori]
                ssim_gen = SSIM(output, inputs_hi)
                ssim_gens += [ssim_gen]
                out_string = 'ssim between 1.5T and 3T is:', ssim_ori, '\nssim between 1.5T+ and 3T is:', ssim_gen
                save_list(out_dir, str(idx)+'.txt', out_string)
            # print('1.5+ accu:', accuracy_score(labels, pred_gens))
            # print('1.5  accu:', accuracy_score(labels, pred_oris))
            # print('3.0  accu:', accuracy_score(labels, pred_tars))
            # print(len(labels))
            # print(np.asarray(preds))
        print('Done. Results saved in', out_dir)
        print('Average ssims:')
        print('\t1.5  & 3:', mean(ssim_oris))
        print('\t1.5+ & 3:', mean(ssim_gens))

    def generate(self):
        # generate & save 1.5T+ images using MRIGAN
        sources = ["/data/datasets/ADNI_NoBack/", "/data/datasets/NACC_NoBack/", "/data/datasets/FHS_NoBack/", "/data/datasets/AIBL_NoBack/"]
        targets = ["/data/datasets/ADNIP_NoBack/", "/data/datasets/NACCP_NoBack/", "/data/datasets/FHSP_NoBack/", "/data/datasets/AIBLP_NoBack/"]

        #data = Data('AIBL_NoBack', class1='AIBL_1.5T_NL', class2='AIBL_1.5T_AD', stage='all')
        data = []
        data += [Data(sources[0], class1='ADNI_1.5T_NL', class2='ADNI_1.5T_AD', stage='all', shuffle=False)]
        data += [Data(sources[1], class1='NACC_1.5T_NL', class2='NACC_1.5T_AD', stage='all', shuffle=False)]
        data += [Data(sources[2], class1='FHS_1.5T_NL', class2='FHS_1.5T_AD', stage='all', shuffle=False)]
        data += [Data(sources[3], class1='AIBL_1.5T_NL', class2='AIBL_1.5T_AD', stage='all', shuffle=False)]
        dataloaders = [DataLoader(d, batch_size=1, shuffle=False) for d in data]
        Data_lists = [d.Data_list for d in data]
        print('Generating 1.5T+ images for datasets: ADNI, NACC, FHS, AIBL')

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
                    np.save(target+Data_list[j], output.data.cpu().numpy().squeeze())

        print('Generation completed!')

        '''
        print('testing now!')
        data = Data(targets[1], class1='NACC_1.5T_NL', class2='NACC_1.5T_AD', stage='all', shuffle=False)
        data = Data(targets[2], class1='FHS_1.5T_NL', class2='FHS_1.5T_AD', stage='all', shuffle=False)
        data = Data(targets[3], class1='AIBL_1.5T_NL', class2='AIBL_1.5T_AD', stage='all', shuffle=False)
        data = DataLoader(data, batch_size=1, shuffle=False)

        with torch.no_grad():
            self.cnn.model.train(False)
            for i, (input, label) in enumerate(data):
                input = input.cuda()
                pred = self.cnn.model(input)
                print(pred)
                break
            sys.exit()
        '''



if __name__ == "__main__":
    gan = GAN('./gan_config_optimal.json', 0)
    #gan.train()
    gan.optimal_epoch=281
    gan.test()
