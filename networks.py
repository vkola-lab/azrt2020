import os
from models import Vanila_CNN, Vanila_CNN_Lite
from utils import * 
from dataloader import Data
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm

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

    def train(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['lr'], betas=(0.5, 0.999))
        self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, self.imbalanced_ratio])).cuda()
        self.optimal_valid_matrix = [[0,0],[0,0]]
        self.valid_optimal_accu, self.epoch = 0, -1
        for epoch in range(self.config['epochs']):
            self.train_model_epoch()
            valid_matrix = self.valid_model_epoch()
            print('This epoch validation confusion matrix:', valid_matrix, 'validation_accuracy:', "%.4f" % get_accu(valid_matrix))
            self.save_checkpoint(valid_matrix, epoch)
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

            
if __name__ == "__main__":
    cnn = CNN('./configuration.json', 1000)
    cnn.test()
