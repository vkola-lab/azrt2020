from models import Vanila_CNN 
from utils import * 

"""
1. augmentation, small rotation, 
2. hyperparameter auto tunning
3. 

"""


class CNN:
    def __init__(self, config, seed):
        self.config = read_json(config)
        self.model = Vanila_CNN(fil_num=10, drop_rate=0.5).cuda()
        self.train_dataloader = DataLoader(Data(self.config['Data_dir'], self.config['class1'], self.config['class2'], seed=seed, stage='train'),
                                           batch_size=self.config['batch_size'], shuffle=True, drop_last=True)
        self.valid_dataloader = DataLoader(Data(self.config['Data_dir'], self.config['class1'], self.config['class2'], seed=seed, stage='valid'),
                                           batch_size=self.config['batch_size'], shuffle=True)
        self.test_dataloader = DataLoader(Data(self.config['Data_dir'], self.config['class1'], self.config['class2'], seed=seed, stage='test'),
                                           batch_size=self.config['batch_size'], shuffle=True)

    def train(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['lr'], betas=(0.5, 0.999))
        self.criterion = nn.CrossEntropyLoss().cuda()
        Optimal_valid_matrix = [[0,0],[0,0]]
        valid_accu, Epoch = 0, -1
        for epoch in range(self.config['epoches']):
            self.train_model_epoch()
            valid_matrix = self.valid_model_epoch()
            #print('This epoch validation confusion matrix:', valid_matrix, 'validation_accuracy:', "%.4f" % get_accu(valid_matrix))
            if get_accu(valid_matrix) >= valid_accu:
                Epoch = epoch
                Optimal_valid_matrix = valid_matrix
                valid_accu = get_accu(valid_matrix)
                for root, Dir, Files in os.walk(model_dir):
                    for File in Files:
                        if File.endswith('.pth'):
                            try:
                                os.remove(model_dir + File)
                            except:
                                pass
                torch.save(model.state_dict(), '{}CNN_{}.pth'.format(model_dir, Epoch))
        print('(CNN) Best validation accuracy saved at the {}th epoch:'.format(Epoch), valid_accu, Optimal_valid_matrix)
        return Epoch


    def test(self):



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
            for l in range(len(self.valid_dataloader)):
                inputs, labels = self.valid_dataloader[l]
                inputs, labels = inputs.cuda(), labels.cuda()
                preds = self.model(inputs)
                valid_matrix = matrix_sum(valid_matrix, get_confusion_matrix(preds, labels))
        return valid_matrix


    