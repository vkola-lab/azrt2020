'''
Train the optimal networks:
    classifier
    Gan
'''

from networks import CNN, GAN
from visual import ROC_plot
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from dataloader import Data
import sys
import torch
import numpy as np

def test(cnn1, cnn2):
    data = []
    names = ['ADNI', 'NACC', 'FHS ', 'AIBL']

    sources = ["/data/datasets/ADNI_NoBack/", "/data/datasets/NACC_NoBack/", "/data/datasets/FHS_NoBack/", "/data/datasets/AIBL_NoBack/"]
    data += [Data(sources[0], class1='ADNI_1.5T_NL', class2='ADNI_1.5T_AD', stage='all', shuffle=False)]
    data += [Data(sources[1], class1='NACC_1.5T_NL', class2='NACC_1.5T_AD', stage='all', shuffle=False)]
    data += [Data(sources[2], class1='FHS_1.5T_NL', class2='FHS_1.5T_AD', stage='all', shuffle=False)]
    data += [Data(sources[3], class1='AIBL_1.5T_NL', class2='AIBL_1.5T_AD', stage='all', shuffle=False)]
    dataloaders = [DataLoader(d, batch_size=1, shuffle=False) for d in data]

    targets = ["/data/datasets/ADNIP_NoBack/", "/data/datasets/NACCP_NoBack/", "/data/datasets/FHSP_NoBack/", "/data/datasets/AIBLP_NoBack/"]
    data += [Data(targets[0], class1='ADNI_1.5T_NL', class2='ADNI_1.5T_AD', stage='all', shuffle=False)]
    data += [Data(targets[1], class1='NACC_1.5T_NL', class2='NACC_1.5T_AD', stage='all', shuffle=False)]
    data += [Data(targets[2], class1='FHS_1.5T_NL', class2='FHS_1.5T_AD', stage='all', shuffle=False)]
    data += [Data(targets[3], class1='AIBL_1.5T_NL', class2='AIBL_1.5T_AD', stage='all', shuffle=False)]
    dataloaders_p = [DataLoader(d, batch_size=1, shuffle=False) for d in data]

    print('testing now!')

    accs_ori = []
    accs_gen = []

    with torch.no_grad():
        cnn1.model.train(False)
        cnn2.model.train(False)

        for i in range(len(names)):
            name = names[i]
            dataloader = dataloaders[i]
            dataloader_p = dataloaders_p[i]

            preds1 = []
            labels = []
            for i, (input, label) in enumerate(dataloader):
                input = input.cuda()
                pred = cnn1.model(input)
                preds1.append(torch.argmax(pred).item())
                labels.append(label.item())
            acc_ori = accuracy_score(labels, preds1)
            accs_ori += [acc_ori]
            print('1.5  accu on', name, 'is:', acc_ori)

            preds2 = []
            labels = []
            for i, (input, label) in enumerate(dataloader_p):
                input = input.cuda()
                pred = cnn2.model(input)
                preds2.append(torch.argmax(pred).item())
                labels.append(label.item())
            acc_gen = accuracy_score(labels, preds2)
            accs_gen += [acc_gen]
            print('1.5+ accu on', name, 'is:', acc_gen)
    return accs_ori, accs_gen

if __name__ == '__main__':
    # cnn = CNN('./cnn_config_1.5T_optimal.json', 0)
    #valid_optimal_accu = cnn.train()
    # cnn.epoch=117
    # cnn.model.load_state_dict(torch.load('{}CNN_{}.pth'.format(cnn.checkpoint_dir, cnn.epoch)))
    # print(cnn.test())
    # valid = 89.02
    # test  = 85.37

    # gan = GAN('./gan_config_optimal.json', 0)
    #gan = GAN('./gan_config_optimal_0001.json', 0)
    #gan.train()
    #'''
    #gan.optimal_epoch=411
    # gan.optimal_epoch=105
    # gan.netG.load_state_dict(torch.load('{}G_{}.pth'.format(gan.checkpoint_dir, gan.optimal_epoch)))
    # print(gan.valid_model_epoch())
    #gan.test(False)
    #gan.test()
    #gan.generate()

    #cnnp = CNN('./cnn_config_1.5TP_optimal.json', 0)
    oris = []
    gens = []
    for seed in range(5):
        cnn = CNN('./cnn_config_1.5T_optimal.json', 0)
        cnn.train(verbose=1)
        print(cnn.epoch)
        cnn.model.load_state_dict(torch.load('{}CNN_{}.pth'.format(cnn.checkpoint_dir, cnn.epoch)))

        cnnp = CNN('./cnn_config_1.5TP_optimal.json', 0)
        cnnp.train(verbose=1)
        print(cnnp.epoch)
        cnnp.model.load_state_dict(torch.load('{}CNN_{}.pth'.format(cnnp.checkpoint_dir, cnnp.epoch)))
        print('iter', seed, 'testing accuracy:', cnn.test(), cnnp.test())
        ori, gen = test(cnn, cnnp)
        oris += [ori]
        gens += [gen]
    oris = np.asarray(oris)
    gens = np.asarray(gens)
    print(oris, gens)
    print('mean value & std:')
    for i in range(4):
        print('1.5  mean:', np.mean(oris[:, i]), 'std:', np.std(oris[:, i]))
        print('1.5+ mean:', np.mean(gens[:, i]), 'std:', np.std(gens[:, i]))




    # valid_optimal_accu = cnnp.train()
    #cnnp.epoch=158
    #cnnp.epoch=196
    #cnnp.epoch=177
    # '''
    #cnnp.epoch=61
    # cnnp.model.load_state_dict(torch.load('{}CNN_{}.pth'.format(cnnp.checkpoint_dir, cnnp.epoch)))
    # print(cnnp.test())
    # valid = 0.8780
    # test  = 0.8415

    #performance on all datasets
    # test(cnn, cnnp)
    #'''
