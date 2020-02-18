'''
train_optimal.py
Last modified: 2/7/2020
Working: Yes
'''

from networks import CNN, GAN
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from dataloader import Data
from utils import p_val, report
import sys
import torch
import numpy as np

def eval_cnns(cnn1, cnn2):
    data  = []
    names = ['ADNI', 'NACC', 'FHS ', 'AIBL']

    sources = ["/data/datasets/ADNI_NoBack/", "/data/datasets/NACC_NoBack/", "/data/datasets/FHS_NoBack/", "/data/datasets/AIBL_NoBack/"]
    data += [Data(sources[0], class1='ADNI_1.5T_NL', class2='ADNI_1.5T_AD', stage='all', shuffle=False)]
    data += [Data(sources[1], class1='NACC_1.5T_NL', class2='NACC_1.5T_AD', stage='all', shuffle=False)]
    data += [Data(sources[2], class1='FHS_1.5T_NL', class2='FHS_1.5T_AD', stage='all', shuffle=False)]
    data += [Data(sources[3], class1='AIBL_1.5T_NL', class2='AIBL_1.5T_AD', stage='all', shuffle=False)]
    dataloaders = [DataLoader(d, batch_size=1, shuffle=False) for d in data]

    data = []
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

def PRF_cnns(cnn1, cnn2):
    data  = []
    names = ['ADNI', 'NACC', 'FHS ', 'AIBL']

    sources = ["/data/datasets/ADNI_NoBack/", "/data/datasets/NACC_NoBack/", "/data/datasets/FHS_NoBack/", "/data/datasets/AIBL_NoBack/"]
    data += [Data(sources[0], class1='ADNI_1.5T_NL', class2='ADNI_1.5T_AD', stage='all', shuffle=False)]
    data += [Data(sources[1], class1='NACC_1.5T_NL', class2='NACC_1.5T_AD', stage='all', shuffle=False)]
    data += [Data(sources[2], class1='FHS_1.5T_NL', class2='FHS_1.5T_AD', stage='all', shuffle=False)]
    data += [Data(sources[3], class1='AIBL_1.5T_NL', class2='AIBL_1.5T_AD', stage='all', shuffle=False)]
    dataloaders = [DataLoader(d, batch_size=1, shuffle=False) for d in data]

    data = []
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

        ps_1 = []
        rs_1 = []
        fs_1 = []
        ps_2 = []
        rs_2 = []
        fs_2 = []
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
            p, r, f, _ = report(labels, preds1)
            ps_1 += [p]
            rs_1 += [r]
            fs_1 += [f]
            # '''
            print('1.5  PRF on', name, ':')
            print('\tprecision:', p)
            print('\trecall:', r)
            print('\tf score:', f)
            # '''

            preds2 = []
            labels = []
            for i, (input, label) in enumerate(dataloader_p):
                input = input.cuda()
                pred = cnn2.model(input)
                preds2.append(torch.argmax(pred).item())
                labels.append(label.item())
            p, r, f, _ = report(labels, preds2)
            ps_2 += [p]
            rs_2 += [r]
            fs_2 += [f]
            # '''
            print('1.5+ PRF on', name, ':')
            print('\tprecision:', p)
            print('\trecall:', r)
            print('\tf score:', f)
            # '''
    return ps_1, rs_1, fs_1, ps_2, rs_2, fs_2


if __name__ == '__main__':
    cnn = CNN('./cnn_config_1.5T_optimal.json', 0)
    #valid_optimal_accu = cnn.train()
    cnn.epoch=146
    cnn.model.load_state_dict(torch.load('{}CNN_{}.pth'.format(cnn.checkpoint_dir, cnn.epoch)))
    cnn.test()
    # valid = 89.02
    # test  = 84.15

    # gan = GAN('./gan_config_optimal.json', 0)
    #gan.train()
    #'''
    # gan.optimal_epoch=105
    # gan.netG.load_state_dict(torch.load('{}G_{}.pth'.format(gan.checkpoint_dir, gan.optimal_epoch)))
    # print(gan.valid_model_epoch())
    #gan.test(False)
    # gan.eval_iqa(zoom=False, metric='brisque')
    # gan.eval_iqa(zoom=True, metric='brisque')
    # metrics = ['brisque', 'niqe', 'piqe']
    # for m in metrics:
    #     gan.eval_iqa_all(zoom=False, metric=m)
    #     gan.eval_iqa_all(zoom=True, metric=m)
    # gan.test(zoom=True, metric='piqe')
    #gan.generate()

    #cnnp = CNN('./cnn_config_1.5TP_optimal.json', 0)

    #this part is for 5-runs accuracy
    #--------------------------------
    if False:
        oris = []
        gens = []
        ps1s = []
        rs1s = []
        fs1s = []
        ps2s = []
        rs2s = []
        fs2s = []
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
            ori, gen = eval_cnns(cnn, cnnp)
            ps_1, rs_1, fs_1, ps_2, rs_2, fs_2 = PRF_cnns(cnn, cnnp)
            ps1s += [ps_1]
            rs1s += [rs_1]
            fs1s += [fs_1]
            ps2s += [ps_2]
            rs2s += [rs_2]
            fs2s += [fs_2]
            oris += [ori]
            gens += [gen]
        oris = np.asarray(oris)
        gens = np.asarray(gens)
        ps1s = np.asarray(ps1s)
        rs1s = np.asarray(rs1s)
        fs1s = np.asarray(fs1s)
        ps2s = np.asarray(ps2s)
        rs2s = np.asarray(rs2s)
        fs2s = np.asarray(fs2s)
        print(oris, gens)
        print('mean value & std & p_value:')
        for i in range(4):
            o = oris[:, i]
            g = gens[:, i]
            p1 = ps1s[:, i]
            r1 = rs1s[:, i]
            f1 = fs1s[:, i]
            p2 = ps2s[:, i]
            r2 = rs2s[:, i]
            f2 = fs2s[:, i]
            print('1.5  mean:', np.mean(o), 'std:', np.std(o))
            print('1.5+ mean:', np.mean(g), 'std:', np.std(g))
            print('p_value:', p_val(o, g))
            print('precision:', np.mean(p1), 'std:', np.std(p1))
            print('recall:', np.mean(r1), 'std:', np.std(r1))
            print('f score:', np.mean(f1), 'std:', np.std(f1))
            print('precision+:', np.mean(p2), 'std:', np.std(p2))
            print('recall+:', np.mean(r2), 'std:', np.std(r2))
            print('f score+:', np.mean(f2), 'std:', np.std(f2))

    #
    #--------------------------------

    cnnp = CNN('./cnn_config_1.5TP_optimal.json', 0)
    # valid_optimal_accu = cnnp.train()
    cnnp.epoch=83
    # '''
    cnnp.model.load_state_dict(torch.load('{}CNN_{}.pth'.format(cnnp.checkpoint_dir, cnnp.epoch)))
    cnnp.test()
    # valid = 0.8780
    # test  = 0.8659
    PRF_cnns(cnn, cnnp)

    #performance on all datasets
    # test(cnn, cnnp)
    #'''
