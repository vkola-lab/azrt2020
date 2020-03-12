from dataloader import Data
from networks import CNN_Wrapper, GAN
from utils import read_json
import numpy as np
from PIL import Image
from utils import read_json, iqa_tensor, p_val
import torch
import sys
sys.path.insert(1, './plot/')
from plot import roc_plot_perfrom_table
from train_curve_plot import *
import matlab.engine
import os
import shutil
from torch.utils.data import Dataset, DataLoader
from dataloader import Data, GAN_Data, CNN_Data
import numpy as np

def gan_main():
    # gan training, record niqe, image, SSIM, brisque, ... over time on validation set
    # after training, generate 1.5T* for CNN /data/datasets/ADNIP_NoBack/
    # ...
    gan = GAN('./gan_config_optimal.json', 0)
    # gan.train()
    # gan.epoch=450
    # gan.netG.load_state_dict(torch.load('{}G_{}.pth'.format(gan.checkpoint_dir, gan.epoch)))
    # gan.generate()
    return gan

def eval_iqa_validation(metrics=['piqe']):
    eng = matlab.engine.start_matlab()
    data = Data("./ADNIP_NoBack/", class1='ADNI_1.5T_NL', class2='ADNI_1.5T_AD', stage='valid', shuffle=False)
    dataloader = DataLoader(data, batch_size=1, shuffle=False)
    Data_list = data.Data_list
    for m in metrics:
        iqa_gens = []
        for j, (input_p, _) in enumerate(dataloader):
            input_p = input_p.squeeze().numpy()
            iqa_gens += [iqa_tensor(input_p, eng, Data_list[j], m, './iqa/')]
        print('Average '+m+' on '+'ADNI validation '+' is:')
        iqa_gens = np.asarray(iqa_gens)
        print('1.5* : ' + str(np.mean(iqa_gens)) + ' ' + str(np.std(iqa_gens)))
    eng.quit()

def eval_iqa_all(metrics=['brisque']):
    # print('Evaluating IQA results on all datasets:')
    eng = matlab.engine.start_matlab()

    data  = []
    names = ['ADNI', 'NACC', 'AIBL']

    sources = ["/data/datasets/ADNI_NoBack/", "/data/datasets/NACC_NoBack/", "/data/datasets/AIBL_NoBack/"]
    data += [Data(sources[0], class1='ADNI_1.5T_NL', class2='ADNI_1.5T_AD', stage='test', shuffle=False)]
    data += [Data(sources[1], class1='NACC_1.5T_NL', class2='NACC_1.5T_AD', stage='all', shuffle=False)]
    data += [Data(sources[2], class1='AIBL_1.5T_NL', class2='AIBL_1.5T_AD', stage='all', shuffle=False)]
    dataloaders = [DataLoader(d, batch_size=1, shuffle=False) for d in data]

    data = []
    # targets = ["/home/sq/gan2020/ADNIP_NoBack/", "/home/sq/gan2020/NACCP_NoBack/", "/home/sq/gan2020/AIBLP_NoBack/"]
    targets = ["./ADNIP_NoBack/", "./NACCP_NoBack/", "./AIBLP_NoBack/"]
    data += [Data(targets[0], class1='ADNI_1.5T_NL', class2='ADNI_1.5T_AD', stage='test', shuffle=False)]
    data += [Data(targets[1], class1='NACC_1.5T_NL', class2='NACC_1.5T_AD', stage='all', shuffle=False)]
    data += [Data(targets[2], class1='AIBL_1.5T_NL', class2='AIBL_1.5T_AD', stage='all', shuffle=False)]
    dataloaders_p = [DataLoader(d, batch_size=1, shuffle=False) for d in data]

    Data_lists = [d.Data_list for d in data]

    out_dir = './iqa/'
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)

    out_str = ''
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
                iqa_oris += [iqa_tensor(input, eng, Data_list[i], m, out_dir)]
            for j, (input_p, _) in enumerate(dataloader_p):
                input_p = input_p.squeeze().numpy()
                iqa_gens += [iqa_tensor(input_p, eng, Data_list[j], m, out_dir)]
            o = 'Average '+m+' on '+name+' is:'
            print(o)
            out_str += o
            iqa_oris = np.asarray(iqa_oris)
            iqa_gens = np.asarray(iqa_gens)
            o = '\t1.5 : ' + str(np.mean(iqa_oris)) + ' ' + str(np.std(iqa_oris))
            print(o)
            out_str += o
            o = '\t1.5* : ' + str(np.mean(iqa_gens)) + ' ' + str(np.std(iqa_gens))
            print(o)
            out_str += o
            o = '\tp_value (1.5 & 1.5+): ' + str(p_val(iqa_oris, iqa_gens))
            print(o)
            out_str += o
    with open(out_dir+'iqa.txt', 'w') as f:
        f.write(out_str)
    eng.quit()


def cnn_main(repe_time, model_name, cnn_setting):
    for exp_idx in range(repe_time):
        cnn = CNN_Wrapper(fil_num        = cnn_setting['fil_num'],
                        drop_rate       = cnn_setting['drop_rate'],
                        batch_size      = cnn_setting['batch_size'],
                        balanced        = cnn_setting['balanced'],
                        Data_dir        = cnn_setting['Data_dir'],
                        exp_idx         = exp_idx,
                        seed            = 1000,
                        model_name      = model_name,
                        metric          = 'accuracy')
        cnn.train(lr     = cnn_setting['learning_rate'],
                  epochs = cnn_setting['train_epochs'])
        cnn.test()

def eval_CNN(json_file, exp_idx, checkpoint_dir, epoch):
    cnn_setting = read_json(json_file)['cnnp']
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
    if epoch:
        cnn.model.load_state_dict(torch.load('{}cnn_{}.pth'.format(checkpoint_dir, epoch)))
    cnn.test(gan=True)
    return cnn

def post_evaluate():
    gan = GAN('./gan_config_optimal.json', 0)
    iqas = []
    accs = []
    for epoch in range(0, 10, 10):
        print(epoch, 'evaluating')
        gan.netG.load_state_dict(torch.load('{}G_{}.pth'.format(gan.checkpoint_dir, epoch)))
        gan.initial_CNN('./cnn_config.json', exp_idx=0, epoch=epoch)
        # gan.generate(dataset_name=['ADNI'], epoch=epoch)
        iqa, acc = gan.eval()
        iqas += [iqa]
        accs += [acc]
    iqa_valid = gan.eval_valid()
    plot_learning_curve(iqas, accs, iqa_valid)

if __name__ == "__main__":

    # gan = gan_main()
    post_evaluate();

    # eval_CNN('./cnn_config.json', 0, gan.checkpoint_dir, gan.epoch)
    # #
    # eval_iqa_all(['brisque', 'niqe'])
    # print('########IQA Done.########')

    # cnn_config = read_json('./cnn_config.json')
    # cnn_main(5, 'cnnp', cnn_config['cnnp']) # train, valid and test CNNP model


    # cnn_main(1, 'cnn', cnn_config['cnn'])  # train, valid and test CNN model
    # print('########CNN Done.########')

    # cnn_config = read_json('./cnn_config.json')
    # cnn_main(5, 'cnn', cnn_config['cnn'])  # train, valid and test CNN model
    # print('########CNN Done.########')
    # cnn_main(5, 'cnnp', cnn_config['cnnp']) # train, valid and test CNNP model
    # print('########CNNP Done.########')
    # roc_plot_perfrom_table()
    # print('########Finished.########')
