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
    #gan.train()
    gan.epoch=1040
    gan.netG.load_state_dict(torch.load('{}G_{}.pth'.format(gan.checkpoint_dir, gan.epoch)))
    # gan.validate(plot=False)
    # print('########Gan Trainig Done.########')
    #''' (108)
    # gan.optimal_epoch=0
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
    gan.generate() # create 1.5T+ numpy array
    print('########Generation Done.########')
    # gan.table()
    # gan_boxplot()
    return gan


def niqe_trend():
    for i in range(0, 2000, 10):
        filename = './output/{}.png'.format(i)
        img = Image.open(filename)
        img = np.asarray(img)[:, :, 0]
        # matlab.double()
        print(img.shape)


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

if __name__ == "__main__":

    # gan = gan_main()
    #
    # eval_iqa_all(['brisque', 'niqe'])
    # print('########IQA Done.########')

    # cnn_config = read_json('./cnn_config.json')
    # cnn_main(5, 'cnnp', cnn_config['cnn']) # train, valid and test CNNP model


    # cnn_main(1, 'cnn', cnn_config['cnn'])  # train, valid and test CNN model
    # print('########CNN Done.########')

#

    cnn_config = read_json('./cnn_config.json')
    cnn_main(5, 'cnn', cnn_config['cnn'])  # train, valid and test CNN model
    print('########CNN Done.########')
    cnn_main(5, 'cnnp', cnn_config['cnnp']) # train, valid and test CNNP model
    print('########CNMP Done.########')
    # roc_plot_perfrom_table()
#     print('########Finished.########')
