from dataloader import Data
from networks import CNN_Wrapper, FCN_Wrapper, FCN_GAN, MLP_Wrapper
from utils import read_json
import numpy as np
from PIL import Image
from utils import read_json, iqa_tensor, p_val
import torch
import sys
sys.path.insert(1, './plot/')
from plot import roc_plot_perfrom_table
from train_curve_plot import train_plot
from heatmap import plot_heatmap
import matlab.engine
import os
import shutil
from torch.utils.data import Dataset, DataLoader
from dataloader import Data, GAN_Data, CNN_Data
import numpy as np

def gan_main():
    gan = FCN_GAN('./gan_config_optimal.json', 0)
    gan.train()
    gan.generate()
    gan.load_trained_FCN('./cnn_config.json', exp_idx=0)
    gan.fcn.test_and_generate_DPMs()
    plot_heatmap('/home/sq/gan2020/DPMs/fcn_gan_exp', 'fcngan_heatmap', exp_idx=0, figsize=(9, 4))
    return gan

def eval_iqa_all(metrics=['brisque', 'niqe', 'piqe']):
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

def fcn_main(repe_time, model_name, fcn_setting):
    for exp_idx in range(repe_time):
        if exp_idx == 0: continue
        fcn = FCN_Wrapper(fil_num        = fcn_setting['fil_num'],
                        drop_rate       = fcn_setting['drop_rate'],
                        batch_size      = fcn_setting['batch_size'],
                        balanced        = fcn_setting['balanced'],
                        Data_dir        = fcn_setting['Data_dir'],
                        patch_size      = fcn_setting['patch_size'],
                        lr              = fcn_setting['learning_rate'],
                        exp_idx         = exp_idx,
                        seed            = 1000,
                        model_name      = 'fcn',
                        metric          = 'accuracy')
        # fcn.train(epochs = fcn_setting['train_epochs'])
        # fcn.optimal_epoch = 1700
        # fcn.test_and_generate_DPMs()
        plot_heatmap('/home/sq/gan2020/DPMs/fcn_exp', 'fcn_heatmap', exp_idx=exp_idx, figsize=(9, 4))

def mlp_main(exp_time, repe_time, model_name, mode, mlp_setting):
    for exp_idx in range(exp_time):
        for repe_idx in range(repe_time):
            mlp = MLP_Wrapper(imbalan_ratio     = mlp_setting['imbalan_ratio'],
                                fil_num         = mlp_setting['fil_num'],
                                drop_rate       = mlp_setting['drop_rate'],
                                batch_size      = mlp_setting['batch_size'],
                                balanced        = mlp_setting['balanced'],
                                roi_threshold   = mlp_setting['roi_threshold'],
                                exp_idx         = exp_idx,
                                seed            = repe_idx*exp_idx,
                                mode            = mode,
                                model_name      = model_name,
                                metric          = 'accuracy')
            mlp.train(lr     = mlp_setting['learning_rate'],
                    epochs = mlp_setting['train_epochs'])
            mlp.test(repe_idx)

def post_evaluate():
    cnn_config = read_json('./cnn_config.json')
    gan = FCN_GAN('./gan_config_optimal.json', 0)
    for epoch in range(90, 1200, 60):
        print('-'*20)
        gan.load_trained_FCN('./cnn_config.json', exp_idx=0, epoch=epoch)
        gan.generate(epoch=epoch)  # 45 s
        gan.fcn.test_and_generate_DPMs() # 150 s
        mlp_main(1, 5, 'mlp_fcn_gan', 'gan_', cnn_config['mlp']) # 180 s
        roc_plot_perfrom_table()  
        eval_iqa_validation() # quick


if __name__ == "__main__":
    # post_evaluate()

    # cnn_config = read_json('./cnn_config.json')
    # gan_main()       # train FCN-GAN; generate 1.5T*; generate DPMs for mlp and plot MCC heatmap
    # mlp_main(1, 5, 'mlp_fcn_gan', 'gan_', cnn_config['mlp']) # train 5 mlp models with random seeds on generated DPMs from FCN-GAN
    roc_plot_perfrom_table()  # plot roc and pr curve; print mlp performance table
    # eval_iqa_all()  # evaluate image quality (niqe, piqe, brisque) on 1.5T and 1.5T* 
    # train_plot()    # plot image quality, accuracy change as function of time; scatter plots between variables

    
    # fcn_main(5, 'fcn', cnn_config['fcn'])
    # mlp_main(1, 5, 'mlp_fcn', '', cnn_config['mlp'])