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

def fcn_main(repe_time, model_name, augment, fcn_setting):
    for exp_idx in range(repe_time):
        fcn = FCN_Wrapper(fil_num        = fcn_setting['fil_num'],
                        drop_rate       = fcn_setting['drop_rate'],
                        batch_size      = fcn_setting['batch_size'],
                        balanced        = fcn_setting['balanced'],
                        Data_dir        = fcn_setting['Data_dir'],
                        patch_size      = fcn_setting['patch_size'],
                        lr              = fcn_setting['learning_rate'],
                        exp_idx         = exp_idx,
                        seed            = exp_idx,
                        model_name      = model_name,
                        metric          = 'accuracy',
                        augment         = augment)
        fcn.train(epochs = fcn_setting['train_epochs'])
        # fcn.optimal_epoch = 1700
        fcn.test_and_generate_DPMs()
        # plot_heatmap('/home/sq/gan2020/DPMs/fcn_exp', 'fcn_heatmap', exp_idx=exp_idx, figsize=(9, 4))

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


if __name__ == "__main__":
    # post_evaluate()

    cnn_config = read_json('./cnn_config.json')
    # gan_main()       # train FCN-GAN; generate 1.5T*; generate DPMs for mlp and plot MCC heatmap
    # mlp_main(1, 5, 'mlp_fcn_gan', 'gan_', cnn_config['mlp']) # train 5 mlp models with random seeds on generated DPMs from FCN-GAN
    # mlp_main(1, 5, 'mlp_fcn', '', cnn_config['mlp'])
    # mlp_main(1, 5, 'mlp_fcn_aug', 'aug_', cnn_config['mlp'])
    roc_plot_perfrom_table(mode=['mlp_fcn', 'mlp_fcn_aug'])  # plot roc and pr curve; print mlp performance table
    # eval_iqa_all()  # evaluate image quality (niqe, piqe, brisque) on 1.5T and 1.5T* 
    # train_plot()    # plot image quality, accuracy change as function of time; scatter plots between variables
    
    # fcn_main(5, 'fcn_aug', True, cnn_config['fcn'])
    # mlp_main(1, 5, 'mlp_fcn', '', cnn_config['mlp'])