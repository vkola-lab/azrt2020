from dataloader import Data
from networks import CNN_Wrapper, FCN_Wrapper, FCN_GAN, MLP_Wrapper
from utils import read_json
import torch
import sys
sys.path.insert(1, './plot/')
from heatmap import plot_heatmap
import matlab.engine
import os
from torch.utils.data import Dataset, DataLoader
from dataloader import Data, GAN_Data, CNN_Data

def gan_main():
    gan = FCN_GAN('./gan_config.json', 0)
    # Training the GAN
    gan.train()
    # Generate 3T* images
    gan.generate()
    # Generate DPMs based on 3T* images
    gan.generate_DPMs()

    return gan


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
        fcn.test_and_generate_DPMs()
        # fcn.test_and_generate_DPMs(epoch=299, stages=['AIBL'])
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
    cnn_config = read_json('./cnn_config.json')
    gan = gan_main()       # train FCN-GAN; generate 3T*; generate DPMs for mlp and plot MCC heatmap
    mlp_main(1, 25, 'fcn_gan_mlp', 'gan_', cnn_config['mlp']) # train 1*25 mlp models with random seeds on generated DPMs from FCN
