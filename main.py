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
from train_curve_plot import train_plot, parse_metric
from heatmap import plot_heatmap
import matlab.engine
import os
import shutil
from torch.utils.data import Dataset, DataLoader
from dataloader import Data, GAN_Data, CNN_Data
import numpy as np
from glob import glob
import imageio
import matplotlib.pyplot as plt

def gan_main():
    # epoch = 5555 for 2000, 3-1-1
    # 2000*0.8*417/(151*0.6)
    # epoch = 7413 for 2000, 4-1-0
    gan = FCN_GAN('./gan_config_optimal.json', 0)
    # gan.train()
    # gan.generate(epoch=390)

    # gan.load_trained_FCN('./cnn_config.json', exp_idx=0, epoch=3780)
    # gan.fcn.test_and_generate_DPMs()

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
        # fcn.test_and_generate_DPMs(epoch=1)
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

def get_best():
    METRIC = {}
    Epoch = []

    png_files = glob('./output/*.png')

    for png in png_files:
        content = parse_metric(png.split('/')[-1])
        METRIC[content[0]] = content[1:]

    txt_file = './fcn_gan_log.txt'
    log = []
    with open(txt_file, 'r') as f:
        for line in f:
            if 'validation accuracy' in line:
                log.append(float(line.strip('\n').replace('validation accuracy ', '')))

    for i, epoch in enumerate(range(0, 5555, 30)):
        METRIC[epoch].append(log[i])

    METRIC = sorted(METRIC.items())
    x, y = zip(*METRIC)
    # print(np.array(y))
    x, y = np.asarray(x), np.asarray(y)

    print(np.max(y[:, 3]))
    print(np.argmax(y[:, 3])*30)
    return

    metrics = ['niqe', 'piqe', 'brisque']
    i = 0
    out = []
    for m in metrics:
        if i == 0:
            i+=1
            continue
        print(m)
        out += list(np.argwhere(y[:, i] <= (np.amin(y[:, i]))))
        print([[o[0]*30, y[:, i][o[0]]] for o in out])
        i+=1
    print(y[:, 1][33])

def sample():
    # print('Evaluating IQA results on all datasets:')
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
    out_dir = './output_eval_/'
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    for id in range(len(names)):
        stop = 50
        name = names[id]
        Data_list = Data_lists[id]
        dataloader = dataloaders[id]
        dataloader_p = dataloaders_p[id]

        # if name == 'ADNI':
        #     gd = GAN_Data("/data/datasets/ADNI_NoBack/", seed=1000, stage='train_p')
        #     plt.set_cmap("gray")
        #     plt.subplots_adjust(wspace=0.3, hspace=0.3)
        #     fig, axs = plt.subplots(1, 4, figsize=(20,15))
        #     for i, ((input_o, _), (input_p, _)) in enumerate(zip(dataloader, dataloader_p)):
        #         # print(name, i)
        #         input_o = np.load(gd.Data_dir + gd.Data_list_lo[i], mmap_mode='r').astype(np.float32)
        #         input_t = np.load(gd.Data_dir + gd.Data_list_hi[i], mmap_mode='r').astype(np.float32)
        #         input_p = np.load("./ADNIP_NoBack/" + gd.Data_list_lo[i], mmap_mode='r').astype(np.float32)
        #         # print(name)
        #         # imageio.imwrite(out_dir+'/'+name+'P/'+Data_list[i].replace('npy','tif'), input_p[:, 100, :])
        #         # imageio.imwrite(out_dir+'/'+name+'O/'+Data_list[i].replace('npy','tif'), input[:, 100, :])
        #         ori = input_o[:, 100, :]
        #         gen = input_p[:, 100, :]
        #         tar = input_t[:, 100, :]
        #         # axs[0, 0].imshow(ori, vmin=-1, vmax=2.5)
        #         axs[0].imshow(ori, vmin=-1, vmax=2.5)
        #         axs[0].set_title('1.5T', fontsize=25)
        #         axs[0].axis('off')
        #         axs[1].imshow(gen, vmin=-1, vmax=2.5)
        #         axs[1].set_title('1.5T*', fontsize=25)
        #         axs[1].axis('off')
        #         axs[2].imshow(tar, vmin=-1, vmax=2.5)
        #         axs[2].set_title('3T', fontsize=25)
        #         axs[2].axis('off')
        #         axs[3].imshow(gen-ori, vmin=-1, vmax=2.5)
        #         axs[3].set_title('mask', fontsize=25)
        #         axs[3].axis('off')
        #         plt.savefig(out_dir+name+'_train_'+str(i)+'.png', dpi=150)
        #         plt.cla()
        #         if i == stop:
        #             break


        for i, ((input_o, _), (input_p, _)) in enumerate(zip(dataloader, dataloader_p)):
            orig = input_o.squeeze().numpy()
            plus = input_p.squeeze().numpy()

            plt.set_cmap("gray")
            plt.subplots_adjust(wspace=0.1, hspace=0.0)
            fig, axs = plt.subplots(3, 3, figsize=(20, 15))

            axs[0, 0].imshow(orig[:, :, 85].T, vmin=-1, vmax=2.5)
            axs[0, 0].set_title('1.5T', fontweight="bold", fontsize=25)
            axs[0, 0].axis('off')
            axs[1, 0].imshow(plus[:, :, 85].T, vmin=-1, vmax=2.5)
            axs[1, 0].set_title('1.5T*', fontweight="bold", fontsize=25)
            axs[1, 0].axis('off')
            axs[2, 0].imshow(orig[:, :, 85].T - plus[:, :, 85].T, vmin=-1, vmax=2.5)
            axs[2, 0].set_title('Mask', fontweight="bold", fontsize=25)
            axs[2, 0].axis('off')

            axs[0, 1].imshow(np.rot90(orig[100, :, :]), vmin=-1, vmax=2.5)
            axs[0, 1].set_title('1.5T', fontweight="bold", fontsize=25)
            axs[0, 1].axis('off')
            im=axs[1, 1].imshow(np.rot90(plus[100, :, :]), vmin=-1, vmax=2.5)
            axs[1, 1].set_title('1.5T*', fontweight="bold", fontsize=25)
            axs[1, 1].axis('off')
            axs[2, 1].imshow(np.rot90(orig[100, :, :]-plus[100, :, :]), vmin=-1, vmax=2.5)
            axs[2, 1].set_title('Mask', fontweight="bold", fontsize=25)
            axs[2, 1].axis('off')

            axs[0, 2].imshow(np.rot90(orig[:, 100, :]), vmin=-1, vmax=2.5)
            axs[0, 2].set_title('1.5T', fontweight="bold", fontsize=25)
            axs[0, 2].axis('off')
            axs[1, 2].imshow(np.rot90(plus[:, 100, :]), vmin=-1, vmax=2.5)
            axs[1, 2].set_title('1.5T*', fontweight="bold", fontsize=25)
            axs[1, 2].axis('off')
            axs[2, 2].imshow(np.rot90(orig[:, 100, :] - plus[:, 100, :]), vmin=-1, vmax=2.5)
            axs[2, 2].set_title('Mask', fontweight="bold", fontsize=25)
            axs[2, 2].axis('off')

            cbar = fig.colorbar(im, ax=axs.ravel().tolist())
            for l in cbar.ax.yaxis.get_ticklabels():
                l.set_weight("bold")
                l.set_fontsize(25)

            plt.savefig(out_dir+name+str(i)+'.png', dpi=150)
            plt.close()


if __name__ == "__main__":

    # cnn_config = read_json('./cnn_config.json')
    # fcn_main(1, 'fcn', False, cnn_config['fcn'])
    # fcn_main(5, 'fcn', False, cnn_config['fcn'])
    # mlp_main(1, 5, 'fcn', '', cnn_config['mlp']) # train 5 mlp models with random seeds on generated DPMs from FCN-GAN

    # gan = gan_main()       # train FCN-GAN; generate 1.5T*; generate DPMs for mlp and plot MCC heatmap
    # mlp_main(1, 5, 'fcn_gan', 'gan_', cnn_config['mlp'])
    # gan.eval_iqa_orig()
    # gan.eval_iqa_gene(epoch=390)
    # gan.eval_iqa_orig(names=['valid'])
    # get_best()
    # train_plot(gan.iqa_hash) # plot image quality, accuracy change as function of time; scatter plots between variables

    roc_plot_perfrom_table()  # plot roc and pr curve; print mlp performance table

    # gan.pick_time()

    # mlp_main(1, 5, 'mlp_fcn_gan', 'gan_', cnn_config['mlp']) # train 5 mlp models with random seeds on generated DPMs from FCN-GAN
    # mlp_main(1, 5, 'mlp_fcn_aug', 'aug_', cnn_config['mlp'])

    # fcn_main(5, 'fcn_aug', True, cnn_config['fcn'])
    # mlp_main(1, 5, 'mlp_fcn', '', cnn_config['mlp'])
    # roc_plot_perfrom_table(mode=['mlp_fcn', 'mlp_fcn_aug'])  # plot roc and pr curve; print mlp performance table
    # sample()
    
