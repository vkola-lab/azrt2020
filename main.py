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
import matplotlib.patches as patches
from visual import bold_axs_stick

def gan_main():
    # epoch = 5555 for 2000, 3-1-1
    # 2000*0.8*417/(151*0.6)
    # epoch = 7413 for 2000, 4-1-0
    gan = FCN_GAN('./gan_config_optimal.json', 0)
    # gan.train()
    gan.generate(epoch=390, special=True)

    # gan.generate_DPMs(epoch=3780)
    # gan.fcn.test_and_generate_DPMs(stages=['AIBL'])
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

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def sample():
    # print('Evaluating IQA results on all datasets:')
    data  = []
    names = ['ADNI', 'NACC', 'AIBL']
    sources = ["/data/datasets/ADNI_NoBack/", "/data/datasets/NACC_NoBack/", "/data/datasets/AIBL_15_NoBack/"]
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


        for i, ((input_o, _), (input_p, _)) in enumerate(zip(dataloader, dataloader_p)):

            orig = input_o.squeeze().numpy()
            plus = input_p.squeeze().numpy()

            plt.set_cmap("gray")
            plt.subplots_adjust(wspace=0.1, hspace=1.0)
            fig, axs = plt.subplots(3, 5, figsize=(45, 20))

            axs[0, 0].imshow(orig[:, ::-1, 85].T, vmin=-1, vmax=2.5)
            axs[0, 0].set_title('1.5T', fontweight="bold", fontsize=25)
            axs[0, 0].set_ylabel('Probability', fontweight="bold", fontsize=25)
            axs[0, 0].axis('off')
            axs[1, 0].imshow(plus[:, ::-1, 85].T, vmin=-1, vmax=2.5)
            axs[1, 0].set_title('1.5T*', fontweight="bold", fontsize=25)
            axs[1, 0].axis('off')
            axs[2, 0].imshow(orig[:, ::-1, 85].T - plus[:, ::-1, 85].T, vmin=-1, vmax=2.5)
            axs[2, 0].set_title('Transformation map', fontweight="bold", fontsize=25)
            axs[2, 0].axis('off')

            axs[0, 1].imshow(np.rot90(orig[100, :, :]), vmin=-1, vmax=2.5)
            axs[0, 1].set_title('1.5T', fontweight="bold", fontsize=25)
            axs[0, 1].axis('off')
            im = axs[1, 1].imshow(np.rot90(plus[100, :, :]), vmin=-1, vmax=2.5)
            axs[1, 1].set_title('1.5T*', fontweight="bold", fontsize=25)
            axs[1, 1].axis('off')
            axs[2, 1].imshow(np.rot90(orig[100, :, :] - plus[100, :, :]), vmin=-1, vmax=2.5)
            axs[2, 1].set_title('Transformation map', fontweight="bold", fontsize=25)
            axs[2, 1].axis('off')

            axs[0, 2].imshow(np.rot90(orig[:, 100, :]), vmin=-1, vmax=2.5)
            axs[0, 2].set_title('1.5T', fontweight="bold", fontsize=25)
            axs[0, 2].axis('off')
            axs[1, 2].imshow(np.rot90(plus[:, 100, :]), vmin=-1, vmax=2.5)
            axs[1, 2].set_title('1.5T*', fontweight="bold", fontsize=25)
            axs[1, 2].axis('off')
            axs[2, 2].imshow(np.rot90(orig[:, 100, :] - plus[:, 100, :]), vmin=-1, vmax=2.5)
            axs[2, 2].set_title('Transformation map', fontweight="bold", fontsize=25)
            axs[2, 2].axis('off')

            # Create a Rectangle patch
            rect = patches.Rectangle((80, 61), 40, 40, linewidth=1, edgecolor='r', facecolor='none')
            # Add the patch to the Axes
            axs[0, 2].add_patch(rect)
            rect = patches.Rectangle((80, 61), 40, 40, linewidth=1, edgecolor='r', facecolor='none')
            axs[1, 2].add_patch(rect)
            rect = patches.Rectangle((80, 61), 40, 40, linewidth=1, edgecolor='r', facecolor='none')
            axs[2, 2].add_patch(rect)

            # zoommed in
            axs[0, 3].imshow(np.rot90(orig[80:120, 100, 80:120]), vmin=-1, vmax=2.5)
            axs[0, 3].set_title('1.5T zoomed', fontweight="bold", fontsize=25)
            axs[0, 3].axis('off')
            axs[1, 3].imshow(np.rot90(plus[80:120, 100, 80:120]), vmin=-1, vmax=2.5)
            axs[1, 3].set_title('1.5T* zoomed', fontweight="bold", fontsize=25)
            axs[1, 3].axis('off')
            axs[2, 3].imshow(np.rot90(orig[80:120, 100, 80:120] - plus[80:120, 100, 80:120]), vmin=-1, vmax=2.5)
            axs[2, 3].set_title('Transformation map zoomed', fontweight="bold", fontsize=25)
            axs[2, 3].axis('off')

            axs[0, 4].hist(np.rot90(orig[80:120, 100, 80:120]).flatten(), bins=50, range=(0, 1.8))
            bold_axs_stick(axs[0, 4], 16)
            axs[0, 4].set_xticks([0, 0.5, 1, 1.5])
            axs[0, 4].set_yticks([0, 20, 40, 60])
            axs[0, 4].set_title('1.5T voxel histogram', fontweight="bold", fontsize=25)
            # axs[0, 4].set_xlabel('Voxel value', fontsize=25, fontweight='bold')
            axs[0, 4].set_ylabel('Count', fontsize=25, fontweight='bold')

            axs[1, 4].hist(np.rot90(plus[80:120, 100, 80:120]).flatten(), bins=50, range=(0, 1.8))
            bold_axs_stick(axs[1, 4], 16)
            axs[1, 4].set_xticks([0, 0.5, 1, 1.5])
            axs[1, 4].set_yticks([0, 20, 40, 60])
            axs[1, 4].set_title('1.5T* voxel histogram', fontweight="bold", fontsize=25)
            # axs[1, 4].set_xlabel('Voxel value', fontsize=25, fontweight='bold')
            axs[1, 4].set_ylabel('Count', fontsize=25, fontweight='bold')

            axs[2, 4].hist(np.rot90(orig[80:120, 100, 80:120]-plus[80:120, 100, 80:120]).flatten(), bins=50, range=(0, 1.8))
            bold_axs_stick(axs[2, 4], 16)
            axs[2, 4].set_xticks([0, 0.5, 1, 1.5])
            axs[2, 4].set_yticks([0, 50, 100, 150])
            axs[2, 4].set_title('Transformation map \nvoxel histogram', fontweight="bold", fontsize=25)
            axs[2, 4].set_xlabel('Voxel value', fontsize=25, fontweight='bold')
            axs[2, 4].set_ylabel('Count', fontsize=25, fontweight='bold')

            p, _ = np.histogram(np.rot90(orig[80:120, 100, 80:120]).flatten(), bins=20, range=(-1, 2.5), density=True)
            q, _ = np.histogram(np.rot90(plus[80:120, 100, 80:120]).flatten(), bins=20, range=(-1, 2.5), density=True)
            p = np.where(p != 0, p, 1)
            q = np.where(q != 0, q, 1)
            print('kl value:', kl_divergence(p, q))

            cbaxes = fig.add_axes([0.08, 0.1, 0.03, 0.8])
            cbar = fig.colorbar(im, ax=axs.ravel().tolist(), cax=cbaxes)
            for l in cbar.ax.yaxis.get_ticklabels():
                l.set_weight("bold")
                l.set_fontsize(25)

            plt.subplots_adjust(hspace=0.25)
            plt.savefig(out_dir + name + str(i) + '.tiff', dpi=80)
            plt.close()

            sys.exit()


if __name__ == "__main__":

    # cnn_config = read_json('./cnn_config.json')
    # fcn_main(5, 'fcn', False, cnn_config['fcn']) # train 25 fcn models with random seeds
    # mlp_main(5, 25, 'fcn_mlp', '', cnn_config['mlp']) # train 5*25 mlp models with random seeds on generated DPMs from FCN
    # print('stage1')

    # gan = gan_main()       # train FCN-GAN; generate 1.5T*; generate DPMs for mlp and plot MCC heatmap
    # print('stage2')
    # mlp_main(1, 25, 'fcn_gan_mlp', 'gan_', cnn_config['mlp']) # train 1*25 mlp models with random seeds on generated DPMs from FCN
    # print('stage3')
    # gan.eval_iqa_orig()   # evaluate the original image's quality
    # gan.eval_iqa_gene(epoch=390)  # generate & evaluate the generated image's quality, see function definition for more options
    # gan.eval_iqa_orig(names=['valid'])
    # get_best()
    # train_plot(gan.iqa_hash) # plot image quality, accuracy change as function of time; scatter plots between variables

    # roc_plot_perfrom_table()  # plot roc and pr curve; print mlp performance table

    # gan.pick_time()   # helper function for picking the optimal model

    # sample() # retrieve & plot sample images from the trained model for visualization & evaluation
