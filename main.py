from dataloader import Data
from networks import CNN_Wrapper, GAN
from utils import read_json
import numpy as np
from PIL import Image
import cv2
import torch
import sys
sys.path.insert(1, './plot/')
from plot import roc_plot_perfrom_table

def gan_main():
    # gan training, record niqe, image, SSIM, brisque, ... over time on validation set
    # after training, generate 1.5T* for CNN /data/datasets/ADNIP_NoBack/
    # ...
    gan = GAN('./gan_config_optimal.json', 0)
    #gan.train()
    gan.epoch=1040
    gan.netG.load_state_dict(torch.load('{}G_{}.pth'.format(gan.checkpoint_dir, gan.epoch)))
    gan.generate()
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
    #gan.generate() # create 1.5T+ numpy array
    #print('########Generation Done.########')
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
    gan = gan_main()
#     gan.eval_iqa_all(['brisque', 'niqe'])
#     print('########IQA Done.########')

    # cnn_config = read_json('./cnn_config.json')
    # cnn_main(5, 'cnnp', cnn_config['cnn']) # train, valid and test CNNP model


    # cnn_main(1, 'cnn', cnn_config['cnn'])  # train, valid and test CNN model
    # print('########CNN Done.########')


#     
#     print('########CNMP Done.########')
    # roc_plot_perfrom_table()
#     print('########Finished.########')