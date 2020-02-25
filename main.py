from dataloader import Data
from networks import CNN_Wrapper, CNN, GAN
from utils import read_json
import torch
import sys
sys.path.insert(1, './plot/')
from plot import roc_plot_perfrom_table

def gan_main():
    # gan training, record niqe, image, SSIM, brisque, ... over time on validation set
    # after training, generate 1.5T* for CNN /data/datasets/ADNIP_NoBack/
    # ...
    gan = GAN('./gan_config_optimal.json', 0)
    gan.train()
    gan.validate()
    print('########Gan Trainig Done.########')
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

#     cnn_config = read_json('./cnn_config.json')
#     cnn_main(5, 'cnn', cnn_config['cnn'])  # train, valid and test CNN model
#     print('########CNN Done.########')
#     cnn_main(5, 'cnnp', cnn_config['cnnp']) # train, valid and test CNNP model
#     print('########CNMP Done.########')
#     roc_plot_perfrom_table()
#     print('########Finished.########')
