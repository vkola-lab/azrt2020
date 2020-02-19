from networks import CNN, GAN
from dataloader import Data


def gan_main():
    # gan training, record niqe, image, SSIM, brisque, ... over time on validation set
    # after training, generate 1.5T* for CNN /data/datasets/ADNIP_NoBack/
    # ...

    gan = GAN('./gan_config_optimal.json', 0)
    gan.train()
    #''' (108)
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



def cnn_main():
    # 5 random splits on 1.5T and 1.5T*
    # generate SS (ROC) and PR curves
    # print CNN performance table, accu, sens, spec, f1, ...
    return

gan_main()
