from networks import GAN
from visual import ROC_plot
import sys
import torch

def main(filename):
    AUTO = True

    """
    hyperparameters: lr, batch_size, droprate, filternumber
    """

    for seed in range(1):
        print('training model with seed {}'.format(seed))
        gan = GAN('./'+filename, seed)
        valid_optimal_accu = gan.train()
        # test_accu = cnn.test()

    # ROC_plot(cnn.get_checkpoint_dir())

    if AUTO:
        print('$' + str(1-valid_optimal_accu) + '$$')


if __name__ == '__main__':
    filename = sys.argv[1]
    with torch.cuda.device(3):
        main(filename)
