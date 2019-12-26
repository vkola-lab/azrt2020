from networks import CNN
from visual import ROC_plot
import sys

def main(filename):
    AUTO = True

    """
    hyperparameters: lr, batch_size, droprate, filternumber
    """

    for seed in range(1):
        print('training model with seed {}'.format(100))
        cnn = CNN('./'+filename, seed)
        valid_optimal_accu = cnn.train()
        # test_accu = cnn.test()

    # ROC_plot(cnn.get_checkpoint_dir())

    if AUTO:
        print('$' + str(valid_optimal_accu) + '$$')


if __name__ == '__main__':
    filename = sys.argv[1]
    main(filename)


