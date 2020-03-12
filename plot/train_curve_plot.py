import matplotlib.pyplot as plt
import os
from glob import glob
import numpy as np

def parse_metric(filename):
    content = filename.split('#')
    epoch = content[0]
    niqe = content[2]
    piqe = content[4]
    bris = content[6].strip('.png')
    content = [epoch, niqe, piqe, bris]
    return list(map(float, content))

def train_plot():
    METRIC = {}
    Epoch = []

    png_files = glob('../output_mix/*.png')

    for png in png_files:
        content = parse_metric(png.split('/')[-1])
        METRIC[content[0]] = content[1:]

    txt_file = '../log_test_mix.txt'
    log = []
    with open(txt_file, 'r') as f:
        for line in f:
            if 'validation accuracy' in line:
                log.append(float(line.strip('\n').replace('validation accuracy ', '')))

    for i, epoch in enumerate(range(0, 1200, 10)):
        METRIC[epoch].append(log[i])

    METRIC = sorted(METRIC.items())
    x, y = zip(*METRIC)
    x, y = np.array(x), np.array(y)

    plt.subplot(2, 2, 1)
    plt.plot(x, y[:, 0], label='niqe')
    plt.ylabel('niqe')
    plt.subplot(2, 2, 2)
    plt.plot(x, y[:, 1], label='piqe')
    plt.ylabel('piqe')
    plt.subplot(2, 2, 3)
    plt.plot(x, y[:, 2], label='bris')
    plt.ylabel('bris')
    plt.subplot(2, 2, 4)
    plt.plot(x, y[:, 3], label='accu')
    plt.ylabel('accu')
    plt.savefig('./train_curve.png')

    plt.clf()
    plt.scatter(y[:, 1], y[:, 3])
    plt.xlabel('piqe')
    plt.ylabel('accuracy')
    plt.savefig('./accu_vs_piqe.png')

    plt.clf()
    plt.scatter(y[:, 0], y[:, 3])
    plt.xlabel('niqe')
    plt.ylabel('accuracy')
    plt.savefig('./accu_vs_niqe.png')

    plt.clf()
    plt.scatter(y[:, 2], y[:, 3])
    plt.xlabel('brisque')
    plt.ylabel('accuracy')
    plt.savefig('./accu_vs_brisque.png')

def plot_learning_curve(iqas, accs, iqa_ori):
    iqas = np.array(iqas)
    x = [10*i for i in range(len(accs))]

    plt.subplot(2, 2, 1)
    plt.plot(x, iqas[:, 0], label='bris')
    plt.plot(x, [iqa_ori[0]]*len(x), color='red')
    plt.ylabel('bris')
    plt.subplot(2, 2, 2)
    plt.plot(x, iqas[:, 1], label='niqe')
    plt.plot(x, [iqa_ori[1]]*len(x), color='red')
    plt.ylabel('niqe')
    plt.subplot(2, 2, 3)
    plt.plot(x, iqas[:, 2], label='piqe')
    plt.plot(x, [iqa_ori[2]]*len(x), color='red')
    plt.ylabel('piqe')
    plt.subplot(2, 2, 4)
    plt.plot(x, accs, label='accu')
    plt.plot(x, [0.859]*len(x), color='red')
    plt.ylabel('accu')
    plt.savefig('./train_curve.png')

    plt.clf()
    plt.scatter(iqas[:, 0], accs)
    plt.xlabel('brisque')
    plt.ylabel('accuracy')
    plt.savefig('./accu_vs_brisque.png')

    plt.clf()
    plt.scatter(iqas[:, 1], accs)
    plt.xlabel('niqe')
    plt.ylabel('accuracy')
    plt.savefig('./accu_vs_niqe.png')

    plt.clf()
    plt.scatter(iqas[:, 2], accs)
    plt.xlabel('piqe')
    plt.ylabel('accuracy')
    plt.savefig('./accu_vs_piqe.png')

if __name__ == "__main__":
    train_plot()
