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

def train_plot(iqa_ori):
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

    for i, epoch in enumerate(range(0, 6000, 30)):
        METRIC[epoch].append(log[i])

    METRIC = sorted(METRIC.items())
    x, y = zip(*METRIC)
    # print(np.array(y))
    x, y = np.asarray(x), np.asarray(y)
    # print(type(y), y.shape)
    plt.subplot(2, 2, 1)
    plt.plot(x, y[:, 0], label='niqe')
    plt.plot(x, [np.mean(iqa_ori['niqe']['valid'])]*len(x), color='red')
    plt.ylabel('niqe')
    plt.subplot(2, 2, 2)
    plt.plot(x, y[:, 1], label='piqe')
    plt.plot(x, [np.mean(iqa_ori['piqe']['valid'])]*len(x), color='red')
    plt.ylabel('piqe')
    plt.subplot(2, 2, 3)
    plt.plot(x, y[:, 2], label='bris')
    plt.plot(x, [np.mean(iqa_ori['brisque']['valid'])]*len(x), color='red')
    plt.ylabel('bris')
    plt.subplot(2, 2, 4)
    plt.plot(x, y[:, 3], label='accu')
    plt.plot(x, [0.8436]*len(x), color='red')
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

if __name__ == "__main__":
    train_plot()
