import csv
import random
import os

def read_txt(txt_file):
    content = []
    with open(txt_file, 'r') as f:
        for line in f:
            content.append(line.strip('\n'))
    return content

def data_split(repe_time):
    GAN_part = read_txt('../lookuptxt/ADNI_1.5T_GAN_NL.txt') + read_txt('../lookuptxt/ADNI_1.5T_GAN_AD.txt')
    print(len(GAN_part))
    with open('./ADNI.csv', 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    col_names, content = your_list[0:1], your_list[1:]
    GAN_content, rest_content = [], []
    for c in content:
        if c[0]+'.npy' in GAN_part:
            GAN_content.append(c)
        else:
            rest_content.append(c)
    print(len(GAN_content), len(rest_content))
    for i in range(repe_time):
        random.shuffle(rest_content)
        folder = 'exp{}/'.format(i)
        if not os.path.exists(folder):
            os.mkdir(folder) 
        with open(folder + 'train.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerows(col_names + GAN_content + rest_content[:170])
        with open(folder + 'valid.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerows(col_names + rest_content[170:257])
        with open(folder + 'test.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerows(col_names + rest_content[257:])


def special_split():
    GAN_part = read_txt('../lookuptxt/ADNI_1.5T_GAN_NL.txt') + read_txt('../lookuptxt/ADNI_1.5T_GAN_AD.txt')
    with open('./ADNI.csv', 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    col_names, content = your_list[0:1], your_list[1:]
    GAN_content, rest_content = [], []
    for c in content:
        if c[0] + '.npy' in GAN_part:
            GAN_content.append(c)
        else:
            rest_content.append(c)
    print(len(GAN_content), len(rest_content))
    random.shuffle(rest_content)
    folder = 'exp6/'
    if not os.path.exists(folder):
        os.mkdir(folder)
    with open(folder + 'train.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerows(col_names + rest_content[:250])
    with open(folder + 'valid.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerows(col_names + rest_content[250:])
    with open(folder + 'test.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerows(col_names + GAN_content)

if __name__ == "__main__":
    special_split()