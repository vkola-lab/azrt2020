import csv
import random
import os

def data_split(repe_time):
    with open('./ADNI.csv', 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    labels, train_valid_test = your_list[0:1], your_list[1:]
    for i in range(repe_time):
        random.shuffle(train_valid_test)
        folder = 'exp{}/'.format(i)
        if not os.path.exists(folder):
            os.mkdir(folder) 
        with open(folder + 'train.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerows(labels + train_valid_test[:250])
        with open(folder + 'valid.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerows(labels + train_valid_test[250:337])
        with open(folder + 'test.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerows(labels + train_valid_test[337:])

if __name__ == "__main__":
    data_split(5)