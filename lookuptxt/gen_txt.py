import csv

csv_file = '../lookupcsv/ADNI.csv'

def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    filenames = [a[0] for a in your_list[1:]]
    labels = [0 if a[1]=='NL' else 1 for a in your_list[1:]]
    return filenames, labels

filenames, labels = read_csv(csv_file)

f1 = open('./ADNI_1.5T_NL_.txt', 'w')
f2 = open('./ADNI_1.5T_AD_.txt', 'w')

for i in range(len(labels)):
    if labels[i] == 0:
        f1.write(filenames[i]+'.npy\n')
    else:
        f2.write(filenames[i]+'.npy\n')

f1.close()
f2.close()
