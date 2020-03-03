"""
ADNI_NC_3T.txt
ADNI_AD_3T.txt
ADNI_MCI_3T.txt

ADNI_NC_1.5T.txt
ADNI_AD_1.5T.txt
ADNI_MCI_1.5T.txt
"""

def parse_data_label(path, data_txt, label_txt):
    id_name_label = {}

    name_list = []
    with open(path + data_txt, 'r') as f:
        for line in f:
            name = line.strip('\n')
            name_list.append(name)

    label_list = []
    with open(path + label_txt, 'r') as f:
        for line in f:
            label = line.strip('\n')
            label_list.append(label)

    for i in range(len(name_list)):
        name, label = name_list[i], label_list[i]
        name = name.replace('nii', 'npy')
        id_name_label[name[5:15]] = (name, label)

    return id_name_label


# ADNI 1.5T part
# id_name_label = parse_data_label('/data/datasets/ADNI/', 'ADNI_New_3WAY_Data.txt', 'ADNI_New_3WAY_Label.txt')
# new_id_name_label = parse_data_label('/data/MRI_GAN/1.5T/', 'low_list.txt', 'Labels.txt')
# for key in new_id_name_label:
#     if key not in id_name_label:
#         print('key not exist', key)
#         id_name_label[key] = new_id_name_label[key]
#         continue
#     if id_name_label[key] != new_id_name_label[key]:
#         print('diff content', key, id_name_label[key], new_id_name_label[key])
#         id_name_label[key] = new_id_name_label[key]
# print(len(id_name_label), len(new_id_name_label))
# for key in new_id_name_label:
#     if key not in id_name_label:
#         print('key not exist', key)
#         id_name_label[key] = new_id_name_label[key]
#         continue
#     if id_name_label[key] != new_id_name_label[key]:
#         print('diff content', key, id_name_label[key], new_id_name_label[key])
#         id_name_label[key] = new_id_name_label[key]
# file_NL = open('./ADNI_1.5T_NL.txt', 'w')
# file_AD = open('./ADNI_1.5T_AD.txt', 'w')
# file_MCI = open('./ADNI_1.5T_MCI.txt', 'w')
# for key in id_name_label:
#     name, label = id_name_label[key]
#     if label == 'AD':
#         file_AD.write(name+'\n')
#     if label == 'NL':
#         file_NL.write(name+'\n')
#     if label == 'MCI':
#         file_MCI.write(name+'\n')
# file_AD.close()
# file_MCI.close()
# file_NL.close()


# ADNI 3T
# id_name_label = parse_data_label('/data/MRI_GAN/1.5T/', 'high_list.txt', 'Labels.txt')
# file_NL = open('./ADNI_3T_NL.txt', 'w')
# file_AD = open('./ADNI_3T_AD.txt', 'w')
# file_MCI = open('./ADNI_3T_MCI.txt', 'w')
# for key in id_name_label:
#     name, label = id_name_label[key]
#     if label == 'AD':
#         file_AD.write(name+'\n')
#     if label == 'NL':
#         file_NL.write(name+'\n')
#     if label == 'MCI':
#         file_MCI.write(name+'\n')
# file_AD.close()
# file_MCI.close()
# file_NL.close()


# def parse_data_label(path, data_txt, label_txt):
#     id_name_label = {}
#     name_list = []
#     with open(path + data_txt, 'r') as f:
#         for line in f:
#             name = line.strip('\n')
#             name_list.append(name)
#     label_list = []
#     with open(path + label_txt, 'r') as f:
#         for line in f:
#             label = line.strip('\n')
#             label_list.append(label)
#     for i in range(len(name_list)):
#         name, label = name_list[i], label_list[i]
#         name = name.replace('nii', 'npy')
#         id_name_label[name] = (name, label)
#     return id_name_label
#
#
# # NACC FHS, AIBL
# id_name_label = parse_data_label('/data/datasets/FHS/', 'FHS_Data.txt', 'FHS_Label.txt')
# file_NL = open('./FHS_1.5T_NL.txt', 'w')
# file_AD = open('./FHS_1.5T_AD.txt', 'w')
# file_MCI = open('./FHS_1.5T_MCI.txt', 'w')
# for key in id_name_label:
#     name, label = id_name_label[key]
#     if label == 'AD':
#         file_AD.write(name+'\n')
#     if label == 'NL':
#         file_NL.write(name+'\n')
#     if label == 'MCI':
#         file_MCI.write(name+'\n')
# file_AD.close()
# file_MCI.close()
# file_NL.close()

content1 = {}
with open('./ADNI_1.5T_MCI.txt', 'r') as f:
    for line in f:
        content1[line[5:15]] = line.strip('\n')

content2 = []
with open('./ADNI_3T_MCI.txt', 'r') as f:
    for line in f:
        content2.append(line[5:15])

content = []

for id in content2:
    content.append(content1[id])

with open('./ADNI_1.5T_GAN_MCI.txt', 'w') as f:
    for c in content:
        f.write(c + '\n')













