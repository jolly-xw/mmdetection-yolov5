import os
import numpy as np

data_dir = '/datasets/VOC2007train_val/VOCdevkit/VOC2007/Annotations/'
img_dir = '/datasets/VOC2007train_val/VOCdevkit/VOC2007/JPEGImages/'
train_path_txt = '/datasets/VOC2007train_val/VOCdevkit/VOC2007/ImageSets/Main/train.txt'
val_path_txt = '/datasets/VOC2007train_val/VOCdevkit/VOC2007/ImageSets/Main/val.txt'
# test_path_txt = 'data_ddc/test.txt'


f_train = open(train_path_txt, 'w')
f_val = open(val_path_txt, 'w')
# f_test = open(test_path_txt, 'w')

files = os.listdir(data_dir)  # os.listdir()用于返回一个由文件名和目录名组成的列表

ids = np.random.permutation(np.arange(len(files)))  # 来随机排列一个数组

ids_train = ids[:int(len(ids) * 0.8)]
ids_val = ids[int(len(ids) * 0.8):]
# ids_test = ids[int(len(ids) * 0.9):]

for i in ids_train:
    img_file = files[ids[i]][:-4]
    img_path = img_file
    print(img_path)
    f_train.write(img_path)
    f_train.write('\n')

for i in ids_val:
    img_file = files[ids[i]][:-4]
    img_path = img_file
    print(img_path)
    f_val.write(img_path)
    f_val.write('\n')

# for i in ids_test:
#     img_file = files[ids[i]][:-3] + 'jpg'
#     img_path = img_dir + '/' + img_file
#     print(img_path)
#     f_test.write(img_path)
#     f_test.write('\n')

f_train.close()
f_val.close()
# f_test.close()