# coding:utf-8

# pip install lxml

import glob
import json
import shutil
import numpy as np
import pickle
import os
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str,
                        help='Directory of images and xml.')
    parser.add_argument('--image_type', type=str, default='.png',
                        choices=['.jpg', '.png'], help='Type of image file.')
    parser.add_argument('--output_dir', type=str,
                        help='Directory of output.')
    parser.add_argument('--train_scale', type=int,default=0.7,
                        help='train_ratio.')
    parser.add_argument('--val_scale', type=int,default=0.2,
                        help='val_ratio.')
    parser.add_argument('--test_scale', type=int,default=0.1,
                        help='test_ratio.')
    a=parser.parse_args()
    image_geshi = a.image_type  # 设置图片的后缀名为png
    origin_ann_dir = a.image_dir  # 设置存放所以xml和图片路径为tmp
    path2 = a.output_dir
    classes = []
    for dirpaths, dirnames, filenames in os.walk(origin_ann_dir):  # os.walk游走遍历目录名
        for filename in filenames:
            if filename.endswith('.xml'):
                if os.path.isfile(os.path.join(origin_ann_dir, filename)):  # 获取原始xml文件绝对路径，isfile()检测是否为文件 isdir检测是否为目录
                    origin_ann_path = os.path.join(r'%s%s' % (origin_ann_dir, filename))  # 如果是，获取绝对路径（重复代码）
                    # new_ann_path = os.path.join(r'%s%s' %(new_ann_dir, filename))
                    tree = ET.parse(origin_ann_path)  # ET是一个xml文件解析库，ET.parse（）打开xml文件。parse--"解析"
                    root = tree.getroot()  # 获取根节点
                    for object in root.findall('object'):
                        xmlbox = object.find('bndbox')  # 找到根节点下所有“object”节点
                        name = str(object.find('name').text)  # 找到object节点下name子节点的值（字符串）
                        if name not in classes:
                            classes.append(name)
    with open(path2 + r"/class.txt", "w") as f:
        f.write('\n'.join(classes))
    f.close()

    START_BOUNDING_BOX_ID = 1
    train_ratio = a.train_scale
    val_ratio = a.val_scale
    test_ratio = a.test_scale
    xml_dir = origin_ann_dir

    xml_list = glob.glob(xml_dir + "/*.xml")
    xml_list = np.sort(xml_list)
    np.random.seed(100)
    np.random.shuffle(xml_list)

    train_num = int(len(xml_list) * train_ratio)
    val_num = int(len(xml_list) * val_ratio)
    xml_list_train = xml_list[:train_num]
    xml_list_val = xml_list[train_num:train_num + val_num]
    xml_list_test = xml_list[train_num + val_num:]
    f1 = open(path2 + r"/train.txt", "w")
    for xml in xml_list_train:
        img = xml[:-4] + image_geshi
        # f1.write(os.path.basename(xml)[:-4]+'.png' + "\n")
        f1.write(os.path.basename(xml)[:-4]  + "\n")
    f2 = open(path2+ r"/val.txt", "w")
    for xml in xml_list_val:
        img = xml[:-4] + image_geshi
        # f2.write(os.path.basename(xml)[:-4]+'.png'  + "\n")
        f2.write(os.path.basename(xml)[:-4] + "\n")
    f3 = open(path2 + r"/test.txt", "w")
    for xml in xml_list_test:
        img = xml[:-4] + image_geshi
        # f3.write(os.path.basename(xml)[:-4]+'.png'  + "\n")
        f3.write(os.path.basename(xml)[:-4]  + "\n")
    f1.close()
    f2.close()
    f3.close()
    print("-------------------------------")
    print("train number:", len(xml_list_train))
    print("val number:", len(xml_list_val))
    print("test number:", len(xml_list_test))