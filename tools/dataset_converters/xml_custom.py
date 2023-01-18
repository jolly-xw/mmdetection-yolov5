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
def get(root, name):
    return root.findall(name)


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_xml_dir', type=str,
                        help='Directory of images and xml.')
    parser.add_argument('--image_type', type=str, default='.png',
                        choices=['.jpg', '.png'], help='Type of image file.')
    parser.add_argument('--output_dir', type=str,
                        help='Directory of output.')
    a=parser.parse_args()
    image_geshi = a.image_type  # 设置图片的后缀名为png
    origin_ann_dir = a.image_xml_dir  # 设置存放所以xml和图片路径为tmp
    path2 =a.output_dir
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

    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1
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
        f1.write(os.path.basename(xml)[:-4] + "\n")
    f2 = open(path2+ r"/val.txt", "w")
    for xml in xml_list_val:
        img = xml[:-4] + image_geshi
        f2.write(os.path.basename(xml)[:-4] + "\n")
    f3 = open(path2 + r"/test.txt", "w")
    for xml in xml_list_test:
        img = xml[:-4] + image_geshi
        f3.write(os.path.basename(xml)[:-4] + "\n")
    f1.close()
    f2.close()
    f3.close()
    print("-------------------------------")
    print("train number:", len(xml_list_train))
    print("val number:", len(xml_list_val))
    print("test number:", len(xml_list_test))

    with open(path2 + r"\train.txt", "r", encoding="utf-8") as f:
        paths = [i.strip() for i in f.readlines()]

    path3 = path2 +'/train'
    if os.path.exists(path3):
        shutil.rmtree(path3)
        os.mkdir(path3)
    else:
        os.mkdir(path3)
    dst_dir=path2 + "/train"
    for i in paths:
        img_path =xml_dir+i+image_geshi
        xml_path=xml_dir+i+".xml"
        shutil.copy(img_path,dst_dir+"/"+i+image_geshi)
        shutil.copy(xml_path,dst_dir+"/"+i+".xml")

    with open(path2 + r"/class.txt", "r", encoding="utf-8") as f:
        a = [i.strip() for i in f.readlines()]
    classes = a
    xml_dir1 = path2 +"/train/"
    xml_list = glob.glob(xml_dir1 + "/*.xml")
    xml_list = np.sort(xml_list)
    pre_define_categories = {}
    pkl_dict = []
    categories = pre_define_categories.copy()
    bnd_id = START_BOUNDING_BOX_ID
    all_categories = {}
    number = 0
    for index, line in enumerate(xml_list):
        box_data = []
        box_data1 = []
        labels_data = []
        xml_f = line
        tree = ET.parse(xml_f)
        root = tree.getroot()
        number = number + 1
        filename = os.path.basename(xml_f)[:-4] + image_geshi
        image_id = 20190000001 + index
        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(float(get_and_check(bndbox, 'xmin', 1).text))
            ymin = int(float(get_and_check(bndbox, 'ymin', 1).text))
            xmax = int(float(get_and_check(bndbox, 'xmax', 1).text))
            ymax = int(float(get_and_check(bndbox, 'ymax', 1).text))
            assert (xmax > xmin), "xmax <= xmin, {}".format(line)
            assert (ymax > ymin), "ymax <= ymin, {}".format(line)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            box_data1 = [xmin, ymin, xmax, ymax]
            box_data.append(box_data1)
            name = int(obj.find('name').text)
            labels_data.append(name)
        image = {'filename': filename, 'width': width, 'height': height, 'ann': {'bboxes': np.array(box_data),
                                                                                      'labels': np.array(labels_data).T}}
        pkl_dict.append(image)
        print(number)
    with open(path2 + r"/train.pkl","wb") as f:
        pickle.dump(pkl_dict,f)
    print("success-train")

    # -------------------------------------
    with open(path2 + r"/val.txt", "r", encoding="utf-8") as f:
        paths = [i.strip() for i in f.readlines()]

    path3 =path2 + '/val'
    if os.path.exists(path3):
        shutil.rmtree(path3)
        os.mkdir(path3)
    else:
        os.mkdir(path3)
    dst_dir=path2 + "/val"
    for i in paths:
        img_path =xml_dir+i+image_geshi
        xml_path=xml_dir+i+".xml"
        shutil.copy(img_path,dst_dir+"/"+i+image_geshi)
        shutil.copy(xml_path,dst_dir+"/"+i+".xml")

    with open(path2 + r"/class.txt", "r", encoding="utf-8") as f:
        a = [i.strip() for i in f.readlines()]
    classes = a
    xml_dir2 = path2 + "/val/"
    xml_list = glob.glob(xml_dir2 + "/*.xml")
    xml_list = np.sort(xml_list)
    pre_define_categories = {}
    pkl_dict = []
    categories = pre_define_categories.copy()
    bnd_id = START_BOUNDING_BOX_ID
    all_categories = {}
    number = 0
    for index, line in enumerate(xml_list):
        box_data = []
        box_data1 = []
        labels_data = []
        xml_f = line
        tree = ET.parse(xml_f)
        root = tree.getroot()
        number = number + 1
        filename = os.path.basename(xml_f)[:-4] + image_geshi
        image_id = 20190000001 + index
        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(float(get_and_check(bndbox, 'xmin', 1).text))
            ymin = int(float(get_and_check(bndbox, 'ymin', 1).text))
            xmax = int(float(get_and_check(bndbox, 'xmax', 1).text))
            ymax = int(float(get_and_check(bndbox, 'ymax', 1).text))
            assert (xmax > xmin), "xmax <= xmin, {}".format(line)
            assert (ymax > ymin), "ymax <= ymin, {}".format(line)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            box_data1 = [xmin, ymin, xmax, ymax]
            box_data.append(box_data1)
            name = int(obj.find('name').text)
            labels_data.append(name)
        image = {'filename': filename, 'width': width, 'height': height, 'ann': {'bboxes': np.array(box_data),
                                                                                      'labels': np.array(labels_data).T}}
        pkl_dict.append(image)
        print(number)
    with open(path2 + r"\val.pkl","wb") as f:
        pickle.dump(pkl_dict,f)
    print("success-val")

    # -------------------------------------------
    with open(path2 + r"\test.txt", "r", encoding="utf-8") as f:
        paths = [i.strip() for i in f.readlines()]

    path3 =path2 +  r'/test'
    if os.path.exists(path3):
        shutil.rmtree(path3)
        os.mkdir(path3)
    else:
        os.mkdir(path3)
    dst_dir=path2 + "/test"
    for i in paths:
        img_path =xml_dir+i+image_geshi
        xml_path =xml_dir +i+".xml"
        shutil.copy(img_path,dst_dir+"/"+i+image_geshi)
        shutil.copy(xml_path,dst_dir+"/"+i+".xml")

    with open(path2 + r"/class.txt", "r", encoding="utf-8") as f:
        a = [i.strip() for i in f.readlines()]
    classes = a
    xml_dir3 =path2 +  "/test/"
    xml_list = glob.glob(xml_dir3 + "/*.xml")
    xml_list = np.sort(xml_list)
    pre_define_categories = {}
    pkl_dict = []
    categories = pre_define_categories.copy()
    bnd_id = START_BOUNDING_BOX_ID
    all_categories = {}
    number = 0
    for index, line in enumerate(xml_list):
        box_data = []
        box_data1 = []
        labels_data = []
        xml_f = line
        tree = ET.parse(xml_f)
        root = tree.getroot()
        number = number + 1
        filename = os.path.basename(xml_f)[:-4] + image_geshi
        image_id = 20190000001 + index
        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(float(get_and_check(bndbox, 'xmin', 1).text))
            ymin = int(float(get_and_check(bndbox, 'ymin', 1).text))
            xmax = int(float(get_and_check(bndbox, 'xmax', 1).text))
            ymax = int(float(get_and_check(bndbox, 'ymax', 1).text))
            assert (xmax > xmin), "xmax <= xmin, {}".format(line)
            assert (ymax > ymin), "ymax <= ymin, {}".format(line)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            box_data1 = [xmin, ymin, xmax, ymax]
            box_data.append(box_data1)
            name = int(obj.find('name').text)
            labels_data.append(name)
        image = {'filename': filename, 'width': width, 'height': height, 'ann': {'bboxes': np.array(box_data),
                                                                                      'labels': np.array(labels_data).T}}
        pkl_dict.append(image)
        print(number)
    with open(path2 + r"/test.pkl","wb") as f:
        pickle.dump(pkl_dict,f)
    print("success-test")