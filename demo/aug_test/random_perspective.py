import cv2
import random
import math
import numpy as np
import xml.etree.ElementTree as ET


# box1(4,n), box2(4,n)
def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    # candidates
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)


def random_perspective(im,
                       targets=(),
                       degrees=10,
                       translate=.1,
                       scale=.1,
                       shear=10,
                       perspective=0.0,
                       border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    # x perspective (about y)
    P[2, 0] = random.uniform(-perspective, perspective)
    # y perspective (about x)
    P[2, 1] = random.uniform(-perspective, perspective)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) *
                       math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) *
                       math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 +
                             translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 +
                             translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(
                width, height), borderValue=(114, 114, 114))
        else:  # affine
            im = cv2.warpAffine(im, M[:2], dsize=(
                width, height), borderValue=(114, 114, 114))

    # # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(im[:, :, ::-1])  # base
    # ax[1].imshow(im2[:, :, ::-1])  # warped
    cv2.imwrite('demo/aug_test/perspective.jpg', im)
    # Transform label coordinates
    n = len(targets)
    if n:
        new = np.zeros((n, 4))
        # warp boxes
        xy = np.ones((n * 4, 3))
        # xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
            n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]
              ).reshape(n, 8)  # perspective rescale or affine

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate(
            (x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(
            box1=targets[:, 0:4].T * s, box2=new.T, area_thr=0.10)
        targets = targets[i]
        targets[:, 0:4] = new[i]

    return im, targets


if __name__ == '__main__':
    image = cv2.imread('demo/images/demo.jpg')  # BGR
    xml_path = 'demo/images/demo.xml'
    f = open(xml_path)
    tree = ET.parse(f)
    root = tree.getroot()
    # xywh = []
    xyxy = []
    for obj in root.iter('object'):
        xmlbox = obj.find('bndbox')
        xmin = float(xmlbox.find('xmin').text)
        xmax = float(xmlbox.find('xmax').text)
        ymin = float(xmlbox.find('ymin').text)
        ymax = float(xmlbox.find('ymax').text)
        xx, yy = (xmin + xmax) / 2, (ymin + ymax) / 2
        ww, hh = xmax - xmin, ymax - ymin
        # xywh.append([0,xx,yy,ww,hh])
        xyxy.append([xmin, ymin, xmax, ymax])
    # target_array = np.array(xywh,dtype=np.float32)
    target_array = np.array(xyxy, dtype=np.float32)
    image, label = random_perspective(image, target_array)  # 传进去的label是xyxy格式的
    for i in range(len(label)):
        b, g, r = random.random()*255, random.random()*255, random.random()*255
        image = cv2.rectangle(image, (int(label[i][0]), int(
            label[i][1])), (int(label[i][2]), int(label[i][3])), (b, g, r), 2)
    cv2.imwrite('demo/111.jpg', image)
