import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def augment_hsv(im, hgain=0.015, sgain=0.7, vgain=0.4):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * \
            [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(
            sat, lut_sat), cv2.LUT(val, lut_val)))
        r = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed
        return r


def yolox_hsv(img, hue_delta, saturation_delta, value_delta):
    hsv_gains = np.random.uniform(-1, 1, 3) * [
        hue_delta, saturation_delta, value_delta
    ]
    # random selection of h, s, v
    hsv_gains *= np.random.randint(0, 2, 3)
    # prevent overflow
    hsv_gains = hsv_gains.astype(np.int16)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

    img_hsv[..., 0] = (img_hsv[..., 0] + hsv_gains[0]) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_gains[1], 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_gains[2], 0, 255)
    r = cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)
    return r


if __name__ == '__main__':
    os.system('export DISPLAY=:0.0')
    path = 'demo/images/demo.jpg'
    img = cv2.imread(path)
    img_rgb_hsv = augment_hsv(img)
    # plt.imshow(img_rgb)
    # cv2.imwrite('demo/demo_ori.jpg',img)
    cv2.imwrite('demo/demo_hsv_yolov5.jpg', img_rgb_hsv)
    img_hsv_yolox = yolox_hsv(
        img, hue_delta=5, saturation_delta=30, value_delta=30)
    cv2.imwrite('demo/demo_hsv_yolox.jpg', img_hsv_yolox)
