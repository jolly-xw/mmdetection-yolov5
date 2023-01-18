import os
import random
import pickle
from typing import List

import numpy as np
from PIL import Image, ImageCms
import cv2
from sklearn.cluster import KMeans


class ImageMatcher:
    k = 6

    def __init__(self, image_dir, try_load=True, do_save=True):
        if not os.path.exists(image_dir):
            raise ValueError
        self.image_dir = image_dir
        self.image_info_list_pkl_path = os.path.join(image_dir, 'image_info_list.pkl')
        self.kmeans_model_pkl_path = os.path.join(image_dir, 'kmeans_model.pkl')

        if try_load:
            try:
                self.load_info()
                return
            except FileNotFoundError:
                print('Incomplete files. Start rebuilding features.')

        image_names = list(filter(lambda s: s.endswith('.png') or s.endswith('.jpg'),
                                  os.listdir(image_dir)))
        color_list = []
        all_colors = np.zeros([len(image_names) * 256, 3], dtype=np.float)
        for i, image_name in enumerate(image_names):
            image_path = os.path.join(self.image_dir, image_name)
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).resize((160, 160))
            img = np.array(self._convert_to_cielab(img))
            colors = self._extract_dominant_colors(img)
            color_list.append(colors)
            all_colors[i * 256: (i + 1) * 256] = colors.astype(np.float)
        # Construct kmeans model
        self.kmeans_model = KMeans(n_clusters=self.k).fit(
            [np.asarray(i, dtype=float) for i in all_colors])

        self.image_info_list = []
        for i, image_name in enumerate(image_names):
            image_path = os.path.join(self.image_dir, image_name)
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Construct sift feature
            sift = cv2.SIFT_create()
            _, descriptors = sift.detectAndCompute(gray, None)

            # Compute color bags
            colors = color_list[i]
            color_bags = np.zeros(self.k)
            center_indexes = self.kmeans_model.predict(colors)
            for j in center_indexes:
                color_bags[j] += 1

            color_bags = self._power_normalize(color_bags)
            self.image_info_list.append([image_name, color_bags, descriptors])

        if do_save:
            self.save_info()

    def get_image_names(self) -> List[str]:
        return [info[0] for info in self.image_info_list]

    def load_info(self):
        with open(self.image_info_list_pkl_path, 'rb') as f:
            self.image_info_list = pickle.load(f)
        with open(self.kmeans_model_pkl_path, 'rb') as f:
            self.kmeans_model = pickle.load(f)

    def save_info(self):
        with open(self.image_info_list_pkl_path, 'wb') as f:
            pickle.dump(self.image_info_list, f)
        with open(self.kmeans_model_pkl_path, 'wb') as f:
            pickle.dump(self.kmeans_model, f)

    def get_scores(self, img: np.ndarray, sift_match_ratio=0.6) -> List[float]:
        # Color bag
        img = Image.fromarray(img).resize((160, 160))
        img = np.array(self._convert_to_cielab(img))
        colors = self._extract_dominant_colors(img)
        matched_bags = np.zeros(self.k)
        center_indexes = self.kmeans_model.predict(colors)
        for j in center_indexes:
            matched_bags[j] += 1
        matched_bags = self._power_normalize(matched_bags)

        distances = [float(np.linalg.norm(matched_bags - info[1])) for info in self.image_info_list]

        sift = cv2.SIFT_create()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 转为灰度图
        keypoints, descriptors = sift.detectAndCompute(gray, None)  # 生成sift特征
        if len(keypoints) == 0 or descriptors is None:
            return ((1 - np.array(distances) / max(distances))).tolist()

        """knn匹配sift特征"""
        good_percentages = []  # 由坏点占比，决定哪些烟盒筛出
        bf = cv2.BFMatcher(cv2.NORM_L2, False)  # 交叉验证
        for info in self.image_info_list:
            try:
                matches = bf.knnMatch(descriptors, info[2], k=2)
            except Exception:
                good_percentages.append(0)
                print('D')
                continue

            """lowe方法：用最佳距离和次佳距离比较，计算与每张图片匹配成功的点占比（比率测试）"""
            good_points = sum(1 for first, second in matches
                              if first.distance / second.distance < sift_match_ratio)
            good_percentages.append(good_points / len(keypoints))

        return (
                (1 - np.array(distances) / max(distances)) * 0.4
                + np.array(good_percentages) * 0.6
        ).tolist()

    @staticmethod
    def _convert_to_cielab(img):
        """
        颜色空间转换，rgb转为lab
        :param img:输入的图片，PIL.Image格式
        :return:PIL.Image格式
        """

        srgb_profile = ImageCms.createProfile("sRGB")
        lab_profile = ImageCms.createProfile("LAB")
        rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile,
                                                                    lab_profile, "RGB",
                                                                    "LAB")
        return ImageCms.applyTransform(img, rgb2lab_transform)

    @staticmethod
    def _extract_dominant_colors(img):
        """
        提取每个图片的主色调
        :param img: np.array [W, H, C = 3] dtype=uint8
        :return: np.array [256, 3] dtype=uint8
        """

        N_BLOCKS = 256
        BLOCK_SIZE = 10

        assert len(img.shape) == 3
        (w, h, c) = img.shape
        assert c == 3

        """记录每张图片BLOCK_SIZE个主色调"""
        colors = np.zeros([N_BLOCKS, 3], dtype=np.uint8)
        k = 0
        for i in range(0, w, BLOCK_SIZE):
            for j in range(0, h, BLOCK_SIZE):
                block = img[i: i + BLOCK_SIZE, j: j + BLOCK_SIZE]
                dcolor = ImageMatcher._dominant_color(block)
                colors[k] = dcolor
                k += 1
        return colors

    @staticmethod
    def _dominant_color(block, thresh=4):
        """
        提取每个block的主色调

        :param block: np.array [W, H, 3]
        :param thresh: int 如果block中多数颜色出现的次数小于此值，
        则从block中随机选择一种颜色（也可以固定位置不随机选取）。
        使用4而不是5因为block相比论文中使用的小了一点
        """
        block = np.reshape(block, [-1, 3])
        hist = {}

        """对每个block建立颜色直方图，统计各颜色出现的次数"""
        for color in block:
            [c, i, e] = color
            key = (c, i, e)
            if key in hist:
                hist[key] += 1
            else:
                hist[key] = 1

        color, count = max(hist.items(), key=lambda e: e[1])
        if count < thresh:
            # 如果没有显著的颜色，那么随机选一个颜色（也可以不随机）
            return list(random.choice(block))
        return list(color)

    @staticmethod
    def _power_normalize(bocs):
        """
        幂律和L1向量归一化
        :param bocs:np.ndarray 待标准化的颜色词袋
        :return:np.ndarray 标准化后的颜色词袋
        """

        """逐元素平方根，然后L1归一化"""
        o = np.sqrt(bocs)
        o /= np.sum(o)
        return o


if __name__ == '__main__':
    imageMatcher = ImageMatcher('sift_images')
    imageMatcher.get_scores(
        cv2.imdecode(np.fromfile('sift_images/ESSE(金)1mg.png', dtype=np.uint8), -1))
