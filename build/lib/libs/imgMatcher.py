"""
图像匹配算法
"""
import math
from PIL import Image, ImageCms
import numpy as np
import random
import cv2
import pandas as pd
import json
from sklearn.cluster import KMeans
import pickle
from time import time
from pathlib import Path
import shutil


class ImageMatch():
    """
    图片匹配工具类
    """

    def __init__(self, scene_path: Path):
        """
        :param scene_path:场景文件夹路径
        """

        """路径"""
        self.scene_path = scene_path  # 场景路径
        self.labels_found_path = (self.scene_path / "labels_found")  # labels_found文件夹
        self.labels_not_found_path = (self.scene_path / "labels_not_found")  # labels_found文件夹
        self.model_path = (self.scene_path / "model")  # model文件夹
        self.thumb_path = (self.scene_path / "thumb")  # thumb文件夹
        self.kmeans_model_path = (self.model_path / "kmeans_model.pkl")
        self.label_infos_path = (self.model_path / "label_infos.pkl")
        self.parameter_path = (self.model_path / "parameter.json")

    def convert_to_cielab(self, img):
        """
        颜色空间转换，rgb转为lab
        :param img:输入的图片，PIL.Image格式
        :return:PIL.Image格式
        """
        """图片如果不是rgb空间，转为rgb"""
        if img.mode != "RGB":
            img = img.convert("RGB")

        """rgb转为lab色彩空间"""
        srgb_profile = ImageCms.createProfile("sRGB")
        lab_profile = ImageCms.createProfile("LAB")
        rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")
        return ImageCms.applyTransform(img, rgb2lab_transform)

    def extract_dominant_colors(self, image):
        """
        提取每个图片的主色调
        :param image: np.array [W, H, C = 3] dtype=uint8
        :return: np.array [256, 3] dtype=uint8
        """

        N_BLOCKS = 256
        BLOCK_SIZE = 16

        assert len(image.shape) == 3
        (w, h, c) = image.shape
        assert c == 3

        """记录每张图片BLOCK_SIZE个主色调"""
        colors = np.zeros([N_BLOCKS, 3], dtype=np.uint8)
        k = 0
        for i in range(0, w, BLOCK_SIZE):
            for j in range(0, h, BLOCK_SIZE):
                block = image[i: i + BLOCK_SIZE, j: j + BLOCK_SIZE]
                dcolor = self.dominant_color(block)
                colors[k] = dcolor
                k += 1
        return colors

    def power_normalize(self, bocs):
        """
        幂律和L1向量归一化
        :param bocs:np.ndarray 待标准化的颜色词袋
        :return:np.ndarray 标准化后的颜色词袋
        """

        """逐元素平方根，然后L1归一化"""
        o = np.sqrt(bocs)
        o /= np.sum(o)
        return o

    def dominant_color(self, block, occurrence_threshold=5):
        """
        提取每个block的主色调

        :param block: np.array [W, H, 3]
        :param occurrence_threshold: int 如果block中多数颜色出现的次数小于此值，
        则从block中随机选择一种颜色（也可以固定位置不随机选取）。
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

        (color, count) = max(hist.items(), key=lambda e: e[1])
        if count < occurrence_threshold:
            # 如果没有显著的颜色，那么随机选一个颜色（也可以不随机）
            return list(random.choice(block))
        return list(color)

    def generate_model_with_thumb(self):
        """
        生成模型：颜色聚类模型，匹配信息，缩略图
        :return:
        """

        """labels_found文件夹中的当前类"""
        labels_found_img_path = list(self.labels_found_path.glob("*.jpg"))  # labels_found文件夹的类路径
        labels_found_num = len(labels_found_img_path)  # 当前labels_found文件夹中的类数

        """读取参数文件"""
        with self.parameter_path.open("r", encoding="utf-8") as f:
            parameter = json.load(f)

        """label_infos特征信息"""
        with self.label_infos_path.open("rb") as f:
            label_infos = pickle.load(f)

        current_labels = [i.stem for i in labels_found_img_path]  # 当前labels_found文件夹中的类名

        """（强制）更新颜色聚类模型"""
        if parameter["update_model"] is True or labels_found_num > 1.5 * parameter["labels_found_num"]:
            """生成所有图片的主色调池"""
            all_colors = np.zeros([labels_found_num * 256, 3], dtype=float)
            for i, img_path in enumerate(labels_found_img_path):
                img = Image.open(img_path).resize([256, 256])
                img = np.array(self.convert_to_cielab(img))
                colors = self.extract_dominant_colors(img)
                all_colors[i * 256: (i + 1) * 256] = colors.astype(float)
                print("已处理完 %d 个图片" % (i + 1))

            """主色调池聚类"""
            data = []
            for i in all_colors:
                a = np.asarray(i, dtype=float)
                data.append(a)
            kmean_model = KMeans(n_clusters=parameter["clusters"]).fit(data)
            with self.kmeans_model_path.open("wb") as f:
                pickle.dump(kmean_model, f)

        """（强制）更新label_infos模型"""
        if parameter["update_model"] is True or labels_found_num > parameter["labels_found_num"]:
            added_labels = set(current_labels) - set(parameter["recognized_labels"])  # 新的未提取特征的类
            # 强制更新模型
            if parameter["update_model"] is True:
                """重建thumb文件夹"""
                shutil.rmtree(str(self.thumb_path))
                self.thumb_path.mkdir(parents=True, exist_ok=True)

                """重新生成所有类的匹配信息"""
                label_infos = pd.DataFrame(columns=["name", "color", "feature", "search"])
                added_labels = current_labels

            """更新labels_info和thumb"""
            # 导入聚类模型
            with self.kmeans_model_path.open("rb") as f:
                kmeans_model = pickle.load(f)  # 聚类模型
            for label in added_labels:
                img_path = (self.scene_path / "labels_found" / (label + ".jpg"))  # 图片路径
                color_bags, descriptors = self.get_label_info(img_path, kmeans_model,
                                                              parameter["clusters"])  # 获得匹配信息
                new_row = {"name": label, "color": color_bags, "feature": descriptors, "search": 0}  # 新的匹配信息
                label_infos = label_infos.append(new_row, ignore_index=True)
                print("已对labels_found生成匹配信息：" + label)
        # 如果有人删除了labels_found文件夹中的图片
        elif labels_found_num < parameter["labels_found_num"]:
            deleted_labels = set(parameter["recognized_labels"]) - set(current_labels)  # 去除不符合labels_found文件夹中的类
            for label in deleted_labels:
                """更新labels_info"""
                index = label_infos[label_infos["name"] == label].index
                label_infos = label_infos.drop(index)  # 删除元素

                """删除缩略图"""
                thumb_path = (self.thumb_path / (label + ".png"))  # 缩略图路径
                thumb_path.unlink()

        """更新参数"""
        parameter["recognized_labels"] = current_labels
        parameter["labels_found_num"] = labels_found_num
        with self.parameter_path.open("w", encoding="utf-8") as f:
            json.dump(parameter, f, ensure_ascii=False)

        """更新labels_info"""
        with self.label_infos_path.open("wb") as f:
            pickle.dump(label_infos, f)

    def get_label_info(self, img_path: Path, kmeans_model, k):
        """
        获得单个label信息，生成缩略图

        :param img_path:
        :param kmeans_model:
        :param k:
        """

        img = Image.open(str(img_path))  # RGB格式读取

        """颜色词袋预测"""
        scaled_img = img.resize([256, 256])  # 图片缩放
        scaled_img = np.array(self.convert_to_cielab(scaled_img))
        colors = self.extract_dominant_colors(scaled_img)
        color_bags = np.zeros([k])
        center_indexes = kmeans_model.predict(colors)  # 词袋预测
        for j in center_indexes:
            color_bags[j] += 1
        color_bags = self.power_normalize(color_bags)  # L1归一化+幂律

        """SIFT特征提取"""
        sift = cv2.SIFT_create()
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)  # 对待匹配的图生成sift特征

        """生成缩略图"""
        label = img_path.stem  # 类名
        scaled_width = 200  # 缩放后的宽
        scaled_height = 200  # 缩放后的高
        thumb_path = (self.thumb_path / (label + ".png"))  # 缩略图路径
        img = Image.open(img_path)
        width, height = img.size
        # 缩放图片
        if width > height:
            img = img.resize((scaled_width, int(scaled_width * height / width)), Image.ANTIALIAS)
        else:
            img = img.resize((int(scaled_height * width / height), scaled_height), Image.ANTIALIAS)
        img.save(thumb_path)

        return (color_bags, descriptors)

    def check_scene_structrue(self):
        """
        检查场景文件结构的完整性，如果结构不完整就自动生成
        todo:生成缩略图
        :param scene_path:场景文件夹路径
        :return:
        """
        # 检查labels_found文件夹是否存在，如果不存在就创建
        self.labels_found_path.mkdir(parents=True, exist_ok=True)

        # 检查labels_not_found文件夹是否存在，如果不存在就创建
        self.labels_found_path.mkdir(parents=True, exist_ok=True)

        # 检查model文件夹是否存在，如果不存在就创建
        self.model_path.mkdir(parents=True, exist_ok=True)

        # 缩略图文件夹（用于界面展示）
        self.thumb_path.mkdir(parents=True, exist_ok=True)

        """读取参数文件"""
        # 如果没有找到颜色聚类模型的参数文件parameter.json，则自动生成
        if not self.parameter_path.is_file():
            parameter = {
                "clusters": 8,  # 聚类数
                "update_model": False,  # 是否更新模型
                "labels_found_num": 0,  # 上一次分好类的标签总数（用于颜色聚类）
                "recognized_labels": []  # 已经识别出的类
            }
            with self.parameter_path.open("w", encoding="utf-8") as f:
                json.dump(parameter, f, ensure_ascii=False)

        # 检查label_infos.pkl文件是否存在，如果不存在就创建
        if not self.label_infos_path.is_file():
            label_infos = pd.DataFrame(columns=["name", "color", "feature", "search"])
            with self.label_infos_path.open("wb") as f:
                pickle.dump(label_infos, f)

    def match(self, match_img):
        """
        按颜色+特征匹配
        :param match_img:待匹配的图片（numpy RGB格式）
        :return:排序后的label列表
        """

        # 如果模型不存在，则返回空值
        if not self.kmeans_model_path.exists() or not self.label_infos_path.exists():
            return []

        # 图像匹配信息
        with self.label_infos_path.open("rb") as f:
            label_infos = pickle.load(f)  # 待筛选的图片，包含[文字，颜色，sift特征，搜索次数]的信息，pandas格式
            label_infos["color_dist"] = 0  # 颜色距离
            label_infos["feature_dist"] = 0  # 匹配好点的个数
            label_infos["eval_dist"] = 0  # 总的评价值

        #################################  颜色匹配  ##########################################

        with self.kmeans_model_path.open("rb") as f:
            kmeans_model = pickle.load(f)  # 颜色空间kmeans聚类模型，用于预测待匹配图片的颜色词袋

        with self.parameter_path.open("r", encoding="utf-8") as f:
            parameter = json.load(f)  # 参数文件

        """生成待匹配图片的词袋"""
        k = parameter["clusters"]  # 聚类数
        img = Image.fromarray(match_img.astype('uint8')).convert('RGB').resize([256, 256])  # 缩放
        img = np.array(self.convert_to_cielab(img))
        colors = self.extract_dominant_colors(img)
        match_bags = np.zeros([k])
        center_indexes = kmeans_model.predict(colors)  # 词袋预测
        for j in center_indexes:
            match_bags[j] += 1
        match_bags = self.power_normalize(match_bags)  # L1归一化+幂律

        """计算颜色空间欧式距离"""
        for index, row in label_infos.iterrows():
            matched_bags = row["color"]  # 图像匹配库的颜色词袋
            label_infos.loc[index, ["color_dist"]] = np.linalg.norm(match_bags - matched_bags)  # 欧式距离

        """最大最小归一化"""
        label_infos["color_dist"] = (label_infos["color_dist"] - label_infos["color_dist"].min()) / (
                label_infos["color_dist"].max() - label_infos["color_dist"].min())

        #################################  特征匹配  ##########################################

        dd = time()

        """根据面积自动调整比率测试参数"""
        height, width, _ = match_img.shape
        area = height * width
        digits = math.log10(area)  # 面积位数
        match_ratio = 0
        if digits > 3:
            match_ratio = 1.2 - digits * 0.1

        """knn匹配sift特征"""
        gray = cv2.cvtColor(match_img, cv2.COLOR_RGB2GRAY)  # 转为灰度图
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(gray, None)  # 对待匹配的图生成sift特征
        bf = cv2.BFMatcher(cv2.NORM_L2, False)  # 交叉验证
        for index, row in label_infos.iterrows():
            descriptors2 = row["feature"]  # 与之匹配的图片库的sift特征
            good_point = 0  # 与图片库中每张图片匹配成功的点（好点）
            try:
                matches = bf.knnMatch(descriptors1, descriptors2, k=2)
                """lowe方法：用最佳距离和次佳距离比较，计算与每张图片匹配成功的点占比（比率测试）"""
                for first, second in matches:
                    if first.distance / second.distance < match_ratio:
                        good_point += 1
                label_infos.loc[index, ["feature_dist"]] = good_point
            except Exception:
                continue

        """最大最小归一化"""
        feature_point_range = label_infos["feature_dist"].max() - label_infos["feature_dist"].min()
        if feature_point_range > 0:
            label_infos["feature_dist"] = (label_infos["feature_dist"].max() - label_infos[
                "feature_dist"]) / feature_point_range

        ff = time()
        print("ff-dd", ff - dd)

        #################################  评价结果  ##########################################

        """按欧式距离综合排序：颜色+特征"""
        label_infos["eval_dist"] = np.sqrt(
            label_infos["color_dist"] ** 2 + label_infos["feature_dist"] ** 2)
        label_infos = label_infos.sort_values("eval_dist")

        return label_infos["name"].tolist()


if __name__ == "__main__":
    img_match = ImageMatch(Path(r"C:\Users\98276\PycharmProjects\rolabelnew\libs\data\yanhe"))
    img_match.check_scene_structrue()
    img_match.generate_model_with_thumb()
