"""
显示图像匹配的结果
"""

from ykjSettings import YkjSettings
from imgMatcher import ImageMatch

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import uuid
from PIL import Image
from time import time
from pathlib import Path

from libs.lib import newIcon, labelValidator

BB = QDialogButtonBox


class CustomImgItem(QWidget):
    """图片Widget"""

    clickedSignal = pyqtSignal()

    def __init__(self, img_path: Path, scaled_width=200, scaled_height=200, parent=None):
        super(CustomImgItem, self).__init__(parent=parent)
        self.img_path = img_path  # 图片路径
        self.img_name = img_path.name  # 图片名

        self.enable_match = True  # 是否允许匹配

        img_layout = QGridLayout(self)

        """摆放图片"""
        pixmap = QPixmap(str(img_path))
        label1 = QLabel(pixmap=pixmap)
        label1.setAlignment(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter)
        label1.setFixedHeight(scaled_height)  # 让图片占据固定的空间，模仿Windows 10的文件管理器

        """摆放图片名"""
        label2 = QLabel()
        wrap_text = self.get_wrap_text(self.img_name, scaled_width)
        label2.setText(wrap_text)
        label2.setMargin(10)
        label2.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)

        img_layout.addWidget(label1, 0, 0)
        img_layout.addWidget(label2, 1, 0)

        img_layout.setVerticalSpacing(0)  # 紧密排列

        self.installEventFilter(self)
        self.setMouseTracking(True)

    def eventFilter(self, source, event):

        if event.type() == QEvent.Leave:
            self.setStyleSheet("background-color: (240,240,240);")
            return True
        if event.type() == QEvent.Enter and self.rect().contains(event.pos()):
            self.setStyleSheet("background-color: white;")
            return True
        if event.type() == QEvent.MouseButtonPress and self.rect().contains(event.pos()):
            self.clickedSignal.emit()
            return True
        if event.type() == QEvent.MouseButtonDblClick and self.rect().contains(event.pos()):
            return True

        return False

    def get_wrap_text(self, text, scaled_width):
        text_width = QWidget.fontMetrics(self).horizontalAdvance(self.img_name)  # 字符串在window中的像素长度
        text_lines = (text_width + scaled_width - 1) // scaled_width  # 字符串卷曲的行数

        wrap_text = []
        quotient, remainder = divmod(len(text), text_lines)
        for i in range(text_lines):
            first_index = i * quotient
            end_index = (i + 1) * quotient if i < text_lines - 1 else None
            temp = text[first_index:end_index]
            wrap_text.append(temp)
        wrap_text = "\n".join(wrap_text).rstrip()
        return wrap_text

    def get_img_name(self):
        """获得图片名"""
        return self.img_path.stem  # 去除.jpg后缀

class LabelDialog(QDialog):

    def __init__(self, parent=None, text="", image_dir='', image_names=None):
        super(LabelDialog, self).__init__(parent)

        """设置路径"""
        self.ykj_settings = YkjSettings()  # 配置
        self.scene_path = self.ykj_settings["scene_dir"]  # 场景路径
        if self.scene_path is None:
            self.scene_path = (Path(__file__).parent.absolute() / "data" / "default")  # 场景路径
            self.ykj_settings["scene_dir"] = str(self.scene_path)  # 路径存字符串形式
        else:
            self.scene_path = Path(self.scene_path)  # 场景路径
        self.labels_found_path = (self.scene_path / "labels_found")  # labels_found文件夹
        self.labels_not_found_path = (self.scene_path / "labels_not_found")  # labels_found文件夹
        self.model_path = (self.scene_path / "model")  # model文件夹
        self.thumb_path = (self.scene_path / "thumb")  # thumb文件夹

        """匹配算法"""
        self.img_match = ImageMatch(self.scene_path)
        self.img_match.check_scene_structrue()
        self.img_match.generate_model_with_thumb()

        self.thumb_items = self.get_thumb_items()  # 用于展示的缩略图（未排序）
        self.sorted_thumb_items = []  # 用于展示的缩略图（已排序）
        self.sorted_labels = []  # 排序好的标签

        self.img = None  # 待匹配图片

        """编辑框"""
        self.edit = QLineEdit()
        self.edit.setText(text)
        self.edit.setValidator(labelValidator())
        self.edit.editingFinished.connect(self.postProcess)
        v_layout = QVBoxLayout()
        v_layout.addWidget(self.edit)

        """显示labels_found路径"""
        self.storage_label = QLabel("storage path: " + str(self.labels_found_path))  # 显示存储路径
        self.storage_label.setWordWrap(True)  # 文字随窗口缩放变化
        v_layout.addWidget(self.storage_label)

        """按钮"""
        h_layout = QHBoxLayout()
        h_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Fixed))
        self.scene_button = QPushButton("scene")  # 选择场景中的labels_found文件夹
        search_button = QPushButton("search")
        difficulty_button = QPushButton("not sure")
        ok_button = QPushButton("ok")
        cancel_button = QPushButton("cancel")

        # 给按钮设置快捷键
        search_button.setShortcut("Ctrl+F")
        difficulty_button.setShortcut("Ctrl+Space")
        ok_button.setShortcut("Ctrl+Return")
        cancel_button.setShortcut("Ctrl+D")

        # 设置快捷键提示
        search_button.setToolTip("Ctrl+F")
        difficulty_button.setToolTip("Ctrl+Space")
        ok_button.setToolTip("Ctrl+Enter")
        cancel_button.setToolTip("Ctrl+D")

        h_layout.addWidget(self.scene_button)
        h_layout.addWidget(search_button)
        h_layout.addWidget(difficulty_button)
        h_layout.addWidget(ok_button)
        h_layout.addWidget(cancel_button)
        v_layout.addLayout(h_layout)

        self.scene_button.clicked.connect(self.set_scene)  # 设置数据集的场景，选择该场景下的labels_found文件夹
        search_button.clicked.connect(self.on_search)
        ok_button.clicked.connect(self.validate)
        difficulty_button.clicked.connect(self.set_difficulty)
        cancel_button.clicked.connect(self.reject)
        self.move(0, 0)

        """展示图片"""
        # 规定gridlayout每个格子的固定大小
        self.scaled_width = 200
        self.scaled_height = 200
        # gridLayout列数
        self.col_num = 7

        self.resize(self.scaled_width * self.col_num, self.scaled_height * 2)  # 缩放窗口

        self.scrollArea = QScrollArea(widgetResizable=True)
        self.contentWidget = QWidget(self)  # 展示的内容窗口

        self.scrollArea.setGeometry(self.rect())

        self.scrollArea.setWidget(self.contentWidget)
        self.gridLayout = QGridLayout(self.contentWidget)
        self.gridLayout.setSpacing(0)  # 紧密排列

        v_layout.addWidget(self.scrollArea)
        self.setLayout(v_layout)

    def get_thumb_items(self):
        """获得缩略图控件"""
        img_items = {}
        thumb_paths = self.thumb_path.glob("*.png")
        for thumb_path in thumb_paths:
            label = thumb_path.stem  # 标签名
            img_item = CustomImgItem(thumb_path, parent=self)
            img_item.clickedSignal.connect(self.set_clicked_img_items)
            img_items[label] = img_item
        return img_items

    def get_sorted_thumb_items(self):
        """
        排序缩略图控件
        """

        self.sorted_thumb_items = []
        for label in self.sorted_labels:
            self.sorted_thumb_items.append(self.thumb_items[label])

    def get_filtered_thumb_items(self):
        """
        筛选缩略图控件
        """

        """获得按文字筛选的图片"""
        text = self.edit.text()
        keywords = text.strip().split(" ")  # 搜索的关键字

        """只要发现关键字，就筛出图片（所有关键字and查询，不是or查询）"""
        self.sorted_thumb_items = []
        for label in self.sorted_labels:
            sign = 0
            for j in keywords:
                if j in label:
                    sign += 1
            if sign == len(keywords):
                self.sorted_thumb_items.append(self.thumb_items[label])

    def on_search(self):
        """获得筛选后的缩略图"""
        self.get_filtered_thumb_items()
        self.delete_grid_items()
        self.render_grid_items()

    def set_scene(self):
        """
        设置数据集场景
        :return:
        """

        dir_path = QFileDialog.getExistingDirectory(self, "选取文件夹", str(self.scene_path))  # 获得labels_found路径
        dir_path = Path(dir_path)
        dir_name = dir_path.stem  # 获取打开的文件夹名

        """验证是否打开labels_found文件夹"""
        # 如果什么也没选，就什么也不做
        if dir_name == "":
            return
        # 如果选中的文件夹名不是labels_found，弹出警告
        elif dir_name != "labels_found":
            self.show_message("Please select the correct directory called 'labels_found' !")  # 文件名必须是labels_found
            return
        # 如果选中的文件夹名是labels_found，重新渲染gridLayout
        else:
            self.storage_label.setText("storage path: " + str(dir_path))  # 刷新路径

            self.scene_path = dir_path.parent  # 场景文件夹
            self.labels_found_path = (self.scene_path / "labels_found")  # labels_found文件夹
            self.labels_not_found_path = (self.scene_path / "labels_not_found")  # labels_found文件夹
            self.model_path = (self.scene_path / "model")  # model文件夹
            self.thumb_path = (self.scene_path / "thumb")  # thumb文件夹

            self.ykj_settings["scene_dir"] = str(self.scene_path)

            """匹配算法"""
            self.img_match = ImageMatch(self.scene_path)
            self.img_match.check_scene_structrue()
            self.img_match.generate_model_with_thumb()

            self.thumb_items = self.get_thumb_items()  # 用于展示的缩略图（未排序）

            self.render_item_process()  # 在gridLayout上渲染图片

    def render_item_process(self):
        """
        在gridLayout上渲染图片item的整套流程
        """
        self.delete_grid_items()  # 清空
        c = time()
        self.sorted_labels = self.img_match.match(self.img)  # 匹配
        d = time()
        self.get_sorted_thumb_items()  # 排序
        print("d-c", d - c)
        self.render_grid_items()  # 渲染

    def show_message(self, text, icon=QMessageBox.Icon.Critical, flag=QMessageBox.Ok):
        """
        显示提示
        :param text: 提示的信息
        :return:
        """
        msg = QMessageBox()
        msg.setWindowTitle("warning")
        msg.setIcon(icon)
        msg.setInformativeText(text)
        msg.setStandardButtons(flag)
        result = msg.exec()  # 显示窗口（运行结果）
        return result

    def set_enable_match(self, enable_match):
        """
        是否允许匹配
        """

        self.enable_match = enable_match

    def validate(self):
        text = self.edit.text().strip()  # 编辑的文字
        if text:
            if text.startswith("unknown_"):
                """不能识别出的图片（unknown_前缀）保存在labels_not_found文件夹"""
                img_path = (self.labels_not_found_path / (text + ".jpg"))
                # 如果labels_not_found文件夹中含有相同名字的图片，则弹出提示
                if img_path.exists():
                    self.show_message(
                        "There exits the same name image in 'labels_not_found' directory, please select another name for this image !")
                    return
                img = Image.fromarray(self.img)
                img.save(img_path)
            self.accept()

    def postProcess(self):
        try:
            self.edit.setText(self.edit.text().trimmed())
        except AttributeError:
            # PyQt5: AttributeError: 'str' object has no attribute 'trimmed'
            self.edit.setText(self.edit.text())

    def set_img(self, img):
        """
        设置待匹配的图片
        """
        self.img = img

    def popUp(self, text=''):
        self.scrollArea.verticalScrollBar().setValue(0)  # 每次弹出对话框，滚动条置顶
        self.render_item_process()  # 在gridLayout上渲染图片

        self.edit.setText(text)
        self.edit.setSelection(0, len(text))
        self.edit.setFocus(Qt.PopupFocusReason)
        return self.edit.text() if self.exec_() else None

    def listItemClick(self, tQListWidgetItem):
        try:
            text = tQListWidgetItem.text().trimmed()
        except AttributeError:
            # PyQt5: AttributeError: 'str' object has no attribute 'trimmed'
            text = tQListWidgetItem.text().strip()
        self.edit.setText(text)
        self.validate()

    def resizeEvent(self, event):
        """
        监听窗口尺寸变化的事件
        :param event:
        :return:
        """
        # 计算girdLayout拉伸时各元素的列距总和
        horizontal_space = self.scrollArea.width() - self.scaled_width * (self.col_num + 1)
        # 当拉伸的长度，在列方向上可以容纳girdLayout一个元素时，加进一个元素
        if horizontal_space >= self.scaled_width:
            self.col_num += 1
            self.delete_grid_items()
            self.render_grid_items()
        # 当缩短的长度，在列方向上挤占girdLayout一个元素时，减去一个元素
        if horizontal_space <= 0:
            self.col_num -= 1
            # 防止gridLayout的列数减为0
            if self.col_num <= 0:
                self.col_num = 1
                return
            self.delete_grid_items()
            self.render_grid_items()

    def render_grid_items(self):
        """
        重新绘制girdLayout中的元素
        :return:
        """
        # 在gridLayout摆放图片
        for index, img_item in enumerate(self.sorted_thumb_items):
            # 决定图片在gridLayout的摆放位置
            row_index = (index + self.col_num) // self.col_num - 1
            col_index = index % self.col_num

            self.gridLayout.addWidget(img_item, row_index, col_index)

        # gridLayout在行方向上各元素占比为1，防止相互挤占
        row_num = (len(self.sorted_thumb_items) + self.col_num - 1) // self.col_num
        for index in range(row_num):
            self.gridLayout.setRowStretch(index, 1)

    def delete_grid_items(self):
        """
        删除gridLayout中的元素
        :return:
        """
        for i in reversed(range(self.gridLayout.count())):
            self.gridLayout.itemAt(i).widget().setParent(None)

    def set_clicked_img_items(self):
        """
        点击图片，设置编辑框显示
        """
        img_item = self.sender()
        img_name = img_item.get_img_name()
        self.edit.setText(img_name)

    def set_difficulty(self):
        """如果不确定，加上unknown前缀，并保存图片到labels_not_found文件夹"""
        text = "unknown_" + uuid.uuid4().hex  # 如果不确定，则随机生成不重复的文件名
        self.edit.setText(text)
