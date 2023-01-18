"""
识别新的类
"""

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from enum import Enum
from PyQt5.QtCore import QEvent
import shutil
from ykjSettings import YkjSettings
from imgMatcher import ImageMatch
from pathlib import Path


class ItemBackGround(str, Enum):
    """控制选中item的背景颜色"""
    LEFT_CLICK = "background-color: white;"
    RIGHT_CLICK = "background-color: rgb(210,210,210);"
    NO_CLICK = "background-color: rgb(240,240,240);"


class ItemClickSign(Enum):
    """控制item选中状态"""
    CLICK = 1
    NO_CLICK = 0


class CustomImgItem(QWidget):
    """图片Widget"""
    right_clicked_signal = pyqtSignal(str)  # 右键点击事件
    left_clicked_signal = pyqtSignal(str)  # 左键点击事件

    def __init__(self, img_path: Path, scaled_width=200, scaled_height=200, parent=None):
        super(CustomImgItem, self).__init__(parent=parent)
        self.scaled_width = scaled_width  # 缩放后的宽
        self.scaled_height = scaled_height  # 缩放后的高

        self.img_name = img_path.stem  # 图片名
        self.img_path = img_path  # 路径

        self.right_clicked = ItemClickSign.NO_CLICK.value  # 右键单击后label背景颜色切换（界面上可以出现多个右键选中的item）
        self.left_clicked = ItemClickSign.NO_CLICK.value  # 左键单击后label背景颜色切换（加上限制，界面上只能出现一次左键选中的item）

        img_layout = QGridLayout(self)

        """调整图片大小"""
        pixmap = QPixmap(str(img_path))
        # 按纵横比自适应缩放图片
        if pixmap.width() >= pixmap.height():
            pixmap = pixmap.scaled(self.scaled_width, int(self.scaled_width * pixmap.height() / pixmap.width()))
        else:
            pixmap = pixmap.scaled(int(self.scaled_height * pixmap.width() / pixmap.height()), self.scaled_height)

        """摆放图片"""
        self.label1 = QLabel(pixmap=pixmap)
        self.label1.setAlignment(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter)
        self.label1.setFixedHeight(self.scaled_height)  # 让图片占据固定的空间，模仿Windows 10的文件管理器

        """摆放图片名"""
        self.label2 = QLabel()
        wrap_text = self.get_wrap_text(self.img_name + img_path.suffix, self.scaled_width)
        self.label2.setText(wrap_text)
        self.label2.setMargin(10)
        self.label2.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)

        img_layout.addWidget(self.label1, 0, 0)
        img_layout.addWidget(self.label2, 1, 0)
        img_layout.setVerticalSpacing(0)  # 紧密排列

        self.installEventFilter(self)
        self.setMouseTracking(True)

    def eventFilter(self, source, event):

        if event.type() == QEvent.MouseButtonPress and self.rect().contains(event.pos()):
            # 左键单选
            if event.button() == Qt.LeftButton:
                self.left_clicked_signal.emit(self.get_img_name())
                return True
            # 右键多选
            if event.button() == Qt.RightButton:
                self.right_clicked_signal.emit(self.get_img_name())
                return True
        if event.type() == QEvent.MouseButtonDblClick and self.rect().contains(event.pos()):
            print("double click:" + self.img_name)
            return True

        return False

    def get_wrap_text(self, text, scaled_width):
        """
        折叠过长的文字，便于label多行显示
        :param text:
        :param scaled_width:
        :return:
        """
        text_width = QWidget.fontMetrics(self).horizontalAdvance(text)  # 字符串在window中的像素长度
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
        return self.img_name  # 去除.jpg后缀

    def set_img_name(self, text):
        """
        重置图片名
        :param text: 编辑框输入的图片名
        :return:
        """

        self.img_name = text
        wrap_text = self.get_wrap_text(self.img_name + self.img_path.suffix, self.scaled_width)
        self.label2.setText(wrap_text)


class RecognizeLabelsDialog(QDialog):
    def __init__(self, parent=None):
        super(RecognizeLabelsDialog, self).__init__(parent)

        self.setWindowTitle("recognize labels")

        """设置路径"""
        self.ykj_settings = YkjSettings()  # 配置
        self.scene_path = self.ykj_settings["scene_dir"]  # 场景路径
        if self.scene_path is None:
            self.scene_path = (Path(__file__).parent.absolute() / "data" / "default")  # 场景路径
            self.ykj_settings["scene_dir"] = str(self.scene_path)  # 路径存字符串形式
        else:
            self.scene_path = Path(self.scene_path)
        self.labels_found_path = (self.scene_path / "labels_found")  # labels_found文件夹
        self.labels_not_found_path = (self.scene_path / "labels_not_found")  # labels_found文件夹
        self.model_path = (self.scene_path / "model")  # model文件夹
        self.thumb_path = (self.scene_path / "thumb")  # thumb文件夹

        """匹配算法"""
        self.img_match = ImageMatch(self.scene_path)
        self.img_match.check_scene_structrue()
        self.img_match.generate_model_with_thumb()

        main_v_layout = QVBoxLayout(self)

        """编辑框"""
        self.edit = QLineEdit()  # 编辑框
        self.edit.editingFinished.connect(self.post_process)
        main_v_layout.addWidget(self.edit)

        """显示labels_not_found路径"""
        self.storage_label = QLabel("storage path: " + str(self.labels_not_found_path))  # 显示存储路径
        self.storage_label.setWordWrap(True)  # 文字随窗口缩放变化
        main_v_layout.addWidget(self.storage_label)

        # 获取图片
        self.img_items = self.get_img_item(self.labels_not_found_path)

        """编辑按钮"""
        h_layout_left_button = QHBoxLayout()
        h_layout_left_button.addItem(
            QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Fixed))  # 将按钮推到最右边

        self.scene_button = QPushButton("scene")  # 选择场景
        self.new_class_button = QPushButton("new label")  # 添加到新类
        self.rename_button = QPushButton("rename")  # 重命名

        # 取消焦点
        self.new_class_button.setFocusPolicy(Qt.NoFocus)
        self.scene_button.setFocusPolicy(Qt.NoFocus)
        self.rename_button.setFocusPolicy(Qt.NoFocus)

        self.scene_button.clicked.connect(self.set_scene)  # 设置数据集的场景，选择该场景下的labels_not_found文件夹
        self.new_class_button.clicked.connect(self.move_to_labels_found)  # 定义为新类
        self.rename_button.clicked.connect(self.rename_item)

        h_layout_left_button.addWidget(self.scene_button)
        h_layout_left_button.addWidget(self.new_class_button)
        h_layout_left_button.addWidget(self.rename_button)

        self.set_left_edit_enabled(False)  # 初始禁用左键编辑单个图像

        main_v_layout.addLayout(h_layout_left_button)

        # 除去最大化按钮，防止渲染gridLayout时出错
        self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)

        # 规定gridlayout每个格子的固定大小
        self.scaled_width = 200
        self.scaled_height = 200
        # gridLayout列数
        self.col_num = 5

        """左右键单击的元素"""
        self.right_clicked_items = []  # 记录右键单击的元素数

        self.left_prev_clicked_item = None  # 记录左键前一单击的元素数
        if len(self.img_items) != 0:
            self.left_prev_clicked_item = self.img_items[0]
        self.left_current_img_item = None  # 记录左键当前单击的元素数

        self.resize(self.scaled_width * self.col_num, self.scaled_height * 2)
        self.scrollArea = QScrollArea(widgetResizable=True)
        self.contentWidget = QWidget(self)  # 展示的内容窗口
        self.scrollArea.setGeometry(self.rect())

        self.scrollArea.setWidget(self.contentWidget)
        self.gridLayout = QGridLayout(self.contentWidget)
        self.gridLayout.setSpacing(0)  # 紧密排列

        main_v_layout.addWidget(self.scrollArea)

        """编辑多个图片，配合右键"""
        h_layout2 = QHBoxLayout()
        h_layout2.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Fixed))
        self.delete_button = QPushButton("delete")  # 删除（多张图片）
        self.delete_button.setFocusPolicy(Qt.NoFocus)  # 取消焦点
        self.delete_button.setEnabled(False)  # 默认禁用，点击图片时才启用
        self.delete_button.clicked.connect(self.delete_right_click_items)
        h_layout2.addWidget(self.delete_button)
        main_v_layout.addLayout(h_layout2)

        self.setLayout(main_v_layout)
        self.render_grid_items()

    def post_process(self):
        """编辑框编辑完后去除首尾的空格"""
        self.edit.setText(self.edit.text().strip())

    def delete_right_click_items(self):
        """
        右键多选删除元素
        :return:
        """

        # 如果右键点击了元素，那么可以删除
        if len(self.right_clicked_items) != 0:
            """对话框提示"""
            result = self.show_message("Are you sure to delete these files ?",
                                       icon=QMessageBox.Icon.Question,
                                       flag=QMessageBox.Ok | QMessageBox.Cancel)
            # 如果取消，则退出
            if result == QMessageBox.Cancel:
                return
            # 如果确定，删除文件
            elif result == QMessageBox.Ok:
                self.delete_button.setEnabled(False)  # 禁用delete按钮

                """删除本地文件"""
                for i in self.right_clicked_items:
                    delete_img_path = (self.labels_not_found_path / (i + ".jpg"))  # 待删除的图片路径
                    delete_img_path.unlink()

                """删除界面上右键多选的item"""
                self.delete_grid_items()
                # 找寻删除的item
                del_items = []
                for i in self.img_items:
                    if i.get_img_name() in self.right_clicked_items:
                        del_items.append(i)
                # 删除界面上的item
                for i in del_items:
                    self.right_clicked_items.remove(i.get_img_name())
                    self.img_items.remove(i)
                self.render_grid_items()

    def rename_item(self):
        """
        重命名labels_not_found文件夹中的元素

        :return:
        """

        text = self.edit.text().strip()  # 输入的文字去除空格
        file_name = self.left_current_img_item.get_img_name()  # 被选中的图片名
        src_path = (self.labels_not_found_path / (file_name + ".jpg"))  # 源文件路径
        dst_path = (self.labels_not_found_path / (text + ".jpg"))  # 目标文件路径

        self.edit.setText(text)  # 显示文字

        """处理异常"""
        # 文件名不能为空
        if text == "":
            self.show_message("Please enter the file name !")
            return
        # 如果源文件不存在，弹出警告
        if not src_path.exists():
            self.show_message("File does not exist !")
            return
        # 如果目标文件存在，弹出警告
        if dst_path.exists():
            self.show_message("File exists, please select another file name !")
            return

        """对话框提示"""
        result = self.show_message("Are you sure to rename this file ?",
                                   icon=QMessageBox.Icon.Question,
                                   flag=QMessageBox.Ok | QMessageBox.Cancel)
        # 如果确定就重命名本地文件
        if result == QMessageBox.Ok:
            src_path.replace(dst_path)  # 文件重命名
            self.left_current_img_item.set_img_name(text)  # 界面上的图片重命名
            return
        # 如果取消就退出
        elif result == QMessageBox.Cancel:
            return

    def set_left_edit_enabled(self, status=False):
        """
        设置左键启用/禁用的状态
        :param status:
        :return:
        """

        self.edit.setEnabled(status)
        self.new_class_button.setEnabled(status)
        self.rename_button.setEnabled(status)
        if status is False:
            self.edit.setText("")

    def set_scene(self):
        """
        设置数据集场景
        :return:
        """

        dir_path = QFileDialog.getExistingDirectory(self, "选取文件夹", str(self.scene_path))  # 弹出对话框，显示之前打开的文件夹
        dir_path = Path(dir_path)
        dir_name = dir_path.stem  # 文件夹名
        # 如果什么也没选，就什么也不做
        if dir_name == "":
            return
        # 如果选中的文件夹名不是labels_not_found，弹出警告
        elif dir_name != "labels_not_found":
            self.show_message(
                "Please select the correct directory called 'labels_not_found' !")  # 文件名必须是labels_not_found
        # 如果选中的文件夹名是labels_not_found，重新渲染gridLayout
        else:
            self.set_left_edit_enabled(False)  # 左键的编辑区域默认禁用
            self.storage_label.setText("storage path: " + str(dir_path))  # 刷新路径

            self.scene_path = dir_path.parent  # 场景文件夹
            self.ykj_settings["scene_dir"] = str(self.scene_path)
            self.labels_found_path = (self.scene_path / "labels_found")  # labels_found文件夹
            self.labels_not_found_path = (self.scene_path / "labels_not_found")  # labels_found文件夹
            self.model_path = (self.scene_path / "model")  # model文件夹
            self.thumb_path = (self.scene_path / "thumb")  # thumb文件夹

            """匹配算法"""
            self.img_match = ImageMatch(self.scene_path)
            self.img_match.check_scene_structrue()
            self.img_match.generate_model_with_thumb()

            # 在gridLayout上渲染图片
            self.delete_grid_items()
            self.img_items = self.get_img_item(dir_path)
            # 记录左键前一点击的元素
            if len(self.img_items) != 0:
                self.left_prev_clicked_item = self.img_items[0]
            self.render_grid_items()

    def get_img_item(self, img_dir: Path):
        """筛选一个文件夹下的所有图片"""
        img_paths = img_dir.glob("*.jpg")

        # 用于缓存图片，不必gridLayout每次渲染时都从dist读一遍图片
        img_items = []
        for img_path in img_paths:
            img_item = CustomImgItem(img_path)
            img_item.left_clicked_signal.connect(self.get_left_click_item)
            img_item.right_clicked_signal.connect(self.get_right_click_item_names)
            img_items.append(img_item)
        return img_items

    def show_message(self, text, icon=QMessageBox.Icon.Critical, flag=QMessageBox.Ok):
        """
        显示提示对话框
        :param text: 提示信息
        :return:
        """
        msg = QMessageBox()
        msg.setWindowTitle("warning")
        msg.setIcon(icon)  # 设置图标
        msg.setInformativeText(text)  # 设置提示信息
        msg.setStandardButtons(flag)
        result = msg.exec()  # 显示窗口（运行结果）
        return result

    def get_left_click_item(self, text):
        """
        获取左键点击的图片item（始终保证每次只能点击一张图）
        :param text:
        :return:
        """
        # 当前左键点击的控件
        img_item = self.sender()
        index = self.gridLayout.indexOf(img_item)
        self.left_current_img_item = self.img_items[index]

        # 如果右键选中的元素没有全部取消，那么左键点击时会弹出警告
        if len(self.right_clicked_items) > 0:
            self.show_message("Please unclick all files to be deleted by right mouse !")
            return
        # 如果没有点击右键元素
        elif len(self.right_clicked_items) == 0:
            # 如果之前点击的控件和当前点击的控件不是同一个
            if self.left_prev_clicked_item != self.left_current_img_item:
                # 切换背景颜色
                if self.left_prev_clicked_item.left_clicked == ItemClickSign.NO_CLICK.value:
                    if self.left_current_img_item.left_clicked == ItemClickSign.CLICK.value:
                        self.left_current_img_item.left_clicked = ItemClickSign.NO_CLICK.value
                        self.left_current_img_item.setStyleSheet(ItemBackGround.NO_CLICK.value)
                        self.set_left_edit_enabled(False)
                        self.left_prev_clicked_item = self.left_current_img_item
                        return
                    if self.left_current_img_item.left_clicked == ItemClickSign.NO_CLICK.value:
                        self.left_current_img_item.left_clicked = ItemClickSign.CLICK.value
                        self.left_current_img_item.setStyleSheet(ItemBackGround.LEFT_CLICK.value)
                        self.set_left_edit_enabled(True)
                        self.left_prev_clicked_item = self.left_current_img_item
                        return
                if self.left_prev_clicked_item.left_clicked == ItemClickSign.CLICK.value:
                    self.left_prev_clicked_item.left_clicked = ItemClickSign.NO_CLICK.value
                    self.left_prev_clicked_item.setStyleSheet(ItemBackGround.NO_CLICK.value)
                    self.left_current_img_item.left_clicked = ItemClickSign.CLICK.value
                    self.left_current_img_item.setStyleSheet(ItemBackGround.LEFT_CLICK.value)
                    self.set_left_edit_enabled(True)
                    self.left_prev_clicked_item = self.left_current_img_item
                    return
            # 如果之前点击的控件和当前点击的控件是同一个
            elif self.left_prev_clicked_item == self.left_current_img_item:
                if self.left_current_img_item.left_clicked == ItemClickSign.CLICK.value:
                    self.left_current_img_item.left_clicked = ItemClickSign.NO_CLICK.value
                    self.left_current_img_item.setStyleSheet(ItemBackGround.NO_CLICK.value)
                    self.set_left_edit_enabled(False)
                    self.left_prev_clicked_item = self.left_current_img_item
                    return
                elif self.left_current_img_item.left_clicked == ItemClickSign.NO_CLICK.value:
                    self.left_current_img_item.left_clicked = ItemClickSign.CLICK.value
                    self.left_current_img_item.setStyleSheet(ItemBackGround.LEFT_CLICK.value)
                    self.set_left_edit_enabled(True)
                    self.left_prev_clicked_item = self.left_current_img_item
                    return

    def move_to_labels_found(self):
        """
        从labels_not_found移动到labels_found文件夹
        """

        """删除文件"""
        file_name = self.left_current_img_item.get_img_name()  # 左键当前选中的图片名
        src_path = (self.labels_not_found_path / (file_name + ".jpg"))  # 源文件路径
        dst_path = (self.labels_found_path / (file_name + ".jpg"))  # 目标文件路径

        """处理异常"""
        # 如果源文件不存在，弹出警告
        if not src_path.exists():
            self.show_message("Source file does not exist !")
            return

        """对话框提示"""
        result = self.show_message("Are you sure to move this file to 'labels_found' directory ?",
                                   icon=QMessageBox.Icon.Question,
                                   flag=QMessageBox.Ok | QMessageBox.Cancel)
        # 如果确定，移动文件到labels_found文件夹（允许覆盖文件）
        if result == QMessageBox.Ok:
            """移动本地文件"""
            self.set_left_edit_enabled(False)  # 禁用编辑区域
            src_path.replace(dst_path)  # 移动文件
            self.img_match.generate_model_with_thumb()  # 重新计算图像匹配信息

            """删除界面上左键点击的元素"""
            self.delete_grid_items()
            # 找寻删除的元素（只有一个）
            del_item = None
            for i in self.img_items:
                if i.get_img_name() == file_name:
                    del_item = i
                    break
            self.img_items.remove(del_item)  # 删除元素
            self.render_grid_items()
            return
        # 如果取消，则退出
        elif result == QMessageBox.Cancel:
            return

    def get_right_click_item_names(self, text):
        """
        获取右键点击的图片名
        :param img_name:
        :return:
        """

        # 点击的控件
        current_img_item = self.sender()
        index = self.gridLayout.indexOf(current_img_item)
        current_img_item = self.img_items[index]

        # 如果一个左键点击的都没有，此时右键可以点击
        if self.left_prev_clicked_item.left_clicked == ItemClickSign.NO_CLICK.value and current_img_item.left_clicked == ItemClickSign.NO_CLICK.value:
            # 切换背景颜色
            if current_img_item.right_clicked == ItemClickSign.CLICK.value:
                current_img_item.right_clicked = ItemClickSign.NO_CLICK.value
                current_img_item.setStyleSheet(ItemBackGround.NO_CLICK.value)
                self.right_clicked_items.remove(text)
                # 如果取消点击后，右键没有选中的元素，delete按钮置为禁用
                if len(self.right_clicked_items) == 0:
                    self.delete_button.setEnabled(False)
                return
            elif current_img_item.right_clicked == ItemClickSign.NO_CLICK.value:
                current_img_item.right_clicked = ItemClickSign.CLICK.value
                current_img_item.setStyleSheet(ItemBackGround.RIGHT_CLICK.value)
                self.right_clicked_items.append(text)
                # 右键点击后，必选中元素，故启用delete按钮
                self.delete_button.setEnabled(True)
                return
        # 如果右键选中的元素没有全部取消，那么左键点击时会弹出警告
        else:
            self.show_message("Please unclick the file to be edited by left mouse!")
            return

    def resizeEvent(self, event):
        """
        监听窗口尺寸变化的事件
        :param event:
        :return:
        """

        # 计算girdLayout拉伸时各元素的列距总和
        self.horizontalSpace = self.scrollArea.width() - self.scaled_width * (self.col_num + 1)
        # 当拉伸的长度，在列方向上可以容纳girdLayout一个元素时，加进一个元素
        if self.horizontalSpace >= self.scaled_width:
            self.col_num += 1
            self.delete_grid_items()
            self.render_grid_items()
        # 当缩短的长度，在列方向上挤占girdLayout一个元素时，减去一个元素
        if self.horizontalSpace <= 0:
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
        for index, img_item in enumerate(self.img_items):
            # 决定图片在gridLayout的摆放位置
            row_index = (index + self.col_num) // self.col_num - 1
            col_index = index % self.col_num

            self.gridLayout.addWidget(img_item, row_index, col_index)

        # gridLayout在行方向上各元素占比为1，防止相互挤占
        row_num = (len(self.img_items) + self.col_num - 1) // self.col_num
        for index in range(row_num):
            self.gridLayout.setRowStretch(index, 1)

    def delete_grid_items(self):
        """
        删除gridLayout中的元素
        :return:
        """
        for i in reversed(range(len(self.img_items))):
            self.gridLayout.itemAt(i).widget().setParent(None)


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    w = RecognizeLabelsDialog()
    w.show()
    sys.exit(app.exec_())
