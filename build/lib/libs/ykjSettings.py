"""
禹空间配置
"""

import PyQt5.QtCore as qc


class YkjSettings(object):

    def __init__(self):
        self.data = qc.QSettings("config.ini", "qc.QSettings.IniFormat")

    def __setitem__(self, key, value):
        self.data.setValue(key, value)

    def __getitem__(self, key):
        return self.data.value(key)


if __name__ == "__main__":
    a = YkjSettings()
    a["scene_dir"] = None
