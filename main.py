import numpy as np
import sys
import os
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
matplotlib.use('Qt5Agg')
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt,QTimer

from utils.feature import *


class App(QWidget):
    def __init__(self, parent=None):
        # 父类初始化方法
        super(App, self).__init__(parent)
        self.status_dict = {
            "decomposition": "pca",
            "feature": ["Response", "Sum", "Time"],
            "data_dir":"./data",
            "class_dict": {
            }
        }

        self.class_dir_dict = {
        }
        self.initUI()

    def initUI(self):
        self.setWindowTitle('降维绘图')
        # self.setFixedSize(1200, 700)
        # self.setMinimumSize(1200, 700)
        # self.setMaximumSize(1200, 700)
        #菜单栏

        # self.myQMenuBar = QMenuBar()
        # self.myQMenuBar.resize(self.myQMenuBar.sizeHint())
        # opendir = QAction('dir', self)
        # opendir.triggered.connect(self.open_data_dir)
        # self.myQMenuBar.addAction(opendir)

        # 几个QWidgets

        self.plot_button = QPushButton('绘图')
        self.plot_button.clicked.connect(self.refresh_fig)

        self.load_data_button = QPushButton('加载')
        self.load_data_button.clicked.connect(self.open_data_dir)


        self.canvas = MyMatplotlibFigure()
        self.right = self.init_right()
        self.top_left = self.init_top_left()

        # self.endBtn = QPushButton('结束')

        # self.endBtn.clicked.connect(self.endTimer)
        # 时间模块
        # self.timer = QTimer(self)
        # self.timer.timeout.connect(self.showTime)
        # 图像模块
        # self.figure = plt.figure()

        # 垂直布局
        # all_widget = QWidget()
        # all_grid = QGridLayout()
        # all_grid.setSpacing(10)
        # all_grid.addWidget(self.myQMenuBar,0,0,1,0)

        # HBox_widget = QWidget()
        HBox_layout = QHBoxLayout(self)
        splitter_left = QSplitter(Qt.Vertical)
        splitter_left.addWidget(self.top_left)
        splitter_left.addWidget(self.canvas)

        splitter_all = QSplitter(Qt.Horizontal)
        splitter_all.addWidget(splitter_left)
        splitter_all.addWidget(self.right)
        # splitter_all.addWidget(self.plot_button)

        # layout.addWidget(self.startBtn)
        # layout.addWidget(self.endBtn)
        # layout.addWidget(self.canvas)
        HBox_layout.addWidget(splitter_all)
        # HBox_widget.setLayout(HBox_layout)

        # all_grid.addWidget(HBox_widget)
        # all_widget.setLayout(all_grid)
        self.setLayout(HBox_layout)
        # self.setCentralWidget(splitter_all)
        # 数组初始化
        # self.x = []

    def open_data_dir(self):
        fname = QFileDialog.getExistingDirectory(self,"选取文件夹","./")
        if fname:
            self.status_dict["data_dir"] = fname
            self.refresh_class_grid()
        else:
            return
        # print(fname)

    def init_right(self):
        return_widget = QWidget()
        right_grid = QGridLayout()
        right_grid.setSpacing(10)

        sc = self.init_class_check_box()
        right_grid.addWidget(sc, 1, 0)
        right_grid.addWidget(self.plot_button, 2, 0)

        return_widget.setLayout(right_grid)
        # return_widget.update()
        return return_widget

    def init_class_check_box(self):
        sc = QScrollArea()

        self.class_qw = QWidget()
        self.class_grid = QGridLayout()
        self.class_grid.setSpacing(10)

        self.refresh_class_grid()

        self.class_qw.setLayout(self.class_grid)

        sc.setWidget(self.class_qw)
        return sc

    def refresh_class_grid(self):
        for i in range(self.class_grid.count())[:]:
            self.class_grid.itemAt(i).widget().deleteLater()

        self.status_dict["class_dict"] = {}
        self.class_dir_dict = {}
        for file_dir in os.listdir(self.status_dict["data_dir"]):
            if os.path.isdir(os.path.join(self.status_dict["data_dir"], file_dir)):
                # class_list.append(file_dir)
                self.status_dict["class_dict"][file_dir]=os.path.join(self.status_dict["data_dir"], file_dir)
                self.class_dir_dict[file_dir]=os.path.join(self.status_dict["data_dir"], file_dir)

        print(self.status_dict["class_dict"])
        for class_name in self.status_dict["class_dict"].keys():
            cb = QCheckBox(class_name)
            cb.toggle()
            cb.clicked.connect(self.changeClass)
            self.class_grid.addWidget(cb)

        self.canvas.load_data(self.class_dir_dict)

        self.timer = QTimer(self)
        self.timer.timeout.connect(lambda: self.class_qw.resize(self.class_qw.sizeHint()))
        self.timer.start(1)
        # self.class_qw.resize(self.sizeHint())

    def init_top_left(self):
        hbox_top_widget = QWidget()
        hbox_top = QHBoxLayout()

        # pca/lda方法选择框
        method_combo = QComboBox()
        method_combo.addItem("pca")
        method_combo.addItem("lda")
        method_combo.currentIndexChanged.connect(self.method_combo_change)
        hbox_top.addWidget(method_combo)

        # 特征选择框
        feature_widget = QWidget()
        feature_grid = QGridLayout()
        for i, feature_name in enumerate(["Response", "Sum", "Time"]):
            cb = QCheckBox(feature_name, self)
            cb.toggle()
            cb.clicked.connect(self.changeFeature)
            feature_grid.addWidget(cb, i // 2, i % 2)
        feature_widget.setLayout(feature_grid)
        hbox_top.addWidget(feature_widget)
        hbox_top_widget.setLayout(hbox_top)

        hbox_down_widget = QWidget()
        hbox_down = QHBoxLayout()
        hbox_down.addWidget(self.load_data_button)
        hbox_down.addWidget(self.plot_button)
        hbox_down_widget.setLayout(hbox_down)

        return_widget = QWidget()
        vbox = QVBoxLayout()
        vbox.addWidget(hbox_top_widget)
        vbox.addWidget(hbox_down_widget)
        return_widget.setLayout(vbox)
        return return_widget

    def changeClass(self,state):
        sender = self.sender()
        if state == True:
            self.status_dict["class_dict"][sender.text()] = self.class_dir_dict[sender.text()]
        else:
            if len(self.status_dict["class_dict"])==2:
                # self.status_dict["feature"].remove(sender.text())
                sender.setChecked(True)
            else:
                del self.status_dict["class_dict"][sender.text()]
        print(self.status_dict["class_dict"],self.class_dir_dict)

    def changeFeature(self,state):
        sender = self.sender()
        if state == True:
            self.status_dict["feature"].append(sender.text())
            # sender.setChecked(True)
        else:
            if len(self.status_dict["feature"])==1:
                # self.status_dict["feature"].remove(sender.text())
                sender.setChecked(True)
            else:
                self.status_dict["feature"].remove(sender.text())
            # sender.setChecked(False)
        print(self.status_dict["feature"])

    def method_combo_change(self):
        sender = self.sender()
        self.status_dict["decomposition"]=sender.currentText()
        print(self.status_dict["decomposition"])

    def refresh_fig(self):
        # plt.cla()
        # # 获取绘图并绘制
        # fig = plt.figure()
        # ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        # ax.plot(np.random.randint(0, 10, [5, 1]), 'o--')
        # cavans = FigureCanvas(fig)
        # cavans.draw()
        self.canvas.mat_plot_drow_axes(self.status_dict)
        print(self.status_dict)

        # shuju = np.random.random_sample() * 10  # 返回一个[0,1)之间的浮点型随机数*10
        # self.x.append(shuju)  # 数组更新
        # ax = self.figure.add_axes([0.1, 0.1, 0.8, 0.8])
        # ax.clear()
        # ax.plot(self.x)
        # ax.plot(np.random.randint(0, 10, [5, 1]), 'o--')
        # self.canvas.draw()

    # 启动函数
    # def startTimer(self):
    #     # 设置计时间隔并启动
    #     self.timer.start(2000)  # 每隔一秒执行一次绘图函数 showTime
    #     self.startBtn.setEnabled(False)  # 开始按钮变为禁用
    #     self.endBtn.setEnabled(True)  # 结束按钮变为可用F
    #
    # def endTimer(self):
    #     self.timer.stop()  # 计时停止
    #     self.startBtn.setEnabled(True)  # 开始按钮变为可用
    #     self.endBtn.setEnabled(False)  # 结束按钮变为可用
    #     self.x = []  # 清空数组


class MyMatplotlibFigure(FigureCanvas):
    """
    创建一个画布类，并把画布放到FigureCanvasQTAgg
    """

    def __init__(self, width=10, heigh=10, dpi=100):
        # plt.rcParams['figure.facecolor'] = 'r'  # 设置窗体颜色
        # plt.rcParams['axes.facecolor'] = 'b'  # 设置绘图区颜色
        # 创建一个Figure,该Figure为matplotlib下的Figure，不是matplotlib.pyplot下面的Figure
        self.figs = Figure(figsize=(width, heigh), dpi=dpi)
        super(MyMatplotlibFigure, self).__init__(self.figs)  # 在父类种激活self.fig，
        self.axes = self.figs.add_subplot(111)  # 添加绘图区
        self.data_dict={}
        # self.colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self.colors = list(plt.cm.tab10(np.arange(10)))
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        self.signal_num = 0

    def load_data(self,class_dir_dict):
        self.data_dict = {}

        def read_file_data(file_path):  # 至少20个以上的采样点！
            data = np.genfromtxt(file_path, delimiter='\t')
            # 删除nan序列
            data = np.delete(data, np.argwhere(np.isnan(data[1, :])), axis=1)
            data = np.concatenate((data, np.arange(data.shape[0]).reshape([-1, 1])), axis=1)
            # 删除时间序列
            # if num_sig and data.shape[1] > num_sig:
            #     print('已删除时间序列', end=",")
            #     data = data[:, -num_sig:]
            # else:
            count_list = np.zeros(data.shape[1])
            sig_len = data.shape[0]
            for i in range(10):  # 取10次，找出时间序列
                rand_indexs = np.sort(np.random.randint(0, sig_len, 3))
                temp_a = np.abs((rand_indexs[1] - rand_indexs[2]) * data[rand_indexs[0], :] - (
                            rand_indexs[0] - rand_indexs[2]) * data[rand_indexs[1], :] + (
                                            rand_indexs[0] - rand_indexs[1]) * data[rand_indexs[2], :])
                zero_index = np.argwhere(temp_a < 1e-5)
                count_list[zero_index] += 1
            timearray_index = np.argwhere(count_list >= 9)
            data = np.delete(data, timearray_index, axis=1)
            print('自动检测出时间序列，已删除', end=",")
            nan_flag = data.sum()
            assert np.isnan(nan_flag) == False
            # assert data.shape[1] == num_sig
            return data

        for class_name,class_dir in class_dir_dict.items():
            for root, dirs, files in os.walk(class_dir):
                for file in files:
                    if file.split(".")[-1] != "txt":
                        continue
                    file_path = root + "/" + file
                    signal = read_file_data(file_path)
                    # signal_class = root.split("\\")[1]
                    if class_name not in self.data_dict:
                        self.data_dict[class_name] = []
                    self.data_dict[class_name].append(signal)
                    print(file_path, "---->", root, signal.shape)
                    self.signal_num = signal.shape[1]
        print("end")


    def mat_plot_drow_axes(self,state_dict):
        """
        用清除画布刷新的方法绘图
        self.status_dict = {
            "decomposition": "pca",
            "feature": ["Response", "Sum", "Time"],
            "data_dir":"./data",
            "class_dict": {
            }
        }
        :return:
        """
        decomp_method = state_dict["decomposition"]
        feature_list  = state_dict["feature"]
        data_dict_show={}
        for class_name in state_dict["class_dict"].keys():
            data_dict_show[class_name] = self.data_dict[class_name]

        def get_all_feature(test_data):
            feature_sum = Feature_sum(test_data)
            feature_sensitive = Feature_sensitive(test_data)
            feature_time = Feature_time(test_data)
            feature_all = np.concatenate(
                [feature_sum, feature_sensitive, feature_time])  # feature_sum,feature_sensitive,feature_time

            return feature_all

        scalar_data = {}
        for name, data_list in data_dict_show.items():
            print(name, len(data_list))
            scalar_data[name] = [get_all_feature(single_data) for single_data in data_list]
        aaa=[]
        for tmp in list(scalar_data.values()):
            for tmp1 in tmp:
                aaa.append(tmp1)
        all_data_temp = np.array(aaa)
        all_data_temp = all_data_temp.reshape([-1, all_data_temp.shape[-1]])
        scaler = StandardScaler()
        all_data_temp = scaler.fit_transform(all_data_temp)

        for name, data_list in scalar_data.items():
            scalar_data[name] = scaler.transform(data_list)

        all_data_lable = []
        for name, data_list in data_dict_show.items():
            all_data_lable += [name] * len(data_list)
        # print(all_data_lable)

        def get_fetured_data(all_data_temp):
            feature_data_list = []
            for feature_name in feature_list:
                if feature_name=="Response":
                    feature_data_list.append(all_data_temp[:,:2 * self.signal_num])
                if feature_name=="Sum":
                    feature_data_list.append(all_data_temp[:, 2 * self.signal_num: 3 * self.signal_num])
                if feature_name == "Time":
                    feature_data_list.append(all_data_temp[:,3 * self.signal_num:])
            feature_data_list = np.concatenate(feature_data_list, axis=1)
            return feature_data_list

        self.axes.cla()  # 清除绘图区
        if decomp_method=="pca":
            pca = PCA(n_components=2)
            pca.fit(get_fetured_data(all_data_temp))
            num_lables = len(scalar_data.items())
            colors = list(plt.cm.tab10(np.arange(num_lables)))
            for i, (name, data_list) in enumerate(scalar_data.items()):
                new_sig = pca.transform(get_fetured_data(data_list))
                self.axes.scatter(new_sig[:, 0], new_sig[:, 1], color=colors[i], alpha=0.9, label=name,s=50)



        if decomp_method=="lda":
            lda=LinearDiscriminantAnalysis()
            lda.fit(get_fetured_data(all_data_temp),all_data_lable)
            num_lables = len(scalar_data.items())
            colors = list(plt.cm.tab10(np.arange(num_lables)))
            for i, (name, data_list) in enumerate(scalar_data.items()):
                new_sig = lda.transform(get_fetured_data(data_list))
                if new_sig.shape[1] != 1:
                    self.axes.scatter(new_sig[:, 0], new_sig[:, 1], color=colors[i], alpha=0.9, label=name)
                else:  # 二分类时没办法……
                    self.axes.scatter(new_sig[:, 0], np.random.random([new_sig.shape[0], 1]), color=colors[i], alpha=0.9,
                               label=name)



        self.axes.legend()

        self.axes.set_title("、".join(feature_list))
        # self.axes.plot(np.random.randint(0, 10, [5, 1]), 'o--')
        # self.axes.spines['top'].set_visible(False)  # 顶边界不可见
        # self.axes.spines['right'].set_visible(False)  # 右边界不可见
        # # 设置左、下边界在（0，0）处相交
        # # self.axes.spines['bottom'].set_position(('data', 0))  # 设置y轴线原点数据为 0
        # self.axes.spines['left'].set_position(('data', 0))  # 设置x轴线原点数据为 0
        # self.axes.plot(t, s, 'o-r', linewidth=0.5)
        self.figs.canvas.draw()  # 这里注意是画布重绘，self.figs.canvas
        self.figs.canvas.flush_events()  # 画布刷新self.figs.canvas


# 运行程序
if __name__ == '__main__':
    # QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    main_window = App()
    main_window.show()
    app.exec()