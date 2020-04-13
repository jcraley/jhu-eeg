import sys

from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog,QMenu,
                                QVBoxLayout,QSizePolicy, QMessageBox, QWidget,
                                QPushButton, QCheckBox, QLabel, QInputDialog,
                                QSlider, QGridLayout, QDockWidget, QListWidget)
from PyQt5.QtGui import QIcon
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.qt_compat import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure

import torch
import matplotlib.pyplot as plt
import numpy as np
import torch

from preprocessing.edf_loader import *
from montages import *
from plot_utils import *

from preds_info import PredsInfo
from filter_info import FilterInfo
from filter_options import FilterOptions

class MainPage(QMainWindow):

    def __init__(self):
        super().__init__()
        self.left = 10
        self.top = 10
        self.title = 'EEG Visualization'
        self.width = 1300
        self.height = 1000
        self.initUI()

    def initUI(self):
        """
        Setup the UI
        """
        #layout = QHBoxLayout()
        layout = QGridLayout()
        layout.setSpacing(10)
        grid_lt = QGridLayout()
        grid_rt = QGridLayout()
        grid_lt.setSpacing(12)
        grid_rt.setSpacing(8)


        button = QPushButton('Select file', self)
        button.clicked.connect(self.load_data)
        button.setToolTip('Click to select EDF file')
        grid_lt.addWidget(button, 0, 0,1,2)

        self.cbox_filter = QCheckBox("Filter signals",self)
        self.cbox_filter.toggled.connect(self.filterChecked)
        self.cbox_filter.setToolTip("Click to filter")
        grid_lt.addWidget(self.cbox_filter,1,0)

        buttonChgFilt = QPushButton("Change Filter",self)
        buttonChgFilt.clicked.connect(self.changeFilter)
        buttonChgFilt.setToolTip("Click to change filter")
        grid_lt.addWidget(buttonChgFilt,1,1)

        test0= QLabel("",self)
        grid_lt.addWidget(test0,2,0)

        buttonLoadPtFile = QPushButton("Load preprocessed data",self)
        buttonLoadPtFile.clicked.connect(self.loadPtData)
        buttonLoadPtFile.setToolTip("Click to load preprocessed data (as a torch tensor)")
        grid_lt.addWidget(buttonLoadPtFile,3,0)

        self.labelLoadPtFile = QLabel("No data loaded.",self)
        grid_lt.addWidget(self.labelLoadPtFile,3,1)

        buttonLoadModel = QPushButton("Load model",self)
        buttonLoadModel.clicked.connect(self.loadModel)
        buttonLoadModel.setToolTip("Click to load model")
        grid_lt.addWidget(buttonLoadModel,4,0)

        self.labelLoadModel= QLabel("No model loaded.",self)
        grid_lt.addWidget(self.labelLoadModel,4,1)

        buttonPredict = QPushButton("Predict",self)
        buttonPredict.clicked.connect(self.predict)
        buttonPredict.setToolTip("Click to run data through model")
        grid_lt.addWidget(buttonPredict,5,1,1,1)

        self.predLabel = QLabel("",self)
        grid_lt.addWidget(self.predLabel,5,0)

        test= QLabel("",self)
        grid_lt.addWidget(test,6,0)

        labelAmp = QLabel("Change amplitude:",self)
        grid_lt.addWidget(labelAmp,7,0)

        buttonAmpInc = QPushButton("+",self)
        buttonAmpInc.clicked.connect(self.incAmp)
        buttonAmpInc.setToolTip("Click to increase signal amplitude")
        grid_lt.addWidget(buttonAmpInc,7,1)

        buttonAmpDec = QPushButton("-",self)
        buttonAmpDec.clicked.connect(self.decAmp)
        buttonAmpDec.setToolTip("Click to decrease signal amplitude")
        grid_lt.addWidget(buttonAmpDec,8,1)

        labelWS = QLabel("Change window size:",self)
        grid_lt.addWidget(labelWS,9,0)

        buttonWSInc = QPushButton("+",self)
        buttonWSInc.clicked.connect(self.incWindow_size)
        buttonWSInc.setToolTip("Click to increase amount of seconds plotted")
        grid_lt.addWidget(buttonWSInc,9,1)

        buttonWSDec = QPushButton("-",self)
        buttonWSDec.clicked.connect(self.decWindow_size)
        buttonWSDec.setToolTip("Click to decrease amount of seconds plotted")
        grid_lt.addWidget(buttonWSDec,10,1)

        buttonPrint = QPushButton("Print",self)
        buttonPrint.clicked.connect(self.print_graph)
        buttonPrint.setToolTip("Click to print a copy of the graph")
        grid_lt.addWidget(buttonPrint,11,0)


        # Right side of the screen
        self.m = PlotCanvas(self, width=5, height=5)
        grid_rt.addWidget(self.m,0,0,6,8)

        self.slider = QSlider(Qt.Horizontal,self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(3000)
        self.slider.setValue(0)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(100)
        self.slider.sliderReleased.connect(self.valuechange)
        grid_rt.addWidget(self.slider, 6,0,1,8)

        self.buttonChgMont = QPushButton("Change montage",self)
        self.buttonChgMont.clicked.connect(self.chgMont)
        self.buttonChgMont.setToolTip("Click to change montage")
        self.buttonChgMont.hide()
        grid_rt.addWidget(self.buttonChgMont, 7,0)

        buttonLt10s = QPushButton("<10",self)
        buttonLt10s.clicked.connect(self.leftPlot10s)
        buttonLt10s.setToolTip("Click to go back")
        grid_rt.addWidget(buttonLt10s, 7,1)

        buttonLt1s = QPushButton("<<1",self)
        buttonLt1s.clicked.connect(self.leftPlot1s)
        buttonLt1s.setToolTip("Click to go back")
        grid_rt.addWidget(buttonLt1s,7,2)

        buttonChgCount = QPushButton("Jump to...",self)
        buttonChgCount.clicked.connect(self.getCount)
        buttonChgCount.setToolTip("Click to select time for graph")
        grid_rt.addWidget(buttonChgCount,7,3,1,2)

        buttonRt1s = QPushButton("1>>",self)
        buttonRt1s.clicked.connect(self.rightPlot1s)
        buttonRt1s.setToolTip("Click to advance")
        grid_rt.addWidget(buttonRt1s,7,5)

        buttonRt10s = QPushButton("10>",self)
        buttonRt10s.clicked.connect(self.rightPlot10s)
        buttonRt10s.setToolTip("Click to advance")
        grid_rt.addWidget(buttonRt10s, 7,6)

        self.time_lbl = QLabel("0:00:00",self)
        grid_rt.addWidget(self.time_lbl,7,7)

        # Annotation dock
        self.scroll = QDockWidget()
        self.ann_qlist = QListWidget()
        self.scroll.setWidget(self.ann_qlist)
        self.scroll.setFloating(False)
        self.addDockWidget(Qt.RightDockWidgetArea, self.scroll)
        self.scroll.hide()
        self.ann_qlist.itemClicked.connect(self.ann_clicked)


        wid = QWidget(self)
        self.setCentralWidget(wid)
        wid.setLayout(layout)
        layout.addLayout(grid_lt,0,0,3,1)
        layout.addLayout(grid_rt,0,1,4,3)

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.count = 0 # the current location in time we are plotting
        self.init = 0 # if any data has been loaded in yet
        self.window_size = 10 # number of seconds to display at a time
        self.filter_checked = 0 # whether or not to plot filtered data
        self.ylim = [150,3] # ylim for unfiltered and filtered data
        self.predicted = 0 # whether or not predictions have been made
        self.filter_win_open = 0

        # Labels for both types of montages
        self.labels = ["Notes","CZ-PZ","FZ-CZ","P4-O2","C4-P4","F4-C4","FP2-F4",
                       "P3-O1","C3-P3","F3-C3","FP1-F3","P8-O2","T8-P8",
                       "F8-T8","FP2-F8","P7-O1","T7-P7","F7-T7","FP1-F7",""]
        self.labelsAR = ["Notes","O2","O1","PZ","CZ","FZ","P8","P7","T8","T7","F8",
                        "F7","P4","P3","C4","C3","F4","F3","FP2","FP1"]

        self.fi = FilterInfo()

        self.show()

    def closeEvent(self, event):
        """
        Called when the main window is closed to act as a destructor and close
        any window that is still open.
        """

        if self.filter_win_open:
            self.filter_ops.closeWindow()
        event.accept()

    def initGraph(self):
        """
        Function to properly initialize everything when new data
        is loaded.
        """
        #self.init = 1 # set in load_data to prevent issues with slider
        self.filter_checked = 0 # whether or not filter checkbox is checked
        self.cbox_filter.setChecked(False)
        self.ylim = [150,3] # reset scale of axis
        self.predicted = 0 # whether or not predictions have been made
        self.max_time = 0 # number of seconds in the recording
        self.window_size = 10 # number of seconds displayed at once
        self.count = 0 # current location in time
        self.ann_list = [] # list of annotations
        self.aspan_list = [] # list of lines on the axis from preds
        self.pi = PredsInfo() # holds data needed to predict
        self.fi = FilterInfo() # holds data needed to filter
        self.ann_qlist.clear() # Clear annotations
        self.populateAnnDock() # Add annotations if they exist
        self.predLabel.setText("") # reset text of predictions
        self.labelLoadPtFile.setText("No data loaded.") # no data label reset
        self.labelLoadModel.setText("No model loaded.") # no data model reset
        self.buttonChgMont.hide()
        self.plot_bipolar = 0

    def ann_clicked(self,item):
        """
        Moves the plot when annotations in the dock are clicked.
        """
        self.count = int(float(self.edf_info.annotations[0][self.ann_qlist.currentRow()]))
        self.callmovePlot(0,0)

    def populateAnnDock(self):
        """
        Fills the annotation dock with annotations if they exist.
        """
        ann = self.edf_info.annotations
        if len(ann[0]) == 0:
            self.scroll.hide()
        else:
            for i in range(len(ann[0])):
                self.ann_qlist.addItem(ann[2][i])
            self.scroll.show()

    def valuechange(self):
        """
        Updates plot when slider is changed.
        """
        if self.init == 1:
            size = self.slider.value()
            self.count = size + 1
            if self.count + self.window_size > self.max_time + 1:
                self.count = self.max_time - self.window_size
            self.callmovePlot(0,1)

    def chgMont(self):
        """
        Funtion to change between bipolar and average reference.
        """
        self.plot_bipolar = not(self.plot_bipolar)
        self.callmovePlot(0,0)


    def load_data(self):
        """
        Function to load in the data

        loads selected .edf file into edf_info and data
        data is initially unfiltered
        """
        name = QFileDialog.getOpenFileName(self, 'Open File')

        name_len = len(name[0])
        if name[0] == None or name_len == 0:
            return
        elif name[0][name_len-4:] != ".edf":
            self.throwAlert('Please select an .edf file')
        else:
            loader = EdfLoader()
            try:
                self.edf_info = loader.load_metadata(name[0])
            except:
                self.throwAlert("The .edf file is invalid.")
                return
            self.edf_info.annotations = np.array(self.edf_info.annotations)

            self.initGraph()

            self.data = loader.load_buffers(self.edf_info)
            self.data = np.array(self.data)
            if self.data.ndim == 1:
                data_temp = np.zeros((self.data.shape[0],self.data[0].shape[0]))
                for i in range(self.data.shape[0]):
                    try:
                        if self.data[i].shape == self.data[0].shape:
                            data_temp[i,:] = self.data[i]
                    except:
                        pass
                self.data = data_temp

            self.data = np.array(self.data)
            fs = self.edf_info.fs
            try:
                fs = fs[0]
                self.edf_info.fs = fs
            except:
                pass
            self.fi.fs = fs
            self.max_time = int(self.data.shape[1] / fs)
            self.slider.setMaximum(self.max_time - self.window_size)

            edf_montages = EdfMontage(self.edf_info)
            self.montage = edf_montages.reorder_data(self.data)
            if self.montage.shape[0] == 19:
                self.montage_bipolar = edf_montages.get_bipolar_from_ar(self.montage)
                self.buttonChgMont.show()

            self.m.fig.clf()
            self.ax = self.m.fig.add_subplot(self.m.gs[0])

            self.movePlot(1,0,self.ylim[0],0,0)
            self.init = 1

    def rightPlot1s(self):
        self.callmovePlot(1,1)

    def leftPlot1s(self):
        self.callmovePlot(0,1)

    def rightPlot10s(self):
        self.callmovePlot(1,10)

    def leftPlot10s(self):
        self.callmovePlot(0,10)

    def incAmp(self):
        if self.init == 1:
            if self.ylim[0] > 50:
                self.ylim[0] = self.ylim[0] - 15
                self.ylim[1] = self.ylim[1] - 0.3
                self.callmovePlot(0,0)

    def decAmp(self):
        if self.init == 1:
            if self.ylim[0] < 250:
                self.ylim[0] = self.ylim[0] + 15
                self.ylim[1] = self.ylim[1] + 0.3
                self.callmovePlot(0,0)

    def incWindow_size(self):
        if self.init == 1:
            if self.window_size + 5 <= 30:
                self.window_size = self.window_size + 5
                self.slider.setMaximum(self.max_time - self.window_size)
                if self.count + self.window_size > self.max_time:
                    self.count = self.max_time - self.window_size
                self.callmovePlot(0,0)

    def decWindow_size(self):
        if self.init == 1:
            if self.window_size - 5 >= 5:
                self.window_size = self.window_size - 5
                self.slider.setMaximum(self.max_time - self.window_size)
                self.callmovePlot(0,0)

    def getCount(self):
        if self.init == 1:
            num,ok = QInputDialog.getInt(self,"integer input","enter a number",
                                            0,0,self.max_time - self.window_size)
            if ok:
                self.count = num + 1
                self.callmovePlot(0,1)

    def print_graph(self):
        self.callmovePlot(0,0,1)

    def callmovePlot(self,right,num_move,print_graph = 0):
        """
        Helper function to call movePlot for various buttons.
        """
        if self.init == 1:
            if self.filter_checked == 1:
                self.movePlot(right,num_move,self.ylim[1],print_graph,self.plot_bipolar)
            else:
                self.movePlot(right,num_move,self.ylim[0],print_graph,self.plot_bipolar)

    def movePlot(self, right, num_move, y_lim, print_graph, use_mont2):
        """
        Function to shift the plot left and right

        inputs:
            right -  0 for left, 1 for right
            num_move - integer in seconds to move by
            y_lim - the values for the y_limits of the plot

        """
        nchns = self.data.shape[0]
        # Clear plot
        del(self.ax.lines[0:nchns])
        for i, a in enumerate(self.ann_list):
            a.remove()
        self.ann_list[:] = []
        for aspan in self.aspan_list:
            aspan.remove()
        self.aspan_list[:] = []

        #self.m.fig.cla()

        fs = self.edf_info.fs

        # self.ax = self.m.fig.add_subplot(self.m.gs[0])

        if right == 0 and self.count - num_move >= 0:
            self.count = self.count - num_move
        elif right == 1 and self.count + num_move + self.window_size <= self.data.shape[1] / fs:
            self.count = self.count + num_move
        self.slider.setValue(self.count)
        t = getTime(self.count)
        self.time_lbl.setText(t)

        if self.filter_checked == 1:
            self.prep_filter_ws()
            plotData = self.filteredData
        elif use_mont2 == 0:
            plotData = self.montage
        else:
            plotData = self.montage_bipolar

        for i in range(plotData.shape[0]):
            if plotData.shape[0] == 18:
                if i < 2:
                    col = 'g'
                elif i < 6 or (i < 14 and i >= 10):
                    col = 'b'
                else:
                    col = 'r'
                self.ax.plot(plotData[i,self.count * fs:(self.count + 1) * fs*self.window_size] + i*y_lim + y_lim,'-',linewidth=0.5,color=col)
                self.ax.set_ylim([-y_lim, y_lim*19])
                self.ax.set_yticks(np.arange(0,20*y_lim,step=y_lim))
                self.ax.set_yticklabels(self.labels, fontdict=None, minor=False)
                if self.predicted == 1:
                    for k in range(self.window_size):
                        if self.windowed_preds[self.count + k]:
                            ax.axvspan(k * fs, (k + 1) * fs, color='paleturquoise', alpha=0.5)
            else:
                col = ['b','r','g','g','g','b','r','b','r','b','r','b','r','b',
                        'r','b','r','b','r','b']
                # average reference
                self.ax.plot(plotData[i,self.count * fs:(self.count + 1) * fs*self.window_size] + i*y_lim + y_lim,'-',linewidth=0.5,color=col[i])
                self.ax.set_ylim([-y_lim, y_lim*20])
                self.ax.set_yticks(np.arange(0,21*y_lim,step=y_lim))
                self.ax.set_yticklabels(self.labelsAR, fontdict=None, minor=False)

            if self.predicted == 1:
                for k in range(self.window_size):
                    if self.preds[self.count + k]:
                        # ax.axvspan(k * fs, (k + 1) * fs, ymin=0,ymax=0.5,color='paleturquoise', alpha=0.5)
                        self.aspan_list.append(self.ax.axvspan(k * fs, (k + 1) * fs,color='paleturquoise', alpha=0.5))

        self.ax.set_xlim([0,self.edf_info.fs*self.window_size])
        step_size = self.edf_info.fs # Updating the x labels with scaling
        step_width = 1
        if self.window_size >= 15 and self.window_size <= 25:
            step_size = step_size * 2
            step_width = step_width * 2
        elif self.window_size > 25:
            step_size = step_size * 3
            step_width = step_width * 3
        self.ax.set_xticks(np.arange(0, self.window_size*self.edf_info.fs + 1, step=step_size))
        self.ax.set_xticklabels(np.arange(self.count, self.count + self.window_size + 1, step=step_width), fontdict=None, minor=False)
        self.ax.set_xlabel("Time (s)")

        ann, idx_w_ann = checkAnnotations(self.count,self.window_size,self.edf_info)
        font_size = 10 - self.window_size / 5
        if font_size < 6:
            font_size = 6
        # Add in annotations
        if len(ann) != 0:
            ann = np.array(ann).T
            txt = ""
            int_prev = int(float(ann[0,0]))
            for i in range(ann.shape[1]):
                int_i = int(float(ann[0,i]))
                if int_prev == int_i:
                    txt = txt + "\n" + ann[2,i]
                else:
                    if idx_w_ann[int_prev - self.count] and int_prev % 2 == 1:
                        self.ann_list.append(self.ax.annotate(txt, xy=((int_prev - self.count)*fs, -y_lim / 2 + y_lim),color='black',size=font_size))
                    else:
                        self.ann_list.append(self.ax.annotate(txt, xy=((int_prev - self.count)*fs, -y_lim / 2),color='black',size=font_size))
                    txt = ann[2,i]
                int_prev = int_i
            if txt != "":
                if idx_w_ann[int_i - self.count] and int_i % 2 == 1:
                    self.ann_list.append(self.ax.annotate(txt, xy=((int_i - self.count)*fs, -y_lim / 2 + y_lim),color='black',size=font_size))
                else:
                    self.ann_list.append(self.ax.annotate(txt, xy=((int_i - self.count)*fs, -y_lim / 2),color='black',size=font_size))

        if print_graph == 1:
            file = QFileDialog.getSaveFileName(self, 'Save File')
            self.ax.figure.savefig(file[0] +".png",bbox_inches='tight')

        self.m.draw()

    def filterChecked(self):
        """
        Function for when the filterbox is checked

        sets self.filter_checked to the appropriate value, generates filteredData
        if needed and re-plots data with the correct type

        prevents the filter box from being able to be checked before data is loaded
        """
        cbox = self.sender()

        if self.init == 1:
            fs = self.edf_info.fs
            if cbox.isChecked():
                self.filter_checked = 1
            else:
                self.filter_checked = 0
            self.callmovePlot(1,0)
        elif self.init == 0 and cbox.isChecked():
            cbox.setChecked(False)

    def prep_filter_ws(self):
        """
        Does filtering for one window of size window_size
        """
        fs = self.edf_info.fs
        if self.plot_bipolar == 0:
            self.filteredData = np.zeros(self.montage.shape)
            filt_window_size = filterData(self.montage[:,self.count * fs:(self.count + self.window_size)*fs],fs,self.fi)
        else:
            self.filteredData = np.zeros(self.montage_bipolar.shape)
            filt_window_size = filterData(self.montage_bipolar[:,self.count * fs:(self.count + self.window_size)*fs],fs,self.fi)
        filt_window_size = np.array(filt_window_size)
        self.filteredData[:,self.count * fs:(self.count + self.window_size)*fs] = filt_window_size

    def changeFilter(self):
        if self.init == 1:
            self.filter_win_open = 1
            self.filter_ops = FilterOptions(self.fi,self)
            self.filter_ops.show()


    def loadPtData(self):
        """
        Load data for prediction
        """
        if self.init == 1:
            ptfile_fn = QFileDialog.getOpenFileName(self, 'Open torch file')
            ptfile_len = len(ptfile_fn[0])
            if ptfile_fn[0] == None or ptfile_len == 0:
                return
            elif ptfile_fn[0][ptfile_len-3:] != ".pt":
                self.throwAlert('Please select a .pt file')
            else:
                if len(ptfile_fn[0].split('/')[-1]) < 18:
                    self.labelLoadPtFile.setText(ptfile_fn[0].split('/')[-1])
                else:
                    self.labelLoadPtFile.setText(ptfile_fn[0].split('/')[-1][0:15] + "...")
                self.pi.set_data(ptfile_fn[0])
                self.predicted = 0
                self.predLabel.setText("")
                self.callmovePlot(0,0,0)

    def loadModel(self):
        """
        Load model for prediction
        """
        if self.init == 1:
            model_fn = QFileDialog.getOpenFileName(self, 'Open model')
            model_fn_len = len(model_fn[0])
            if model_fn[0] == None or model_fn_len == 0:
                return
            elif model_fn[0][model_fn_len-3:] != ".pt":
                self.throwAlert('Please select a .pt file')
            else:
                if len(model_fn[0].split('/')[-1]) < 18:
                    self.labelLoadModel.setText(model_fn[0].split('/')[-1])
                else:
                    self.labelLoadModel.setText(model_fn[0].split('/')[-1][0:15] + "...")
                self.pi.set_model(model_fn[0])
                self.predicted = 0
                self.predLabel.setText("")
                self.callmovePlot(0,0,0)

    def predict(self):
        """
        Take loaded model and data and compute predictions
        """
        if self.init == 0:
            return
        if self.pi.ready:
            self.preds = predict(self.pi.data,self.pi.model,self)
            if self.predicted == 1:
                if self.max_time != self.preds.shape[0]:
                    self.throwAlert('Predictions are not the same amount of seconds as the .edf file you loaded. Please check your file.')
                else:
                    self.predLabel.setText("Predictions plotted.")
                    self.callmovePlot(0,0,0)
        elif not self.pi.data_loaded:
            self.throwAlert('Please load data')
        else:
            self.throwAlert('Please load a model')

    def throwAlert(self, msg):
        alert = QMessageBox()
        alert.setText(msg)
        alert.setIcon(QMessageBox.Information)
        retval = alert.exec_()

class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, constrained_layout=False)
        self.gs = self.fig.add_gridspec(1,1,wspace=0.0, hspace=0.0)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainPage()
    sys.exit(app.exec_())
