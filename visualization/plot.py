import sys

from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog,QMenu,
                                QVBoxLayout,QSizePolicy, QMessageBox, QWidget,
                                QPushButton, QCheckBox, QLabel, QInputDialog,
                                QSlider)
from PyQt5.QtGui import QIcon
from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
if is_pyqt5():
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure

import torch
import matplotlib.pyplot as plt
import numpy as np
import torch

from preprocessing.edf_loader import *
from montages import * #TODO: do something about this
import utils.pathmanager as pm # TODO: get rid of this
import utils.testconfiguration as tc # TODO: get rid of this
from utils.pipeline import Pipeline
# from utils.pipeline import aggregate_consecutive
from utils import pipeline
# TODO: get rid of read_file
from plot_utils import *

from preds_info import PredsInfo

import time

class MainPage(QMainWindow):

    def __init__(self):
        super().__init__()
        self.left = 10
        self.top = 10
        self.title = 'EEG Visualization'
        self.width = 1000
        self.height = 1000
        self.initUI()

    def initUI(self):
        """
        Setup the UI
        """
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.m = PlotCanvas(self, width=10, height=8)
        self.m.move(0,0)

        button = QPushButton('Select file', self)
        button.clicked.connect(self.load_data)
        button.setToolTip('Click to select EDF file')
        button.move(425,0)
        #button.resize(150,75)

        buttonRt1s = QPushButton("1>>",self)
        buttonRt1s.clicked.connect(self.rightPlot1s)
        buttonRt1s.setToolTip("Click to advance")
        buttonRt1s.move(550,0)
        #buttonRt.resize(50,50)

        buttonLt1s = QPushButton("<<1",self)
        buttonLt1s.clicked.connect(self.leftPlot1s)
        buttonLt1s.setToolTip("Click to go back")
        buttonLt1s.move(300,0)

        buttonRt10s = QPushButton("10>",self)
        buttonRt10s.clicked.connect(self.rightPlot10s)
        buttonRt10s.setToolTip("Click to advance")
        buttonRt10s.move(650,0)

        buttonLt10s = QPushButton("<10",self)
        buttonLt10s.clicked.connect(self.leftPlot10s)
        buttonLt10s.setToolTip("Click to go back")
        buttonLt10s.move(200,0)

        self.cbox_filter = QCheckBox("Filter",self)
        self.cbox_filter.toggled.connect(self.filterChecked)
        self.cbox_filter.setToolTip("Click to filter")
        self.cbox_filter.move(900,0)

        buttonPredict = QPushButton("Predict",self)
        buttonPredict.clicked.connect(self.predict)
        buttonPredict.setToolTip("Click to run data through model")
        buttonPredict.move(425,30)

        buttonLoadPtFile = QPushButton("Load data",self)
        buttonLoadPtFile.clicked.connect(self.loadPtData)
        buttonLoadPtFile.setToolTip("Click to load preprocessed data (as a torch tensor)")
        buttonLoadPtFile.move(300,30)

        buttonPredict = QPushButton("Load model",self)
        buttonPredict.clicked.connect(self.loadModel)
        buttonPredict.setToolTip("Click to load model")
        buttonPredict.move(200,30)
        #buttonPredict.move(375,30)
        #buttonPredict.resize(200,50)

        #labelAmp = QLabel("amplitude",self)
        #buttonPredict.clicked.connect(self.predict)
        #buttonPredict.setToolTip("Click to run data through model")
        #labelAmp.move(30,0)

        buttonChgCount = QPushButton("Jump to...",self)
        buttonChgCount.clicked.connect(self.getCount)
        buttonChgCount.setToolTip("Click to select time for graph")
        buttonChgCount.move(900,600)

        buttonAmpInc = QPushButton("+",self)
        buttonAmpInc.clicked.connect(self.incAmp)
        buttonAmpInc.setToolTip("Click to increase signal amplitude")
        buttonAmpInc.move(10,0)

        buttonAmpDec = QPushButton("-",self)
        buttonAmpDec.clicked.connect(self.decAmp)
        buttonAmpDec.setToolTip("Click to decrease signal amplitude")
        buttonAmpDec.move(10,30)

        buttonWSInc = QPushButton("+",self)
        buttonWSInc.clicked.connect(self.incWindow_size)
        buttonWSInc.setToolTip("Click to increase amount of seconds plotted")
        buttonWSInc.move(900,630)

        buttonWSDec = QPushButton("-",self)
        buttonWSDec.clicked.connect(self.decWindow_size)
        buttonWSDec.setToolTip("Click to decrease amount of seconds plotted")
        buttonWSDec.move(900,660)

        buttonPrint = QPushButton("Print",self)
        buttonPrint.clicked.connect(self.print_graph)
        buttonPrint.setToolTip("Click to print a copy of the graph")
        buttonPrint.move(850,30)

        self.slider = QSlider(Qt.Horizontal,self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(3000)
        self.slider.setValue(0)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(100)
        self.slider.move(100,760)
        self.slider.resize(800,20)
        #layout.addWidget(self.sl)
        self.slider.sliderReleased.connect(self.valuechange)


        self.count = 0 # the current location in time we are plotting
        self.init = 0 # if any data has been loaded in yet
        self.window_size = 10 # number of seconds to display at a time
        self.filter_checked = 0 # whether or not to plot filtered data
        self.ylim = [150,2] # ylim for unfiltered and filtered data
        self.predicted = 0 # whether or not predictions have been made

        # Labels for both types of montages
        self.labels = ["","CZ-PZ","FZ-CZ","P4-O2","C4-P4","F4-C4","FP2-F4",
                       "P3-O1","C3-P3","F3-C3","FP1-F3","P8-O2","T8-P8",
                       "F8-T8","FP2-F8","P7-O1","T7-P7","F7-T7","FP1-F7",""]
        self.labelsAR = ["","O2","O1","PZ","CZ","FZ","P8","P7","T8","T7","F8",
                        "F7","P4","P3","C4","C3","F4","F3","FP2","FP1"]
        self.show()

    def closeEvent(self, event):
        """
        Called when the main window is closed to act as a destructor and close
        any window that is still open.
        """
        print ("User has clicked the red x on the main window")
        # self.examp.change()
        event.accept()

    def initGraph(self):
        """
        Function to properly initialize everything when new data
        is loaded.
        """
        #self.init = 1 # set in load_data to prevent issues with slider
        self.filter_checked = 0
        self.cbox_filter.setChecked(False)
        self.ylim = [150,2] # reset scale of axis
        self.predicted = 0
        self.max_time = 0
        self.window_size = 10
        self.count = 0
        self.ann_list = [] # list of annotations
        self.aspan_list = [] # list of lines on the axis from preds
        self.pi = PredsInfo() # holds data needed to predict

    def valuechange(self):
        """
        Updates plot when slider is changed.
        """
        if self.init == 1:
            size = self.slider.value()
            self.count = size + 1
            if self.count + self.window_size > self.max_time + 1:
                self.count = self.max_time - self.window_size
            self.callmovePlot(0,1,0)


    def load_data(self):
        """
        Function to load in the data

        loads selected .edf file into edf_info and data
        data is initially unfiltered
        """
        name = QFileDialog.getOpenFileName(self, 'Open File')
        #name = []
        #name.append('JH01scalp1_0001.edf')
        #name.append('chb01_01.edf')
        name_len = len(name[0])
        if name[0] == None:
            return
        elif name[0][name_len-4:] != ".edf":
            alert = QMessageBox()
            alert.setText('Please select an .edf file')
            alert.setIcon(QMessageBox.Information)
            retval = alert.exec_()
        else:
            loader = EdfLoader()
            self.edf_info = loader.load_metadata(name[0])
            self.edf_info.annotations = np.array(self.edf_info.annotations)

            self.initGraph()

            self.data = loader.load_buffers(self.edf_info)
            self.data = np.array(self.data)
            if self.data.ndim == 1:
                data_temp = np.zeros((self.data.shape[0],self.data[0].shape[0]))
                for i in range(self.data.shape[0]):
                    data_temp[i,:] = self.data[i]
                self.data = data_temp

            self.data = np.array(self.data)
            fs = self.edf_info.fs
            self.max_time = int(self.data.shape[1] / fs)
            self.slider.setMaximum(self.max_time - self.window_size)

            edf_montages = EdfMontage(self.edf_info)
            self.montage = edf_montages.reorder_data(self.data)

            self.m.fig.clf()
            self.ax = self.m.fig.add_subplot(self.m.gs[0])

            self.movePlot(1,self.data.shape[1] / fs,self.ylim[0],0)
            self.init = 1

    def rightPlot1s(self):
        #self.examp = Example(self.edf_info)
        #self.examp.show()
        self.callmovePlot(1,1,0)

    def leftPlot1s(self):
        self.callmovePlot(0,1,0)

    def rightPlot10s(self):
        self.callmovePlot(1,10,0)

    def leftPlot10s(self):
        self.callmovePlot(0,10,0)

    def incAmp(self):
        if self.init == 1:
            if self.ylim[0] > 50:
                self.ylim[0] = self.ylim[0] - 15
                self.ylim[1] = self.ylim[1] - 0.2
                self.callmovePlot(0,0,0)

    def decAmp(self):
        if self.init == 1:
            if self.ylim[0] < 250:
                self.ylim[0] = self.ylim[0] + 15
                self.ylim[1] = self.ylim[1] + 0.2
                self.callmovePlot(0,0,0)

    def incWindow_size(self):
        if self.init == 1:
            if self.window_size + 5 <= 30:
                self.window_size = self.window_size + 5
                self.slider.setMaximum(self.max_time - self.window_size)
                if self.count + self.window_size > self.max_time:
                    self.count = self.max_time - self.window_size
                self.callmovePlot(0,0,0)

    def decWindow_size(self):
        if self.init == 1:
            if self.window_size - 5 >= 5:
                self.window_size = self.window_size - 5
                self.slider.setMaximum(self.max_time - self.window_size)
                self.callmovePlot(0,0,0)

    def getCount(self):
        if self.init == 1:
            num,ok = QInputDialog.getInt(self,"integer input","enter a number",
                                            0,0,self.max_time - self.window_size)
            if ok:
                self.count = num + 1
                self.callmovePlot(0,1,0)

    def print_graph(self):
        self.callmovePlot(0,0,1)

    def callmovePlot(self,right,num_move,print_graph):
        """
        Helper function to call movePlot for various buttons.
        """
        if self.init == 1:
            if self.filter_checked == 1:
                self.movePlot(right,num_move,self.ylim[1],print_graph)
            else:
                self.movePlot(right,num_move,self.ylim[0],print_graph)

    def movePlot(self, right, num_move, y_lim, print_graph):
        """
        Function to shift the plot left and right

        inputs:
            right -  0 for left, 1 for right
            num_move - integer in seconds to move by
            y_lim - the values for the y_limits of the plot

        """
        self.nchns = self.data.shape[0] # put this somewhere else
        # Clear plot
        # TODO: write a function for this
        del(self.ax.lines[0:self.nchns])
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

        if self.filter_checked == 1:
            self.prep_filter_ws()
            plotData = self.filteredData
        else:
            plotData = self.montage

        for i in range(self.montage.shape[0]):
            if self.montage.shape[0] == 18:
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


        #ax.figure.savefig("test.png",bbox_inches='tight')
        ann, idx_w_ann = checkAnnotations(self.count,self.window_size,self.edf_info)

        # Add in annotations
        if ann != []:
            txt = ""
            int_prev = int(float(ann[0,0]))
            for i in range(ann.shape[1]):
                int_i = int(float(ann[0,i]))
                if int_prev == int_i:
                    txt = txt + "\n" + ann[2,i]
                else:
                    if idx_w_ann[int_prev - self.count] and int_prev % 2 == 1:
                        self.ann_list.append(self.ax.annotate(txt, xy=((int_prev - self.count)*fs, -y_lim / 2 + y_lim),color='red'))
                    else:
                        self.ann_list.append(self.ax.annotate(txt, xy=((int_prev - self.count)*fs, -y_lim / 2),color='red'))
                    txt = ann[2,i]
                int_prev = int_i
            if txt != "":
                if idx_w_ann[int_i - self.count] and int_i % 2 == 1:
                    self.ann_list.append(self.ax.annotate(txt, xy=((int_i - self.count)*fs, -y_lim / 2 + y_lim),color='red'))
                else:
                    self.ann_list.append(self.ax.annotate(txt, xy=((int_i - self.count)*fs, -y_lim / 2),color='red'))

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
                self.filteredData = np.zeros(self.montage.shape)
                self.filter_checked = 1
                self.movePlot(1,0,self.ylim[1],0)
            else:
                self.filter_checked = 0
                self.movePlot(1,0,self.ylim[0],0)
        elif self.init == 0 and cbox.isChecked():
            cbox.setChecked(False)

    def prep_filter_ws(self):
        """
        Does filtering for one window of size window_size
        """
        fs = self.edf_info.fs
        filt_window_size = filterData(self.montage[:,self.count * fs:(self.count + self.window_size)*fs],fs,30,1.6,True)
        filt_window_size = np.array(filt_window_size)
        self.filteredData[:,self.count * fs:(self.count + self.window_size)*fs] = filt_window_size

    def loadPtData(self):
        """
        Load data for prediction
        """
        if self.init != 1:
            alert = QMessageBox()
            alert.setText('You must load data to predict')
            alert.setIcon(QMessageBox.Information)
            retval = alert.exec_()
        else:
            ptfile_fn = QFileDialog.getOpenFileName(self, 'Open torch file')
            ptfile_len = len(ptfile_fn[0])
            if ptfile_fn[0] == None:
                return
            elif ptfile_fn[0][ptfile_len-3:] != ".pt":
                alert = QMessageBox()
                alert.setText('Please select a .pt file')
                alert.setIcon(QMessageBox.Information)
                retval = alert.exec_()
            else:
                self.pi.set_data(ptfile_fn[0])

    def loadModel(self):
        """
        Load model for prediction
        """
        if self.init != 1:
            alert = QMessageBox()
            alert.setText('You must load data to predict')
            alert.setIcon(QMessageBox.Information)
            retval = alert.exec_()
        else:
            model_fn = QFileDialog.getOpenFileName(self, 'Open model')
            model_fn_len = len(model_fn[0])
            if model_fn[0] == None:
                return
            elif model_fn[0][model_fn_len-3:] != ".pt":
                alert = QMessageBox()
                alert.setText('Please select a .pt file')
                alert.setIcon(QMessageBox.Information)
                retval = alert.exec_()
            else:
                self.pi.set_model(model_fn[0])

    def predict(self):
        # TODO: replace this function with loadModel/loadPtData
        if self.init == 0:
            return
        if self.pi.ready:
            self.preds = predict(self.pi.data,self.pi.model)
            self.predicted = 1
            self.callmovePlot(0,0,0)
        elif not self.pi.data_loaded:
            alert = QMessageBox()
            alert.setText('Please load data')
            alert.setIcon(QMessageBox.Information)
            retval = alert.exec_()
        else:
            alert = QMessageBox()
            alert.setText('Please load a model')
            alert.setIcon(QMessageBox.Information)
            retval = alert.exec_()


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, constrained_layout=False)
        self.gs = self.fig.add_gridspec(1,1,wspace=0.0, hspace=0.0)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

    #def plot(self,data):
    #    ax = self.figure.add_subplot(111)
    #    ax.plot(data, 'r-')
    #    ax.set_title('PyQt Matplotlib Example')
    #    self.draw()

class Example(QWidget):
    def __init__(self,data):
        super().__init__()
        self.left = 10
        self.top = 10
        self.title = 'Test GUI2'
        self.width = 500
        self.height = 500
        data.fs = 2
        self.setupUI()
        #self.setupUi(self)

    def setupUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        button1 = QPushButton('Exit', self)
        button1.clicked.connect(self.change)
        #button1.setToolTip('Click to select EDF file')
        #button.move(425,0)
        #self.show()


    def change(self):
        # self.main = MainPage()
        # self.main.show()
        self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainPage()
    sys.exit(app.exec_())
