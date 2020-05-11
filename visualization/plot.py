import sys

from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog,QMenu,
                                QVBoxLayout,QSizePolicy, QMessageBox, QWidget,
                                QPushButton, QCheckBox, QLabel, QInputDialog,
                                QSlider, QGridLayout, QDockWidget, QListWidget,
                                QStatusBar, QListWidgetItem)
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

from preprocessing.edf_loader import *
from montages import *
from plot_utils import *
import pyedflib

from preds_info import PredsInfo
from pred_options import PredictionOptions
from filter_info import FilterInfo
from filter_options import FilterOptions
from channel_options import ChannelOptions
from channel_info import ChannelInfo

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
        layout = QGridLayout()
        layout.setSpacing(10)
        grid_lt = QGridLayout()
        grid_rt = QGridLayout()
        grid_lt.setSpacing(12)
        grid_rt.setSpacing(8)

        # left side of the screen
        button = QPushButton('Select file', self)
        button.clicked.connect(self.load_data)
        button.setToolTip('Click to select EDF file')
        grid_lt.addWidget(button, 0, 0,1,2)

        self.buttonChgSig = QPushButton("Change signals",self)
        self.buttonChgSig.clicked.connect(self.chgSig)
        self.buttonChgSig.setToolTip("Click to change signals")
        grid_lt.addWidget(self.buttonChgSig, 1,1)

        self.cbox_filter = QCheckBox("Filter signals",self)
        self.cbox_filter.toggled.connect(self.filterChecked)
        self.cbox_filter.setToolTip("Click to filter")
        grid_lt.addWidget(self.cbox_filter,2,0)

        buttonChgFilt = QPushButton("Change Filter",self)
        buttonChgFilt.clicked.connect(self.changeFilter)
        buttonChgFilt.setToolTip("Click to change filter")
        grid_lt.addWidget(buttonChgFilt,2,1)

        test0= QLabel("",self)
        grid_lt.addWidget(test0,3,0)

        buttonPredict = QPushButton("Load model / predictions",self)
        buttonPredict.clicked.connect(self.changePredictions)
        buttonPredict.setToolTip("Click load data, models, and predictions")
        grid_lt.addWidget(buttonPredict,5,0,1,1)

        self.predLabel = QLabel("",self)
        grid_lt.addWidget(self.predLabel,5,1,1,1)

        threshLbl = QLabel("Change threshold of prediction:",self)
        grid_lt.addWidget(threshLbl,6,0)

        self.threshLblVal = QLabel("(threshold = 0.5)",self)
        grid_lt.addWidget(self.threshLblVal,6,1)

        self.threshSlider = QSlider(Qt.Horizontal,self)
        self.threshSlider.setMinimum(0)
        self.threshSlider.setMaximum(100)
        self.threshSlider.setValue(50)
        #self.threshSlider.setTickPosition(QSlider.TicksBelow)
        #self.threshSlider.setTickInterval(5)
        self.threshSlider.sliderReleased.connect(self.changeThreshSlider)
        grid_lt.addWidget(self.threshSlider, 7,0,1,2)

        test = QLabel("",self)
        grid_lt.addWidget(test,8,0)

        labelAmp = QLabel("Change amplitude:",self)
        grid_lt.addWidget(labelAmp,9,0)

        buttonAmpInc = QPushButton("+",self)
        buttonAmpInc.clicked.connect(self.incAmp)
        buttonAmpInc.setToolTip("Click to increase signal amplitude")
        grid_lt.addWidget(buttonAmpInc,9,1)

        buttonAmpDec = QPushButton("-",self)
        buttonAmpDec.clicked.connect(self.decAmp)
        buttonAmpDec.setToolTip("Click to decrease signal amplitude")
        grid_lt.addWidget(buttonAmpDec,10,1)

        labelWS = QLabel("Change window size:",self)
        grid_lt.addWidget(labelWS,11,0)

        buttonWSInc = QPushButton("+",self)
        buttonWSInc.clicked.connect(self.incWindow_size)
        buttonWSInc.setToolTip("Click to increase amount of seconds plotted")
        grid_lt.addWidget(buttonWSInc,11,1)

        buttonWSDec = QPushButton("-",self)
        buttonWSDec.clicked.connect(self.decWindow_size)
        buttonWSDec.setToolTip("Click to decrease amount of seconds plotted")
        grid_lt.addWidget(buttonWSDec,12,1)

        buttonPrint = QPushButton("Print",self)
        buttonPrint.clicked.connect(self.print_graph)
        buttonPrint.setToolTip("Click to print a copy of the graph")
        grid_lt.addWidget(buttonPrint,13,0)

        buttonSaveEDF = QPushButton("Save to .edf",self)
        buttonSaveEDF.clicked.connect(self.save_to_edf)
        buttonSaveEDF.setToolTip("Click to save current signals to an .edf file")
        grid_lt.addWidget(buttonSaveEDF,14,0)

        # Right side of the screen
        self.m = PlotCanvas(self, width=5, height=5)
        grid_rt.addWidget(self.m,0,0,6,8)

        self.slider = QSlider(Qt.Horizontal,self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(3000)
        self.slider.setValue(0)
        # self.slider.setTickPosition(QSlider.TicksBelow)
        # self.slider.setTickInterval(100)
        self.slider.sliderReleased.connect(self.valuechange)
        grid_rt.addWidget(self.slider, 6,0,1,8)

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
        ann_title = QLabel("Annotations")
        self.scroll.setTitleBarWidget(ann_title)
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
        self.ylim = [150,100] # ylim for unfiltered and filtered data
        self.predicted = 0 # whether or not predictions have been made
        self.filter_win_open = 0 # whether or not filter options window is open
        self.preds_win_open = 0 # whether or not the predictions window is open
        self.chn_win_open = 0 # whether or not the channel selection window is open
        self.max_time = 0 # number of seconds in the recording
        self.pi = PredsInfo() # holds data needed to predict
        self.ci = ChannelInfo() # holds channel information

        self.show()

    def closeEvent(self, event):
        """
        Called when the main window is closed to act as a destructor and close
        any window that is still open.
        """
        if self.filter_win_open:
            self.filter_ops.closeWindow()
        if self.preds_win_open:
            self.pred_ops.closeWindow()
        if self.chn_win_open:
            self.chn_ops.closeWindow()
        event.accept()

    def initGraph(self):
        """
        Function to properly initialize everything when new data
        is loaded.
        """
        #self.init = 1 # set in load_data to prevent issues with slider
        self.fi = FilterInfo() # holds data needed to filter
        self.filter_checked = 0 # whether or not filter checkbox is checked
        self.cbox_filter.setChecked(False)

        # check if this file is already filtered
        ann = self.edf_info.annotations
        if len(ann[0]) > 0 and ann[2][0] == "filtered":
            self.filter_checked = 1 # whether or not filter checkbox is checked
            strFilt = ann[2][1].split("Hz")
            strLP = strFilt[0][4:]
            strHP = strFilt[1][5:]
            strN = strFilt[2][4:]
            if int(strLP):
                self.fi.lp = int(strLP)
            else:
                self.fi.do_lp = 0
            if int(strHP):
                self.fi.hp = int(strHP)
            else:
                self.fi.do_hp = 0
            if int(strN):
                self.fi.notch = int(strN)
            else:
                self.fi.do_notch = 0

        self.ylim = [150, 100]# [150,3] # reset scale of axis
        self.window_size = 10 # number of seconds displayed at once
        self.count = 0 # current location in time
        self.ann_list = [] # list of annotations
        self.aspan_list = [] # list of lines on the axis from preds
        self.predLabel.setText("") # reset text of predictions
        self.thresh = 0.5 # threshold for plotting
        self.threshLblVal.setText("(threshold = " + str(self.thresh) + ")") # reset label
        self.filteredData = [] # set filteredData

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

    def changeThreshSlider(self):
        """
        Updates the value of the threshold when the slider is changed.
        """
        val = self.threshSlider.value()
        self.thresh = val / 100
        self.threshLblVal.setText("(threshold = " + str(self.thresh) + ")")
        if self.predicted == 1:
            self.callmovePlot(0,0)

    def chgSig(self):
        """
        Funtion to open channel_options so users can change the signals being
        plotted.
        """
        if self.init and not self.chn_win_open:
            self.chn_win_open = 1
            self.chn_ops = ChannelOptions(self.ci,self)
            self.chn_ops.show()

    def save_to_edf(self):
        """
        Function to save current data to .edf file
        """
        if self.init == 1:
            if self.filter_checked == 1:
                dataToSave = filterData(self.ci.data_to_plot, self.edf_info.fs, self.fi)
                if self.fi.filter_canceled == 1:
                    self.fi.filter_canceled = 0
                    return
            else:
                dataToSave = self.ci.data_to_plot
            file = QFileDialog.getSaveFileName(self, 'Save File')
            nchns = self.ci.nchns_to_plot
            labels = self.ci.labels_to_plot

            # if predictions, save them as well
            """if (self.predicted == 1 and len(self.pi.preds_to_plot.shape) > 1
                    and self.pi.preds_to_plot.shape[1] != nchns):
                self.predicted = 0"""

            if self.predicted == 1:
                if self.pi.pred_by_chn:
                    savedEDF = pyedflib.EdfWriter(file[0] + '.edf', nchns * 2)
                else:
                    savedEDF = pyedflib.EdfWriter(file[0] + '.edf', nchns + 1)
            else:
                savedEDF = pyedflib.EdfWriter(file[0] + '.edf', nchns)

            # Set fs and physical min/max
            fs = self.edf_info.fs
            for i in range(nchns):
                savedEDF.setPhysicalMaximum(i, np.max(dataToSave[i]))
                savedEDF.setPhysicalMinimum(i, np.min(dataToSave[i]))
                savedEDF.setSamplefrequency(i, fs)
                savedEDF.setLabel(i, labels[i + 1])
            # if predictions, save them as well
            if self.predicted == 1:
                temp = []
                for i in range(nchns):
                    temp.append(dataToSave[i])
                if self.pi.pred_by_chn:
                    for i in range(nchns):
                        savedEDF.setPhysicalMaximum(nchns + i, 1)
                        savedEDF.setPhysicalMinimum(nchns + i, 0)
                        savedEDF.setSamplefrequency(nchns + i, fs / self.pi.pred_width)
                        savedEDF.setLabel(nchns + i, "PREDICTIONS_" + str(i))
                    for i in range(nchns):
                        temp.append(self.pi.preds_to_plot[:,i])
                else:
                    savedEDF.setPhysicalMaximum(nchns, 1)
                    savedEDF.setPhysicalMinimum(nchns, 0)
                    savedEDF.setSamplefrequency(nchns, fs / self.pi.pred_width)
                    savedEDF.setLabel(nchns, "PREDICTIONS")
                    temp.append(self.pi.preds_to_plot)
                dataToSave = temp

            savedEDF.writeSamples(dataToSave)

            # write annotations
            ann = self.edf_info.annotations
            if len(ann[0]) > 0 and ann[2][0] == "filtered":
                ann = np.delete(ann, 0, axis = 1)
                ann = np.delete(ann, 0, axis = 1)
            if self.filter_checked == 1:
                if len(ann[0]) == 0:
                    ann = np.array([0.0,-1.0,"filtered"])
                    ann = ann[...,np.newaxis]
                else:
                    ann = np.insert(ann, 0,[0.0,-1.0,"filtered"], axis=1)
                strFilt = ""
                strFilt += "LP: " + str(self.fi.do_lp * self.fi.lp) + "Hz"
                strFilt += " HP: " + str(self.fi.do_hp * self.fi.hp) + "Hz"
                strFilt += " N: " + str(self.fi.do_notch * self.fi.hp) + "Hz"
                ann = np.insert(ann, 1,[0.0,-1.0,strFilt], axis=1)
            for i in range(len(ann[0])):
                savedEDF.writeAnnotation(float(ann[0][i]), float((ann[1][i])), ann[2][i])

            savedEDF.close()

    def load_data(self):
        """
        Function to load in the data

        loads selected .edf file into edf_info and data
        data is initially unfiltered
        """
        name = QFileDialog.getOpenFileName(self, 'Open file','.','EDF files (*.edf)')

        if name[0] == None or len(name[0]) == 0:
            return
        else:
            loader = EdfLoader()
            try:
                self.edf_info_temp = loader.load_metadata(name[0])
            except:
                self.throwAlert("The .edf file is invalid.")
                return
            self.edf_info_temp.annotations = np.array(self.edf_info_temp.annotations)

            self.data_temp = loader.load_buffers(self.edf_info_temp)
            data_for_preds = self.data_temp
            self.data_temp = np.array(self.data_temp)
            if self.data_temp.ndim == 1:
                data_temp = np.zeros((self.data_temp.shape[0],self.data_temp[0].shape[0]))
                for i in range(self.data_temp.shape[0]):
                    try:
                        if self.data_temp[i].shape == self.data_temp[0].shape:
                            data_temp[i,:] = self.data_temp[i]
                    except:
                        pass
                self.data_temp = data_temp

            self.data_temp = np.array(self.data_temp)
            edf_montages = EdfMontage(self.edf_info_temp)
            fs_idx = edf_montages.getIndexForFs(self.edf_info_temp.labels2chns)
            fs = self.edf_info_temp.fs
            try:
                fs = fs[fs_idx]
                self.edf_info_temp.fs = fs
            except:
                pass

            # setting temporary variables that will be overwritten if
            # the user selects signals to plot
            self.max_time_temp = int(self.data_temp.shape[1] / fs)
            self.ci_temp = ChannelInfo() # holds channel information
            self.ci_temp.chns2labels = self.edf_info_temp.chns2labels
            self.ci_temp.labels2chns = self.edf_info_temp.labels2chns
            self.ci_temp.fs = self.edf_info_temp.fs
            self.ci_temp.max_time = self.max_time_temp

            self.chn_win_open = 1
            self.chn_ops = ChannelOptions(self.ci_temp,self,data_for_preds)
            self.chn_ops.show()

    def callInitialMovePlot(self):
        """
        Function called by channel_options when channels are loaded
        """
        self.initGraph()

        self.fi.fs = self.ci.fs
        self.slider.setMaximum(self.max_time - self.window_size)
        self.threshSlider.setValue(self.thresh * 100)

        self.ann_qlist.clear() # Clear annotations
        self.populateAnnDock() # Add annotations if they exist

        self.m.fig.clf()
        self.ax = self.m.fig.add_subplot(self.m.gs[0])

        if self.filter_checked == 1:
            self.movePlot(0,0,self.ylim[1],0)
        else:
            self.movePlot(0,0,self.ylim[0],0)
        self.callmovePlot(1,0)
        self.init = 1
        ann = self.edf_info.annotations
        if len(ann[0]) > 0 and ann[2][0] == "filtered":
             self.cbox_filter.setChecked(True) # must be set after init = 1

        if self.predicted == 1:
            self.pi.plot_preds_preds = 1
            self.pi.preds_loaded = 1
            self.pi.preds_fn = "loaded from edf file"

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
                self.ylim[1] = self.ylim[1] - 10
                self.callmovePlot(0,0)

    def decAmp(self):
        if self.init == 1:
            if self.ylim[0] < 250:
                self.ylim[0] = self.ylim[0] + 15
                self.ylim[1] = self.ylim[1] + 10
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
        """
        Used for the "jump to" button to update self.count to the user's input
        """
        if self.init == 1:
            num,ok = QInputDialog.getInt(self,"integer input","enter a number",
                                            0,0,self.max_time)
            if ok:
                if num > self.max_time - self.window_size:
                    num = self.max_time - self.window_size
                self.count = num
                self.callmovePlot(0,0)

    def print_graph(self):
        self.callmovePlot(0,0,1)

    def callmovePlot(self,right,num_move,print_graph = 0):
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
            print_graph - whether or not to print a copy of the graph
        """

        fs = self.edf_info.fs

        if right == 0 and self.count - num_move >= 0:
            self.count = self.count - num_move
        elif right == 1 and self.count + num_move + self.window_size <= self.data.shape[1] / fs:
            self.count = self.count + num_move
        self.slider.setValue(self.count)
        t = getTime(self.count)
        self.time_lbl.setText(t)

        if self.filter_checked == 1:
            self.prep_filter_ws()
            plotData = np.zeros(self.filteredData.shape)
            plotData += self.filteredData
            stddev = np.std(plotData[:,self.count * fs:(self.count + 10) * fs])
            plotData[plotData > 3 * stddev] = 3 * stddev # float('nan') # clip amplitude
            plotData[plotData < -3 * stddev] = -3 * stddev
        else:
            plotData = np.zeros(self.ci.data_to_plot.shape)
            plotData += self.ci.data_to_plot

        nchns = self.ci.nchns_to_plot
        if self.predicted == 1:
            """if len(self.pi.preds_to_plot.shape) > 1 and self.pi.preds_to_plot.shape[1] != nchns:
                self.predLabel.setText("")
            else:"""
            self.predLabel.setText("Predictions plotted.")
        else:
            self.predLabel.setText("")
        # Clear plot
        del(self.ax.lines[:])
        for i, a in enumerate(self.ann_list):
            a.remove()
        self.ann_list[:] = []
        for aspan in self.aspan_list:
            aspan.remove()
        self.aspan_list[:] = []

        for i in range(nchns):
            self.ax.plot(plotData[i,self.count * fs:(self.count + 1) * fs*self.window_size]
                            + (i + 1) * y_lim,'-',linewidth=0.5,color=self.ci.colors[i])
            self.ax.set_ylim([-y_lim, y_lim * (nchns + 1)])
            self.ax.set_yticks(np.arange(0,(nchns + 2)*y_lim,step=y_lim))
            self.ax.set_yticklabels(self.ci.labels_to_plot, fontdict=None, minor=False)

            width = 1 / (nchns + 2)
            if self.predicted == 1:
                starts, ends, chns = self.pi.compute_starts_ends_chns(self.thresh, self.count, self.window_size, fs, nchns)
                for k in range(len(starts)):
                    if self.pi.pred_by_chn:
                        if chns[k][i]:
                            if i == plotData.shape[0] - 1:
                                self.aspan_list.append(self.ax.axvspan(starts[k] - self.count * fs,ends[k] - self.count * fs,
                                                        ymin=width*(i+1.5),ymax=1,color='paleturquoise', alpha=1))
                            else:
                                self.aspan_list.append(self.ax.axvspan(starts[k] - self.count * fs,ends[k] - self.count * fs,
                                                        ymin=width*(i+1.5),ymax=width*(i+2.5),color='paleturquoise', alpha=1))
                            x_vals = range(int(starts[k]) - self.count * fs,int(ends[k]) - self.count * fs)
                            self.ax.plot(x_vals,plotData[i,int(starts[k]):int(ends[k])] + i*y_lim + y_lim,
                                            '-',linewidth=1,color=self.ci.colors[i])
                    else:
                        self.aspan_list.append(self.ax.axvspan(starts[k] - self.count * fs,ends[k] - self.count * fs,color='paleturquoise', alpha=0.5))

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
        if font_size < 7:
            font_size = 7
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
            # if data was already filtered do not uncheck box
            ann = self.edf_info.annotations
            if len(ann[0]) > 0 and ann[2][0] == "filtered":
                self.filter_checked = 1
                cbox.setChecked(True)
            self.callmovePlot(1,0)
        elif self.init == 0 and cbox.isChecked():
            cbox.setChecked(False)

    def prep_filter_ws(self):
        """
        Does filtering for one window of size window_size
        """
        fs = self.edf_info.fs
        if len(self.filteredData) == 0:
            self.filteredData = np.zeros(self.ci.data_to_plot.shape)
        elif self.filteredData.shape != self.ci.data_to_plot.shape:
            self.filteredData = np.zeros(self.ci.data_to_plot.shape)
        filt_window_size = filterData(self.ci.data_to_plot[:,self.count * fs:(self.count + self.window_size)*fs],fs,self.fi)
        filt_window_size = np.array(filt_window_size)
        self.filteredData[:,self.count * fs:(self.count + self.window_size)*fs] = filt_window_size

    def changeFilter(self):
        """
        Opens the FilterOptions window
        """
        if self.init == 1:
            self.filter_win_open = 1
            self.filter_ops = FilterOptions(self.fi,self)
            self.filter_ops.show()

    def changePredictions(self):
        """
        Take loaded model and data and compute predictions
        """
        if self.init == 1:
            self.preds_win_open = 1
            self.pred_ops = PredictionOptions(self.pi,self)
            self.pred_ops.show()

    def throwAlert(self, msg):
        """
        Throws an alert to the user.
        """
        alert = QMessageBox()
        alert.setIcon(QMessageBox.Information)
        alert.setText(msg)
        # alert.setInformativeText(msg)
        alert.setWindowTitle("Warning")
        alert.exec_()

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
