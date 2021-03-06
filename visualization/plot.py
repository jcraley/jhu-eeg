from channel_info import ChannelInfo
from channel_options import ChannelOptions
from filter_options import FilterOptions
from filter_info import FilterInfo
from pred_options import PredictionOptions
from preds_info import PredsInfo
from spec_options import SpecOptions
from spec_info import SpecInfo
from saveImg_info import SaveImgInfo
from saveImg_options import SaveImgOptions
from saveEdf_info import SaveEdfInfo
from saveEdf_options import SaveEdfOptions

import pyedflib
from plot_utils import *
from montages import *
from anonymize_edf import anonymizeFile
from preprocessing.edf_loader import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.backends.qt_compat import QtCore, QtWidgets
import sys

from PyQt5.QtCore import Qt, QTime, QUrl

from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QMenu,
                             QVBoxLayout, QSizePolicy, QMessageBox, QWidget,
                             QPushButton, QCheckBox, QLabel, QInputDialog,
                             QSlider, QGridLayout, QDockWidget, QListWidget,
                             QStatusBar, QListWidgetItem, QLineEdit, QSpinBox,
                             QTimeEdit, QComboBox, QFrame, QGroupBox, QStyle)
from PyQt5.QtGui import QIcon, QBrush, QColor, QPen, QFont, QDesktopServices
import pyqtgraph as pg
import pyqtgraph.exporters
# pg.setConfigOptions(useOpenGL=True) # To make plotting faster when line width > 1

import matplotlib
matplotlib.use("Qt5Agg")
from scipy import signal

import argparse as ap
import os.path
from os import path
import cProfile as profile


class MainPage(QMainWindow):

    def __init__(self, argv, app):
        super().__init__()
        self.argv = argv
        self.left = 10
        self.top = 10
        self.title = 'EEG Visualization'
        sizeObject = QtWidgets.QDesktopWidget().screenGeometry(-1)
        # print(" Screen size : "  + str(sizeObject.height()) + "x"  + str(sizeObject.width()))
        self.width = sizeObject.width() * 0.9
        self.height = sizeObject.height() * 0.9
        self.app = app
        self.initUI()

    def initUI(self):
        """
        Setup the UI
        """
        layout = QGridLayout()
        layout.setSpacing(10)
        grid_lt = QGridLayout()
        self.grid_rt = QGridLayout()
        #grid_lt.setSpacing(12)
        #self.grid_rt.setSpacing(8)

        # left side of the screen

        groupBox = QGroupBox()


        # radio1 = QRadioButton("&Radio button 1")
        # radio2 = QRadioButton("R&adio button 2")
        # radio3 = QRadioButton("Ra&dio button 3")

        button = QPushButton('Select file', self)
        button.clicked.connect(self.load_data)
        button.setToolTip('Click to select EDF file')
        grid_lt.addWidget(button, 0, 0, 1, 2)

        self.lblFn = QLabel("No file loaded.",self)
        grid_lt.addWidget(self.lblFn, 1, 0, 1, 2)

        self.buttonChgSig = QPushButton("Change signals", self)
        self.buttonChgSig.clicked.connect(self.chgSig)
        self.buttonChgSig.setToolTip("Click to change signals")
        grid_lt.addWidget(self.buttonChgSig, 2, 1)

        self.cbox_filter = QCheckBox("Filter signals", self)
        self.cbox_filter.toggled.connect(self.filterChecked)
        self.cbox_filter.setToolTip("Click to filter")
        grid_lt.addWidget(self.cbox_filter, 3, 0)

        buttonChgFilt = QPushButton("Change Filter", self)
        buttonChgFilt.clicked.connect(self.changeFilter)
        buttonChgFilt.setToolTip("Click to change filter")
        grid_lt.addWidget(buttonChgFilt, 3, 1)

        vbox = QGridLayout()
        vbox.addWidget(button,0,0,1,2)
        vbox.addWidget(self.lblFn, 1, 0, 1, 2)
        vbox.addWidget(self.buttonChgSig, 2, 1)
        vbox.addWidget(self.cbox_filter, 3, 0)
        vbox.addWidget(buttonChgFilt, 3, 1)
        # vbox.addStretch(1)
        groupBox.setLayout(vbox)
        # grid_lt.addWidget(groupBox,0,0,3,2)

        test01 = QLabel("", self)
        grid_lt.addWidget(test01, 4, 0)

        grid_lt.addWidget(QHLine(), 5, 0, 1, 2)

        test0 = QLabel("", self)
        grid_lt.addWidget(test0, 6, 0)

        groupBox2 = QGroupBox()

        buttonPredict = QPushButton("Load model / predictions", self)
        buttonPredict.clicked.connect(self.changePredictions)
        buttonPredict.setToolTip("Click load data, models, and predictions")
        grid_lt.addWidget(buttonPredict, 7, 0, 1, 2)

        self.predLabel = QLabel("", self)
        grid_lt.addWidget(self.predLabel, 8, 0, 1, 1)

        #test22 = QLabel("", self)
        #grid_lt.addWidget(test22, 8, 1)

        self.threshLblVal = QLabel("Change threshold of prediction:  (threshold = 0.5)", self)
        grid_lt.addWidget(self.threshLblVal, 9, 0,1,2)

        # self.threshLblVal = QLabel("(threshold = 0.5)", self)
        # grid_lt.addWidget(self.threshLblVal, 7, 1)

        self.threshSlider = QSlider(Qt.Horizontal, self)
        self.threshSlider.setMinimum(0)
        self.threshSlider.setMaximum(100)
        self.threshSlider.setValue(50)
        # self.threshSlider.setTickPosition(QSlider.TicksBelow)
        # self.threshSlider.setTickInterval(5)
        self.threshSlider.sliderReleased.connect(self.changeThreshSlider)
        grid_lt.addWidget(self.threshSlider, 10, 0, 1, 2)

        vbox2 = QGridLayout()
        vbox2.addWidget(buttonPredict, 0, 0, 1, 2)
        vbox2.addWidget(self.predLabel, 0, 1)
        # vbox2.addWidget(threshLbl, 1, 0)
        vbox2.addWidget(self.threshLblVal, 1, 0, 1, 2)
        vbox2.addWidget(self.threshSlider, 2, 0, 1, 2)
        # vbox.addStretch(1)
        groupBox2.setLayout(vbox2)
        # grid_lt.addWidget(groupBox2,6,0,3,2)

        test = QLabel("", self)
        grid_lt.addWidget(test, 11, 0)

        grid_lt.addWidget(QHLine(), 12, 0, 1, 2)

        test11 = QLabel("", self)
        grid_lt.addWidget(test11, 13, 0)

        groupBox3 = QGroupBox()

        self.btnZoom = QPushButton("Open zoom", self)
        self.btnZoom.clicked.connect(self.openZoomPlot)
        self.btnZoom.setToolTip("Click to open the zoom window")
        grid_lt.addWidget(self.btnZoom, 14, 0)

        buttonChgSpec = QPushButton("Power spectrum", self)
        buttonChgSpec.clicked.connect(self.loadSpec)
        buttonChgSpec.setToolTip("Click to plot the spectrogram of a signal")
        grid_lt.addWidget(buttonChgSpec, 14, 1)

        labelAmp = QLabel("Change amplitude:", self)
        grid_lt.addWidget(labelAmp, 15, 0)

        buttonAmpInc = QPushButton("+", self)
        buttonAmpInc.clicked.connect(self.incAmp)
        buttonAmpInc.setToolTip("Click to increase signal amplitude")
        grid_lt.addWidget(buttonAmpInc, 15, 1)

        buttonAmpDec = QPushButton("-", self)
        buttonAmpDec.clicked.connect(self.decAmp)
        buttonAmpDec.setToolTip("Click to decrease signal amplitude")
        grid_lt.addWidget(buttonAmpDec, 16, 1)

        labelWS = QLabel("Window size:", self)
        grid_lt.addWidget(labelWS, 17, 0)

        self.wsComboBox = QComboBox()
        self.wsComboBox.addItems(["1s","5s","10s","15s","20s","25s","30s"])
        self.wsComboBox.setCurrentIndex(2)
        self.wsComboBox.currentIndexChanged['int'].connect(self.chgWindow_size)
        grid_lt.addWidget(self.wsComboBox, 17, 1)

        vbox3 = QGridLayout()
        vbox3.addWidget(self.btnZoom, 0, 0)
        vbox3.addWidget(buttonChgSpec, 0, 1)
        vbox3.addWidget(QLabel("", self),1,0)
        vbox3.addWidget(labelAmp, 2, 0)
        vbox3.addWidget(buttonAmpInc, 2, 1)
        vbox3.addWidget(buttonAmpDec, 3, 1)
        vbox3.addWidget(labelWS, 4, 0)
        vbox3.addWidget(self.wsComboBox, 4, 1)
        # vbox.addStretch(1)
        groupBox3.setLayout(vbox3)
        # grid_lt.addWidget(groupBox3,10,0,4,2)

        test2 = QLabel("", self)
        grid_lt.addWidget(test2, 19, 0)

        grid_lt.addWidget(QHLine(), 20, 0, 1, 2)

        test11 = QLabel("", self)
        grid_lt.addWidget(test11, 21, 0)

        """buttonWSInc = QPushButton("+", self)
        buttonWSInc.clicked.connect(self.incWindow_size)
        buttonWSInc.setToolTip("Click to increase amount of seconds plotted")
        grid_lt.addWidget(buttonWSInc, 13, 1)"""

        """buttonWSDec = QPushButton("-", self)
        buttonWSDec.clicked.connect(self.decWindow_size)
        buttonWSDec.setToolTip("Click to decrease amount of seconds plotted")
        grid_lt.addWidget(buttonWSDec, 14, 1)"""

        groupBox4 = QGroupBox()

        buttonPrint = QPushButton("Export to .png", self)
        buttonPrint.clicked.connect(self.print_graph)
        buttonPrint.setToolTip("Click to print a copy of the graph")
        grid_lt.addWidget(buttonPrint, 22, 0)

        buttonSaveEDF = QPushButton("Save to .edf", self)
        buttonSaveEDF.clicked.connect(self.save_to_edf)
        buttonSaveEDF.setToolTip(
            "Click to save current signals to an .edf file")
        grid_lt.addWidget(buttonSaveEDF, 22, 1)

        """
        popupButton = QPushButton("Export")
        menu = QMenu(self)
        menu.addAction("...to .png")
        menu.addAction("...to .edf")
        popupButton.setMenu(menu)
        grid_lt.addWidget(popupButton,15,1)"""

        vbox4 = QGridLayout()
        vbox4.addWidget(buttonPrint, 0, 0)
        vbox4.addWidget(buttonSaveEDF, 0, 1)
        # vbox.addStretch(1)
        groupBox4.setLayout(vbox4)
        # grid_lt.addWidget(groupBox4,16,0,2,2)

        test3 = QLabel("", self)
        grid_lt.addWidget(test3, 23, 0)

        test4 = QLabel("", self)
        grid_lt.addWidget(test4, 24, 0)

        test5 = QLabel("", self)
        grid_lt.addWidget(test5, 25, 0)

        test6 = QLabel("", self)
        grid_lt.addWidget(test6, 26, 0)

        btn0 = QPushButton('Help')
        btn0.setIcon(self.style().standardIcon(getattr(QStyle, 'SP_TitleBarContextHelpButton')))
        btn0.clicked.connect(self.openHelp)
        grid_lt.addWidget(btn0,27,0)
        # icon  = QtGui.QPixmap('add.gif')
        # button = QtGui.QPushButton()
        # button.setIcon(icon)


        # Right side of the screen
        # self.m = PlotCanvas(self, width=5, height=5)
        # self.grid_rt.addWidget(self.m, 0, 0, 6, 8)
        self.plotLayout = pg.GraphicsLayoutWidget()
        self.mainPlot = self.plotLayout.addPlot(row=0, col=0)
        self.mainPlot.setMouseEnabled(x=False, y=False)
        self.plotLayout.setBackground('w')
        self.grid_rt.addWidget(self.plotLayout,0,0,6,8)


        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(3000)
        self.slider.setValue(0)
        # self.slider.setTickPosition(QSlider.TicksBelow)
        # self.slider.setTickInterval(100)
        self.slider.sliderReleased.connect(self.valuechange)
        self.grid_rt.addWidget(self.slider, 6, 0, 1, 8)

        self.btnOpenAnnDock = QPushButton("Add annotations", self)
        self.btnOpenAnnDock.clicked.connect(self.openAnnDock)
        self.btnOpenAnnDock.setToolTip("Click to open annotations dock")
        self.grid_rt.addWidget(self.btnOpenAnnDock, 7, 0)
        self.btnOpenAnnDock.hide()

        buttonLt10s = QPushButton("<10", self)
        buttonLt10s.clicked.connect(self.leftPlot10s)
        buttonLt10s.setToolTip("Click to go back")
        self.grid_rt.addWidget(buttonLt10s, 7, 1)

        buttonLt1s = QPushButton("<<1", self)
        buttonLt1s.clicked.connect(self.leftPlot1s)
        buttonLt1s.setToolTip("Click to go back")
        self.grid_rt.addWidget(buttonLt1s, 7, 2)

        buttonChgCount = QPushButton("Jump to...", self)
        buttonChgCount.clicked.connect(self.getCount)
        buttonChgCount.setToolTip("Click to select time for graph")
        self.grid_rt.addWidget(buttonChgCount, 7, 3, 1, 2)

        buttonRt1s = QPushButton("1>>", self)
        buttonRt1s.clicked.connect(self.rightPlot1s)
        buttonRt1s.setToolTip("Click to advance")
        self.grid_rt.addWidget(buttonRt1s, 7, 5)

        buttonRt10s = QPushButton("10>", self)
        buttonRt10s.clicked.connect(self.rightPlot10s)
        buttonRt10s.setToolTip("Click to advance")
        self.grid_rt.addWidget(buttonRt10s, 7, 6)

        self.time_lbl = QLabel("0:00:00", self)
        self.grid_rt.addWidget(self.time_lbl, 7, 7)

        # Annotation dock
        self.scroll = QDockWidget()
        #ann_title = QLabel("Annotations")
        #self.scroll.setTitleBarWidget(ann_title)
        self.btnOpenEditAnn = QPushButton("Open annotation editor", self)
        self.btnOpenEditAnn.clicked.connect(self.openAnnEditor)
        self.btnOpenEditAnn.setToolTip("Click to open annotation editor")
        self.scroll.setTitleBarWidget(self.btnOpenEditAnn)
        self.ann_qlist = QListWidget()
        self.scroll.setWidget(self.ann_qlist)
        self.scroll.setFloating(False)
        self.addDockWidget(Qt.RightDockWidgetArea, self.scroll)
        self.scroll.hide()
        self.ann_qlist.itemClicked.connect(self.ann_clicked)

        # Annotation editor dock
        self.annEditDock = QDockWidget()
        self.annEditDock.setTitleBarWidget(QLabel("Annotation editor"))
        self.annEditMainWidget = QWidget()
        self.annEditLayout = QGridLayout()
        annTxtLabel = QLabel("Text: ", self)
        annTimeLabel = QLabel("Time: ", self)
        annDurationLabel = QLabel("Duration: ", self)
        self.annTxtEdit = QLineEdit(self)
        self.annTimeEditTime = QTimeEdit(self)
        self.annTimeEditTime.setMinimumTime(QTime(0,0,0))
        self.annTimeEditTime.setDisplayFormat("hh:mm:ss")
        self.annTimeEditTime.timeChanged.connect(self.updateCountTime)
        self.annTimeEditCount = QSpinBox(self)
        self.annTimeEditCount.valueChanged.connect(self.updateNormalTime)
        # self.annTimeEditCount.setValue()
        # self.btnGetLP.setRange(0, self.data.fs / 2)
        self.annDuration = QSpinBox(self)
        self.btnAnnEdit = QPushButton("Update", self)
        self.btnAnnEdit.clicked.connect(self.annEditorUpdate)
        self.btnAnnEdit.setToolTip("Click to modify selected annotation")
        self.btnAnnDel = QPushButton("Delete", self)
        self.btnAnnDel.clicked.connect(self.annEditorDel)
        self.btnAnnDel.setToolTip("Click to delete selected annotation")
        self.btnAnnCreate = QPushButton("Create", self)
        self.btnAnnCreate.clicked.connect(self.annEditorCreate)
        self.btnAnnCreate.setToolTip("Click to create new annotation")
        self.annEditLayout.addWidget(annTxtLabel,0,0)
        self.annEditLayout.addWidget(self.annTxtEdit,0,1,1,2)
        self.annEditLayout.addWidget(annTimeLabel,1,0)
        self.annEditLayout.addWidget(self.annTimeEditTime,1,1)
        self.annEditLayout.addWidget(self.annTimeEditCount,1,2)
        self.annEditLayout.addWidget(annDurationLabel,2,0)
        self.annEditLayout.addWidget(self.annDuration,2,1)
        self.annEditLayout.addWidget(self.btnAnnEdit,3,0)
        self.annEditLayout.addWidget(self.btnAnnDel,3,1)
        self.annEditLayout.addWidget(self.btnAnnCreate,3,2)
        self.annEditMainWidget.setLayout(self.annEditLayout)
        self.annEditDock.setWidget(self.annEditMainWidget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.annEditDock)
        self.annEditDock.hide()

        #self.tabifyDockWidget(self.scroll,self.annEditDock)
        self.splitDockWidget(self.annEditDock, self.scroll, Qt.Vertical)

        wid = QWidget(self)
        self.setCentralWidget(wid)
        wid.setLayout(layout)
        layout.addLayout(grid_lt, 0, 0, 3, 1)
        layout.addLayout(self.grid_rt, 0, 1, 4, 3)

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.count = 0  # the current location in time we are plotting
        self.init = 0  # if any data has been loaded in yet
        self.window_size = 10  # number of seconds to display at a time
        self.filter_checked = 0  # whether or not to plot filtered data
        self.ylim = [150, 100]  # ylim for unfiltered and filtered data
        self.predicted = 0  # whether or not predictions have been made
        self.filter_win_open = 0  # whether or not filter options window is open
        self.preds_win_open = 0  # whether or not the predictions window is open
        self.chn_win_open = 0  # whether or not the channel selection window is open
        self.organize_win_open = 0  # whether or not the signal organization window is open
        self.spec_win_open = 0 # whether or not the spectrogram window is open
        self.saveimg_win_open = 0 # whether or not the print preview window is open
        self.saveedf_win_open = 0 # whether or not the save edf options window is open
        self.anon_win_open = 0 # whether or not the anonymize window is open
        self.max_time = 0  # number of seconds in the recording
        self.pi = PredsInfo()  # holds data needed to predict
        self.ci = ChannelInfo()  # holds channel information
        self.si = SpecInfo() # holds spectrogram information
        self.sii = SaveImgInfo() # holds info to save the img
        self.sei = SaveEdfInfo() # holds header for edf saving

        if self.argv.show:
            self.show()
            if not self.argv.fn is None:
                self.load_data(self.argv.fn)
        else:
            fn = self.argv.fn
            self.argv_pred_fn = self.argv.predictions_file
            self.argv_mont_fn = self.argv.montage_file
            self.load_data(fn)

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
        if self.organize_win_open:
            self.chn_org.closeWindow()
        if self.spec_win_open:
            self.spec_ops.closeWindow()
        if self.saveimg_win_open:
            self.saveimg_ops.closeWindow()
        if self.saveedf_win_open:
            self.saveedf_ops.closeWindow()
        if self.anon_win_open:
            self.anon_ops.closeWindow()

        event.accept()

    def initGraph(self):
        """
        Function to properly initialize everything when new data
        is loaded.
        """
        # self.init = 1 # set in load_data to prevent issues with slider
        self.fi = FilterInfo()  # holds data needed to filter
        self.filter_checked = 0  # whether or not filter checkbox is checked
        self.cbox_filter.setChecked(False)

        # check if this file is already filtered
        ann = self.edf_info.annotations
        if len(ann[0]) > 0 and ann[2][0] == "filtered":
            self.filter_checked = 1  # whether or not filter checkbox is checked
            # strFilt = ann[2][1].split("Hz")
            strLP = ann[2][1].split("Hz")[0][4:]
            strHP = ann[2][2].split("Hz")[0][4:]
            strN = ann[2][3].split("Hz")[0][3:]
            strBP1 = ann[2][4].split("-")[0][4:]
            strBP2 = ann[2][4].split("-")[1].split("Hz")[0]
            if float(strLP) > 0:
                self.fi.lp = float(strLP)
            else:
                self.fi.do_lp = 0
            if float(strHP) > 0:
                self.fi.hp = float(strHP)
            else:
                self.fi.do_hp = 0
            if float(strN) > 0:
                self.fi.notch = float(strN)
            else:
                self.fi.do_notch = 0
            if float(strBP1) > 0 and float(strBP2) > 0:
                self.fi.do_bp = 1
                self.fi.bp1 = float(strBP1)
                self.fi.bp2 = float(strBP2)
            else:
                self.fi.do_bp = 0
        else:
            self.fi.lp = self.argv.filter[1]
            self.fi.hp = self.argv.filter[2]
            self.fi.notch = self.argv.filter[3]
            self.fi.bp1 = self.argv.filter[4]
            self.fi.bp2 = self.argv.filter[5]
            self.fi.do_lp = self.fi.lp != 0
            self.fi.do_hp = self.fi.hp != 0
            self.fi.do_notch = self.fi.notch != 0
            self.fi.do_bp = self.fi.bp1 != 0 and self.fi.bp2 != 0
            if (self.fi.do_lp or self.fi.do_hp or self.fi.do_notch or self.fi.do_bp) and self.argv.filter[0] == 1:
                self.filter_checked = 1

        if self.btnZoom.text() == "Close zoom":
            self.btnZoom.setText("Open zoom")
            self.plotLayout.removeItem(self.zoomPlot)
            self.mainPlot.removeItem(self.zoomRoi)

        if self.si.plotSpec:
            self.si.plotSpec = 0
            self.removeSpecPlot()
            self.si.chnPlotted = -1

        self.ylim = [150, 100]  # [150,3] # reset scale of axis
        self.window_size = self.argv.window_width # number of seconds displayed at once
        self.wsComboBox.setCurrentIndex(2)
        ind = self.wsComboBox.findText(str(self.window_size) + "s")
        if ind != -1: # -1 for not found
            self.wsComboBox.setCurrentIndex(ind)
        # self.count = 0  # current location in time
        self.ann_list = []  # list of annotations
        self.rect_list = [] # list of prediction rectangles
        self.aspan_list = []  # list of lines on the axis from preds
        self.predLabel.setText("")  # reset text of predictions
        self.thresh = 0.5  # threshold for plotting
        self.threshLblVal.setText(
            "Change threshold of prediction:  (threshold = " + str(self.thresh) + ")")  # reset label
        self.filteredData = []  # set filteredData
        self.si = SpecInfo()

    def ann_clicked(self, item):
        """
        Moves the plot when annotations in the dock are clicked.
        """
        self.count = int(
            float(self.edf_info.annotations[0][self.ann_qlist.currentRow()]))
        self.callmovePlot(0, 0)

        # Update annotation dock if it is open
        if self.btnOpenEditAnn.text() == "Close annotation editor":
            self.annTxtEdit.setText(self.edf_info.annotations[2][self.ann_qlist.currentRow()])
            self.annTimeEditCount.setValue(int(
                float(self.edf_info.annotations[0][self.ann_qlist.currentRow()])))
            self.annDuration.setValue(int(float(self.edf_info.annotations[1][self.ann_qlist.currentRow()])))
            self.btnAnnEdit.setEnabled(True)
            self.btnAnnDel.setEnabled(True)
            blackPen = QPen(QColor(0,0,0))
            # a1 = pg.InfiniteLine(pos=0,movable=True, bounds=None, hoverPen=blackPen)
            # self.mainPlot.addItem(a1)
            # a1.sigDragged.connect(self.printVal)
            # a1.sigPositionChangeFinished.connect(self.printVal)

    def openAnnEditor(self):
        """
        Create and open the annotation editor.
        """
        if self.btnOpenEditAnn.text() == "Open annotation editor":
            self.annTxtEdit.clear()
            self.annDuration.setRange(-1,self.max_time)
            self.annDuration.setValue(-1)
            hrs, min, sec = convertFromCount(self.max_time)
            t = QTime(hrs, min, sec)
            self.annTimeEditTime.setMaximumTime(t)
            self.annTimeEditCount.setMaximum(self.max_time)
            self.annTimeEditCount.setValue(self.count)
            self.btnAnnEdit.setEnabled(False)
            self.btnAnnDel.setEnabled(False)
            selectedListItems = self.ann_qlist.selectedItems()
            if len(selectedListItems) > 0:
                selectedListItems[0].setSelected(False)
            self.btnOpenEditAnn.setText("Close annotation editor")
            self.annEditDock.show()
        else:
            self.btnOpenEditAnn.setText("Open annotation editor")
            self.annEditDock.hide()
            if len(self.edf_info.annotations[0]) == 0:
                self.populateAnnDock()

    def openAnnDock(self):
        self.scroll.show()
        self.btnOpenAnnDock.hide()
        self.openAnnEditor()

    def populateAnnDock(self):
        """
        Fills the annotation dock with annotations if they exist.
        """
        self.ann_qlist.clear()
        ann = self.edf_info.annotations
        if len(ann[0]) == 0:
            self.scroll.hide()
            self.btnOpenAnnDock.show()
        else:
            for i in range(len(ann[0])):
                self.ann_qlist.addItem(ann[2][i])
            self.scroll.show()
            self.btnOpenAnnDock.hide()

    def annEditorUpdate(self):
        annTxt = self.annTxtEdit.text()
        loc = self.annTimeEditCount.value()
        dur = self.annDuration.value()
        self.edf_info.annotations[0][self.ann_qlist.currentRow()] = loc
        self.edf_info.annotations[1][self.ann_qlist.currentRow()] = dur
        self.edf_info.annotations[2][self.ann_qlist.currentRow()] = annTxt
        self.populateAnnDock()
        self.callmovePlot(0,0)

    def annEditorDel(self):
        self.edf_info.annotations = np.delete(self.edf_info.annotations,self.ann_qlist.currentRow(),axis = 1)
        self.btnAnnEdit.setEnabled(False)
        self.btnAnnDel.setEnabled(False)
        self.annEditDock.hide()
        self.populateAnnDock()
        if len(self.edf_info.annotations[0]) == 0:
            self.scroll.show()
            self.btnOpenAnnDock.hide()
        self.annEditDock.show()
        self.callmovePlot(0,0)

    def annEditorCreate(self):
        annTxt = self.annTxtEdit.text()
        loc = self.annTimeEditCount.value()
        dur = self.annDuration.value()
        i = 0
        while i < len(self.edf_info.annotations[0]):
            if int(float(self.edf_info.annotations[0][i])) > loc:
                # if i > 0:
                #     i -= 1
                break
            i += 1
        if len(self.edf_info.annotations[0]) == 0:
            self.edf_info.annotations = np.append(self.edf_info.annotations, np.array([[loc], [dur], [annTxt]]), axis = 1)
        else:
            self.edf_info.annotations = np.insert(self.edf_info.annotations, i, [loc, dur, annTxt], axis = 1)
        self.populateAnnDock()
        self.callmovePlot(0,0)

    def updateNormalTime(self):
        """
        Updates self.annTimeEditTime when self.annTimeEditCount is changed.
        """
        hrs, min, sec = convertFromCount(self.annTimeEditCount.value())
        t = QTime(hrs, min, sec)
        self.annTimeEditTime.setTime(t)
        self.annDuration.setRange(-1,self.max_time - self.annTimeEditCount.value())

    def updateCountTime(self):
        """
        Updates self.annTimeEditCount when self.annTimeEditTime is changed.
        """
        c = ( 3600 * self.annTimeEditTime.time().hour() +
                60 * self.annTimeEditTime.time().minute() +
                self.annTimeEditTime.time().second() )
        self.annTimeEditCount.setValue(c)
        self.annDuration.setRange(-1,self.max_time - c)

    def openZoomPlot(self):
        if self.init:
            if self.btnZoom.text() == "Open zoom":
                if self.si.plotSpec:
                    self.throwAlert("Please close the spectrogram plot before opening zoom.")
                else:
                    self.zoomPlot = self.plotLayout.addPlot(row=1, col=0, border=True)
                    self.zoomPlot.setMouseEnabled(x=False, y=False)
                    qGraphicsGridLayout = self.plotLayout.ci.layout
                    qGraphicsGridLayout.setRowStretchFactor(0, 2)
                    qGraphicsGridLayout.setRowStretchFactor(1, 1)
                    #pg.setConfigOptions(imageAxisOrder='row-major')
                    self.zoomRoi = pg.RectROI([0,0], [self.edf_info.fs * 2,200], pen=(1,9))
                    self.zoomRoi.addScaleHandle([0.5,1],[0.5,0.5])
                    self.zoomRoi.addScaleHandle([0,0.5],[0.5,0.5])
                    self.mainPlot.addItem(self.zoomRoi)
                    self.zoomRoi.setZValue(1000)
                    self.zoomRoi.sigRegionChanged.connect(self.updateZoomPlot)
                    self.btnZoom.setText("Close zoom")
                    self.zoom_plot_lines = []
                    self.zoom_rect_list = []
                    self.updateZoomPlot()
            else:
                self.btnZoom.setText("Open zoom")
                self.plotLayout.removeItem(self.zoomPlot)
                self.mainPlot.removeItem(self.zoomRoi)

    def updateZoomPlot(self):
        roiPos = self.zoomRoi.pos()
        roiSize = self.zoomRoi.size()

        fs = self.edf_info.fs
        nchns = self.ci.nchns_to_plot

        plotData = np.zeros((self.ci.nchns_to_plot,self.window_size * fs))
        if self.filter_checked == 1:
            y_lim = self.ylim[1]
            self.prep_filter_ws()
            # plotData = np.zeros(self.filteredData.shape)
            plotData += self.filteredData
            stddev = np.std(
                plotData)
            plotData[plotData > 3 * stddev] = 3 * stddev
            plotData[plotData < -3 * stddev] = -3 * stddev
        else:
            # plotData = np.zeros(self.ci.data_to_plot.shape)
            plotData += self.ci.data_to_plot[:,self.count * fs:(self.count + self.window_size) * fs]
            y_lim = self.ylim[0]

        if not (len(self.zoom_plot_lines) > 0 and len(self.zoom_plot_lines) == nchns):
            # self.plotWidget.clear()
            self.zoomPlot.clear()
            self.zoom_plot_lines = []
            for i in range(nchns):
                pen = pg.mkPen(color=self.ci.colors[i], width=2, style=QtCore.Qt.SolidLine)
                self.zoom_plot_lines.append(self.zoomPlot.plot(plotData[i, :]
                             + (i + 1) * y_lim, clickable=False, pen=pen))
        else:
            for i in range(nchns):
                self.zoom_plot_lines[i].setData(plotData[i, :]
                            + (i + 1) * y_lim)

        # add predictions
        if len(self.zoom_rect_list) > 0:
            for a in self.zoom_rect_list:
                self.zoomPlot.removeItem(a)
            self.zoom_rect_list[:] = []

        width = 1 / (nchns + 2)
        if self.predicted == 1:
            blueBrush = QBrush(QColor(38,233,254,50))
            starts, ends, chns = self.pi.compute_starts_ends_chns(self.thresh,
                                        self.count, self.window_size, fs, nchns)
            for k in range(len(starts)):
                if self.pi.pred_by_chn:
                    for i in range(nchns):
                        if chns[k][i]:
                            if i == plotData.shape[0] - 1:
                                r1 = pg.QtGui.QGraphicsRectItem(starts[k] - self.count * fs, y_lim *(i+0.5),
                                        ends[k] - starts[k], y_lim) # (x, y, w, h)
                                r1.setPen(pg.mkPen(None))
                                r1.setBrush(pg.mkBrush(color = (38,233,254,50))) # (r,g,b,alpha)
                                self.zoomPlot.addItem(r1)
                                self.zoom_rect_list.append(r1)
                            else:
                                r1 = pg.QtGui.QGraphicsRectItem(starts[k] - self.count * fs, y_lim *(i + 0.5),
                                        ends[k] - starts[k], y_lim) # (x, y, w, h)
                                r1.setPen(pg.mkPen(None))
                                r1.setBrush(blueBrush) # (r,g,b,alpha)
                                self.zoomPlot.addItem(r1)
                                self.zoom_rect_list.append(r1)
                            x_vals = range(
                                int(starts[k]) - self.count * fs, int(ends[k]) - self.count * fs)
                            pen = pg.mkPen(color=self.ci.colors[i], width=3, style=QtCore.Qt.SolidLine)
                            self.zoom_plot_lines.append(self.zoomPlot.plot(x_vals, plotData[i, int(starts[k]) - self.count * fs:int(ends[k]) - self.count * fs] + i*y_lim + y_lim, clickable=False, pen=pen))
                else:
                    r1 = pg.LinearRegionItem(values=(starts[k] - self.count * fs, ends[k] - self.count * fs),
                                    brush=blueBrush, movable=False, orientation=pg.LinearRegionItem.Vertical)
                    self.zoomPlot.addItem(r1)
                    self.zoom_rect_list.append(r1)

        x_ticks = []
        for i in range(self.window_size):
            x_ticks.append((i * fs, str(self.count + i)))
        x_ticks = [x_ticks]
        y_ticks = []
        for i in range(nchns + 1):
            y_ticks.append((i * y_lim, self.ci.labels_to_plot[i]))
        y_ticks = [y_ticks]

        blackPen = QPen(QColor(0,0,0))
        font = QFont()
        font.setPixelSize(16)
        self.zoomPlot.setYRange(roiPos[1], roiPos[1] + roiSize[1])
        self.zoomPlot.getAxis('left').setPen(blackPen)
        self.zoomPlot.getAxis('left').setTicks(y_ticks)
        self.zoomPlot.getAxis('left').setTextPen(blackPen)
        self.zoomPlot.getAxis("left").setStyle(tickTextOffset = 10)
        self.zoomPlot.setLabel('left', ' ', pen=(0,0,0), fontsize=20)
        self.zoomPlot.setXRange(roiPos[0], roiPos[0] + roiSize[0], padding=0)
        self.zoomPlot.getAxis('bottom').setTicks(x_ticks)
        self.zoomPlot.getAxis('bottom').setTextPen(blackPen)
        self.zoomPlot.getAxis("bottom").tickFont = font
        self.zoomPlot.getAxis('bottom').setPen(blackPen)
        self.zoomPlot.setLabel('bottom', 'Time (s)', pen = blackPen)
        self.zoomPlot.getAxis('top').setWidth(200)

    def openHelp(self):
        """
        Called when you click the help button.
        """
        QDesktopServices.openUrl(QUrl("https://github.com/jcraley/jhu-eeg"))

    def valuechange(self):
        """
        Updates plot when slider is changed.
        """
        if self.init == 1:
            size = self.slider.value()
            self.count = size + 1
            if self.count + self.window_size > self.max_time + 1:
                self.count = self.max_time - self.window_size
            self.callmovePlot(0, 1)

    def changeThreshSlider(self):
        """
        Updates the value of the threshold when the slider is changed.
        """
        val = self.threshSlider.value()
        self.thresh = val / 100
        self.threshLblVal.setText("Change threshold of prediction:  (threshold = " + str(self.thresh) + ")")
        if self.predicted == 1:
            self.callmovePlot(0, 0)

    def chgSig(self):
        """
        Funtion to open channel_options so users can change the signals being
        plotted.
        """
        if self.init and not self.chn_win_open:
            self.chn_win_open = 1
            self.chn_ops = ChannelOptions(self.ci, self)
            self.chn_ops.show()

    def save_to_edf(self):
        """
        Opens window for anonymization. Anonymizer window calls save_sig_to_edf
        to save to file.
        """
        if self.init == 1:
            self.saveedf_win_open = 1
            self.saveedf_ops = SaveEdfOptions(self.sei, self)

    def save_sig_to_edf(self):
        """
        Function to save current data to .edf file, called by anonymization windows
        """
        if self.init == 1:
            if self.filter_checked == 1:
                dataToSave = filterData(
                    self.ci.data_to_plot, self.edf_info.fs, self.fi)
                if self.fi.filter_canceled == 1:
                    self.fi.filter_canceled = 0
                    return
            else:
                dataToSave = self.ci.data_to_plot

            # write annotations
            ann = self.edf_info.annotations
            if len(ann[0]) > 0 and ann[2][0] == "filtered":
                self.throwAlert("If filter values have since been changed, filter history will not be saved.\n"  +
                            "If you would like to append some record of previous filters, please add an annotation.")
            if self.argv.save_edf_fn is None:
                file = QFileDialog.getSaveFileName(self, 'Save File')
                file = file[0]
            else:
                file = self.argv.save_edf_fn

            nchns = self.ci.nchns_to_plot
            labels = self.ci.labels_to_plot

            # if predictions, save them as well
            if self.predicted == 1:
                if self.pi.pred_by_chn:
                    savedEDF = pyedflib.EdfWriter(file + '.edf', nchns * 2)
                else:
                    savedEDF = pyedflib.EdfWriter(file + '.edf', nchns + 1)
            else:
                savedEDF = pyedflib.EdfWriter(file + '.edf', nchns)

            self.sei.convertToHeader()
            savedEDF.setHeader(self.sei.pyedf_header)
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
                        savedEDF.setSamplefrequency(
                            nchns + i, fs / self.pi.pred_width)
                        savedEDF.setLabel(nchns + i, "PREDICTIONS_" + str(i))
                    for i in range(nchns):
                        temp.append(self.pi.preds_to_plot[:, i])
                else:
                    savedEDF.setPhysicalMaximum(nchns, 1)
                    savedEDF.setPhysicalMinimum(nchns, 0)
                    savedEDF.setSamplefrequency(nchns, fs / self.pi.pred_width)
                    savedEDF.setLabel(nchns, "PREDICTIONS")
                    temp.append(self.pi.preds_to_plot)
                dataToSave = temp

            savedEDF.writeSamples(dataToSave)

            # write annotations
            if len(ann[0]) > 0 and ann[2][0] == "filtered":
                for aa in range(5): # remove any old annotations
                    ann = np.delete(ann, 0, axis=1)
            if self.filter_checked == 1:
                if len(ann[0]) == 0:
                    ann = np.array([0.0, -1.0, "filtered"])
                    ann = ann[..., np.newaxis]
                else:
                    ann = np.insert(ann, 0, [0.0, -1.0, "filtered"], axis=1)
                strFilt = ""
                strFilt += "LP: " + str(self.fi.do_lp * self.fi.lp) + "Hz"
                ann = np.insert(ann, 1, [0.0, -1.0, strFilt], axis=1)
                strFilt = "" + "HP: " + str(self.fi.do_hp * self.fi.hp) + "Hz"
                ann = np.insert(ann, 2, [0.0, -1.0, strFilt], axis=1)
                strFilt = "" + "N: " + str(self.fi.do_notch * self.fi.notch) + "Hz"
                ann = np.insert(ann, 3, [0.0, -1.0, strFilt], axis=1)
                strFilt = "" + "BP: " + str(self.fi.do_bp * self.fi.bp1) + "-" + str(self.fi.do_bp * self.fi.bp2) + "Hz"
                ann = np.insert(ann, 4, [0.0, -1.0, strFilt], axis=1)
                # ann = np.insert(ann, 1, [0.0, -1.0, strFilt], axis=1)
            for i in range(len(ann[0])):
                savedEDF.writeAnnotation(
                    float(ann[0][i]), float((ann[1][i])), ann[2][i])

            # Close file
            savedEDF.close()

            # if you are just saving to edf, close the window
            if not self.argv.save_edf_fn is None:
                self.argv.save_edf_fn = None
                if self.argv.show == 0:
                    sys.exit()


    def load_data(self, name=""):
        """
        Function to load in the data

        loads selected .edf file into edf_info and data
        data is initially unfiltered
        """
        if self.init or self.argv.fn is None:
            name = QFileDialog.getOpenFileName(
                self, 'Open file', '.', 'EDF files (*.edf)')
            name = name[0]
        if name == None or len(name) == 0:
            return
        else:
            loader = EdfLoader()
            try:
                self.edf_info_temp = loader.load_metadata(name)
            except:
                self.throwAlert("The .edf file is invalid.")
                return
            self.edf_info_temp.annotations = np.array(
                self.edf_info_temp.annotations)

            edf_montages = EdfMontage(self.edf_info_temp)
            # fs_idx = edf_montages.getIndexForFs(self.edf_info_temp.labels2chns)

            self.data_temp = loader.load_buffers(self.edf_info_temp)
            data_for_preds = self.data_temp
            self.edf_info_temp.fs, self.data_temp = loadSignals(
                self.data_temp, self.edf_info_temp.fs)

            # setting temporary variables that will be overwritten if
            # the user selects signals to plot
            self.max_time_temp = int(
                self.data_temp.shape[1] / self.edf_info_temp.fs)
            self.ci_temp = ChannelInfo()  # holds channel information
            self.ci_temp.chns2labels = self.edf_info_temp.chns2labels
            self.ci_temp.labels2chns = self.edf_info_temp.labels2chns
            self.ci_temp.fs = self.edf_info_temp.fs
            self.ci_temp.max_time = self.max_time_temp
            self.fn_full_temp = name
            if len(name.split('/')[-1]) < 40:
                self.fn_temp = name.split('/')[-1]
            else:
                self.fn_temp = name.split('/')[-1][0:37] + "..."

            self.chn_win_open = 1
            self.chn_ops = ChannelOptions(self.ci_temp, self, data_for_preds)
            if self.argv.show and self.argv.montage_file is None:
                self.chn_ops.show()

    def callInitialMovePlot(self):
        """
        Function called by channel_options when channels are loaded
        """
        self.initGraph()

        self.fi.fs = self.ci.fs
        self.slider.setMaximum(self.max_time - self.window_size)
        self.threshSlider.setValue(self.thresh * 100)

        self.ann_qlist.clear()  # Clear annotations
        self.populateAnnDock()  # Add annotations if they exist

        nchns = self.ci.nchns_to_plot
        self.plot_lines = []
        if not self.init and self.argv.location < self.max_time - self.window_size:
            self.count = self.argv.location

        ann = self.edf_info.annotations
        if self.filter_checked == 1 or (len(ann[0]) > 0 and ann[2][0] == "filtered"):
            self.movePlot(0, 0, self.ylim[1], 0)
        else:
            # profile.runctx('self.movePlot(0, 0, self.ylim[0], 0)', globals(), locals())
            self.movePlot(0, 0, self.ylim[0], 0)

        if not self.argv.save_edf_fn is None and self.init == 0:
            self.init = 1
            self.save_to_edf()

        self.init = 1

        ann = self.edf_info.annotations
        if len(ann[0]) > 0 and ann[2][0] == "filtered" or self.filter_checked == 1:
            self.cbox_filter.setChecked(True)  # must be set after init = 1

    def rightPlot1s(self):
        self.callmovePlot(1, 1)

    def leftPlot1s(self):
        self.callmovePlot(0, 1)

    def rightPlot10s(self):
        self.callmovePlot(1, 10)

    def leftPlot10s(self):
        self.callmovePlot(0, 10)

    def incAmp(self):
        if self.init == 1:
            if self.ylim[0] > 50:
                self.ylim[0] = self.ylim[0] - 15
                self.ylim[1] = self.ylim[1] - 10
                self.callmovePlot(0, 0)

    def decAmp(self):
        if self.init == 1:
            if self.ylim[0] < 250:
                self.ylim[0] = self.ylim[0] + 15
                self.ylim[1] = self.ylim[1] + 10
                self.callmovePlot(0, 0)

    """def incWindow_size(self):
        if self.init == 1:
            if self.window_size + 5 <= 30:
                if self.window_size == 1:
                    self.window_size = 5
                else:
                    self.window_size = self.window_size + 5
                self.slider.setMaximum(self.max_time - self.window_size)
                if self.count + self.window_size > self.max_time:
                    self.count = self.max_time - self.window_size
                self.callmovePlot(0, 0)

    def decWindow_size(self):
        if self.init == 1:
            if self.window_size - 5 >= 5:
                self.window_size = self.window_size - 5
                self.slider.setMaximum(self.max_time - self.window_size)
                self.callmovePlot(0, 0)
            else:
                self.window_size = 1
                self.callmovePlot(0, 0)"""

    def chgWindow_size(self):
        if self.init == 1:
            new_ws = self.wsComboBox.currentText()
            new_ws = int(new_ws.split("s")[0])
            self.window_size = new_ws
            self.slider.setMaximum(self.max_time - self.window_size)
            self.callmovePlot(0, 0)
        else:
            self.wsComboBox.setCurrentIndex(2)

    def getCount(self):
        """
        Used for the "jump to" button to update self.count to the user's input
        """
        if self.init == 1:
            num, ok = QInputDialog.getInt(self, "Jump to...", "Enter a time in seconds:",
                                          0, 0, self.max_time)
            if ok:
                if num > self.max_time - self.window_size:
                    num = self.max_time - self.window_size
                self.count = num
                self.callmovePlot(0, 0)

    def print_graph(self):
        if self.init and self.saveimg_win_open == 0:
            self.callmovePlot(0, 0, 1)
            self.saveimg_win_open = 1
            self.saveimg_ops = SaveImgOptions(self.sii, self)
            self.saveimg_ops.show()

    def callmovePlot(self, right, num_move, print_graph=0):
        """
        Helper function to call movePlot for various buttons.
        """
        if self.init == 1:
            if self.filter_checked == 1:
                self.movePlot(right, num_move, self.ylim[1], print_graph)
            else:
                self.movePlot(right, num_move, self.ylim[0], print_graph)

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
        if not self.argv.predictions_file is None and self.init == 0:
            self.predicted = 1
            self.pi.set_preds(self.argv.predictions_file, self.max_time,
                              fs, self.ci.nchns_to_plot)
            self.pi.preds_to_plot = self.pi.preds

        if right == 0 and self.count - num_move >= 0:
            self.count = self.count - num_move
        elif right == 1 and self.count + num_move + self.window_size <= self.data.shape[1] / fs:
            self.count = self.count + num_move
        self.slider.setValue(self.count)
        t = getTime(self.count)
        self.time_lbl.setText(t)

        plotData = np.zeros((self.ci.nchns_to_plot,self.window_size * fs))
        if self.filter_checked == 1:
            self.prep_filter_ws()
            # plotData = np.zeros(self.filteredData.shape)
            plotData += self.filteredData
            stddev = np.std(plotData)
            plotData[plotData > 3 * stddev] = 3 * stddev  # float('nan') # clip amplitude
            plotData[plotData < -3 * stddev] = -3 * stddev
        else:
            plotData += self.ci.data_to_plot[:,self.count * fs:(self.count + self.window_size) * fs]
            stddev = np.std(plotData)
            plotData[plotData > 5 * stddev] = 5 * stddev  # float('nan') # clip amplitude
            plotData[plotData < -5 * stddev] = -5 * stddev

        nchns = self.ci.nchns_to_plot
        if self.predicted == 1:
            self.predLabel.setText("Predictions plotted.")
        else:
            self.predLabel.setText("")
        # Clear plot
        """del(self.ax.lines[:])
        for i, a in enumerate(self.ann_list):
            a.remove()
        self.ann_list[:] = []
        for aspan in self.aspan_list:
            aspan.remove()
        self.aspan_list[:] = []

        for i in range(nchns):
            self.ax.plot(plotData[i, self.count * fs:(self.count + 1) * fs*self.window_size]
                         + (i + 1) * y_lim, '-', linewidth=0.5, color=self.ci.colors[i])
            self.ax.set_ylim([-y_lim, y_lim * (nchns + 1)])
            self.ax.set_yticks(np.arange(0, (nchns + 2)*y_lim, step=y_lim))
            self.ax.set_yticklabels(
                self.ci.labels_to_plot, fontdict=None, minor=False, fontsize=12)

            width = 1 / (nchns + 2)
            if self.predicted == 1:
                starts, ends, chns = self.pi.compute_starts_ends_chns(self.thresh,
                                                                      self.count, self.window_size, fs, nchns)
                for k in range(len(starts)):
                    if self.pi.pred_by_chn:
                        if chns[k][i]:
                            if i == plotData.shape[0] - 1:
                                self.aspan_list.append(self.ax.axvspan(starts[k] - self.count * fs, ends[k] - self.count * fs,
                                                                       ymin=width*(i+1.5), ymax=1, color='paleturquoise', alpha=1))
                            else:
                                self.aspan_list.append(self.ax.axvspan(starts[k] - self.count * fs, ends[k] - self.count * fs,
                                                                       ymin=width*(i+1.5), ymax=width*(i+2.5), color='paleturquoise', alpha=1))
                            x_vals = range(
                                int(starts[k]) - self.count * fs, int(ends[k]) - self.count * fs)
                            self.ax.plot(x_vals, plotData[i, int(starts[k]):int(ends[k])] + i*y_lim + y_lim,
                                         '-', linewidth=1, color=self.ci.colors[i])
                    else:
                        self.aspan_list.append(self.ax.axvspan(
                            starts[k] - self.count * fs, ends[k] - self.count * fs, color='paleturquoise', alpha=0.5))

        self.ax.set_xlim([0, self.edf_info.fs*self.window_size])
        step_size = self.edf_info.fs  # Updating the x labels with scaling
        step_width = 1
        if self.window_size >= 15 and self.window_size <= 25:
            step_size = step_size * 2
            step_width = step_width * 2
        elif self.window_size > 25:
            step_size = step_size * 3
            step_width = step_width * 3
        self.ax.set_xticks(np.arange(0, self.window_size *
                                     self.edf_info.fs + 1, step=step_size))
        self.ax.set_xticklabels(np.arange(self.count, self.count + self.window_size + 1,
                                          step=step_width), fontdict=None, minor=False, fontsize=12)
        self.ax.set_xlabel("Time (s)")

        ann, idx_w_ann = checkAnnotations(
            self.count, self.window_size, self.edf_info)
        font_size = 10 - self.window_size / 5
        if font_size < 7:
            font_size = 7
        # Add in annotations
        if len(ann) != 0:
            ann = np.array(ann).T
            txt = ""
            int_prev = int(float(ann[0, 0]))
            for i in range(ann.shape[1]):
                int_i = int(float(ann[0, i]))
                if int_prev == int_i:
                    txt = txt + "\n" + ann[2, i]
                else:
                    if idx_w_ann[int_prev - self.count] and int_prev % 2 == 1:
                        self.ann_list.append(self.ax.annotate(txt, xy=(
                            (int_prev - self.count)*fs, -y_lim / 2 + y_lim), color='black', size=font_size))
                    else:
                        self.ann_list.append(self.ax.annotate(txt, xy=(
                            (int_prev - self.count)*fs, -y_lim / 2), color='black', size=font_size))
                    txt = ann[2, i]
                int_prev = int_i
            if txt != "":
                if idx_w_ann[int_i - self.count] and int_i % 2 == 1:
                    self.ann_list.append(self.ax.annotate(txt, xy=(
                        (int_i - self.count)*fs, -y_lim / 2 + y_lim), color='black', size=font_size))
                else:
                    self.ann_list.append(self.ax.annotate(
                        txt, xy=((int_i - self.count)*fs, -y_lim / 2), color='black', size=font_size))

        if print_graph == 1:
            file = QFileDialog.getSaveFileName(self, 'Save File')
            self.ax.figure.savefig(file[0] + ".png", bbox_inches='tight')
        elif len(self.argv) > 0:
            self.ax.figure.savefig(print_fn, bbox_inches='tight')
            sys.exit()

        self.m.draw()"""
        # self.mainPlot.disableAutoRange()

        if not (len(self.plot_lines) > 0 and len(self.plot_lines) == nchns):
            # self.plotWidget.clear()
            self.mainPlot.clear()
            self.plot_lines = []
            for i in range(nchns):
                pen = pg.mkPen(color=self.ci.colors[i], width=2, style=QtCore.Qt.SolidLine)
                self.plot_lines.append(self.mainPlot.plot(plotData[i, :]
                             + (i + 1) * y_lim, clickable=False, pen=pen))
        else:
            for i in range(nchns):
                self.plot_lines[i].setData(plotData[i, :]
                            + (i + 1) * y_lim)

        # add predictions
        if len(self.rect_list) > 0:
            for a in self.rect_list:
                self.mainPlot.removeItem(a)
            self.rect_list[:] = []

        width = 1 / (nchns + 2)
        if self.predicted == 1:
            blueBrush = QBrush(QColor(38,233,254,50))
            starts, ends, chns = self.pi.compute_starts_ends_chns(self.thresh,
                                        self.count, self.window_size, fs, nchns)
            for k in range(len(starts)):
                if self.pi.pred_by_chn:
                    for i in range(nchns):
                        if chns[k][i]:
                            if i == plotData.shape[0] - 1:
                                r1 = pg.QtGui.QGraphicsRectItem(starts[k] - self.count * fs, y_lim *(i+0.5),
                                        ends[k] - starts[k], y_lim) # (x, y, w, h)
                                r1.setPen(pg.mkPen(None))
                                r1.setBrush(pg.mkBrush(color = (38,233,254,50))) # (r,g,b,alpha)
                                self.mainPlot.addItem(r1)
                                self.rect_list.append(r1)
                            else:
                                r1 = pg.QtGui.QGraphicsRectItem(starts[k] - self.count * fs, y_lim *(i + 0.5),
                                        ends[k] - starts[k], y_lim) # (x, y, w, h)
                                r1.setPen(pg.mkPen(None))
                                r1.setBrush(blueBrush) # (r,g,b,alpha)
                                self.mainPlot.addItem(r1)
                                self.rect_list.append(r1)
                            x_vals = range(
                                int(starts[k]) - self.count * fs, int(ends[k]) - self.count * fs)
                            pen = pg.mkPen(color=self.ci.colors[i], width=3, style=QtCore.Qt.SolidLine)
                            self.plot_lines.append(self.mainPlot.plot(x_vals, plotData[i, int(starts[k]) - self.count * fs:int(ends[k]) - self.count * fs] + i*y_lim + y_lim, clickable=False, pen=pen))
                else:
                    r1 = pg.LinearRegionItem(values=(starts[k] - self.count * fs, ends[k] - self.count * fs),
                                    brush=blueBrush, movable=False, orientation=pg.LinearRegionItem.Vertical)
                    self.mainPlot.addItem(r1)
                    self.rect_list.append(r1)

        step_size = fs  # Updating the x labels with scaling
        step_width = 1
        if self.window_size >= 15 and self.window_size <= 25:
            step_size = step_size * 2
            step_width = step_width * 2
        elif self.window_size > 25:
            step_size = step_size * 3
            step_width = step_width * 3
        x_ticks = []
        spec_x_ticks = []
        for i in range(int(self.window_size / step_width) + 1):
            x_ticks.append((i * step_size, str(self.count + i * step_width)))
            spec_x_ticks.append((i * step_width, str(self.count + i * step_width)))
        x_ticks = [x_ticks]
        spec_x_ticks = [spec_x_ticks]

        y_ticks = []
        for i in range(nchns + 1):
            y_ticks.append((i * y_lim, self.ci.labels_to_plot[i]))
        y_ticks = [y_ticks]

        blackPen = QPen(QColor(0,0,0))
        font = QFont()
        font.setPixelSize(16)

        self.mainPlot.setYRange(-y_lim, (nchns + 1) * y_lim)
        self.mainPlot.getAxis('left').setStyle(tickFont = font)
        self.mainPlot.getAxis('left').setTextPen(blackPen)
        self.mainPlot.getAxis('left').setTicks(y_ticks)
        self.mainPlot.getAxis("left").setStyle(tickTextOffset = 10)
        self.mainPlot.setLabel('left', ' ', pen=(0,0,0), fontsize=20)
        # self.mainPlot.getAxis("left").setScale(y_lim * (nchns + 2))

        self.mainPlot.setXRange(0 * fs, (0 + self.window_size) * fs, padding=0)
        self.mainPlot.getAxis('bottom').setTicks(x_ticks)
        self.mainPlot.getAxis('bottom').setStyle(tickFont = font)
        self.mainPlot.getAxis('bottom').setTextPen(blackPen)
        self.mainPlot.setLabel('bottom', 'Time (s)', pen = blackPen)
        self.mainPlot.getAxis('top').setWidth(200)

        # add annotations
        if len(self.ann_list) > 0:
            for a in self.ann_list:
                self.mainPlot.removeItem(a)
            self.ann_list[:] = []

        ann, idx_w_ann = checkAnnotations(self.count, self.window_size, self.edf_info)
        font_size = 10
        if len(ann) != 0:
            ann = np.array(ann).T
            txt = ""
            int_prev = int(float(ann[0, 0]))
            for i in range(ann.shape[1]):
                int_i = int(float(ann[0, i]))
                if int_prev == int_i:
                    txt = txt + "\n" + ann[2, i]
                else:
                    if idx_w_ann[int_prev - self.count] and int_prev % 2 == 1:
                        txt_item = pg.TextItem(text=txt, color='k', anchor=(0,1))
                        self.mainPlot.addItem(txt_item)
                        txt_item.setPos((int_prev - self.count)*fs, -(3/2)*y_lim)
                        self.ann_list.append(txt_item)
                    else:
                        txt_item = pg.TextItem(text=txt, color='k', anchor=(0,1))
                        self.mainPlot.addItem(txt_item)
                        txt_item.setPos((int_prev - self.count)*fs, -y_lim)
                        self.ann_list.append(txt_item)
                    txt = ann[2, i]
                int_prev = int_i
            if txt != "":
                if idx_w_ann[int_i - self.count] and int_i % 2 == 1:
                    txt_item = pg.TextItem(text=txt, color='k', anchor=(0,1))
                    self.mainPlot.addItem(txt_item)
                    txt_item.setPos((int_i - self.count)*fs, -(3 / 2) *y_lim)
                    self.ann_list.append(txt_item)
                else:
                    txt_item = pg.TextItem(text=txt, color='k', anchor=(0,1))
                    self.mainPlot.addItem(txt_item)
                    txt_item.setPos((int_i - self.count)*fs, -y_lim)
                    self.ann_list.append(txt_item)

        if print_graph == 1 or (not self.argv.export_png_file is None and self.init == 0):
            # exporter = pg.exporters.ImageExporter(self.plotWidget.scene())
            # exporter.export(file[0] + '.png')
            self.sii.data = plotData
            self.sii.pi = self.pi
            self.sii.ci = self.ci
            self.sii.predicted = self.predicted
            self.sii.fs = fs
            self.sii.count = self.count
            self.sii.window_size = self.window_size
            self.sii.y_lim = y_lim
            self.sii.thresh = self.thresh
        if not self.argv.export_png_file is None and self.init == 0:
            self.saveimg_win_open = 1
            self.saveimg_ops = SaveImgOptions(self.sii, self)

        if self.si.plotSpec:
            # dataForSpec = self.si.data
            # f, t, Sxx = scipy.signal.spectrogram(self.si.data[self.count * fs:(self.count + self.window_size) * fs], fs=fs, nperseg=fs, noverlap=0)
            # Fit the min and max levels of the histogram to the data available
            # self.hist.axis.setPen(blackPen)
            # self.hist.setLevels(0,200)#np.min(Sxx), np.max(Sxx))
            # This gradient is roughly comparable to the gradient used by Matplotlib
            # You can adjust it and then save it using hist.gradient.saveState()
            # self.hist.gradient.restoreState(
            #     {'mode': 'rgb',
            #     'ticks': [(0.5, (0, 182, 188, 255)),
            #            (1.0, (246, 111, 0, 255)),
            #            (0.0, (75, 0, 113, 255))]})
            # Sxx contains the amplitude for each pixel
            # self.img.setImage(Sxx)
            # Scale the X and Y Axis to time and frequency (standard is pixels)
            # self.img.scale(self.window_size/np.size(Sxx, axis=1),
            #         f[-1]/np.size(Sxx, axis=0))
            # Limit panning/zooming to the spectrogram
            # self.specPlot.setLimits(xMin=0, xMax=self.window_size, yMin=self.si.minFs, yMax=self.si.maxFs)
            self.specTimeSelectChanged()
            self.specPlot.getAxis('bottom').setTextPen(blackPen)
            # self.specPlot.getAxis('bottom').setTicks(spec_x_ticks)
            # Add labels to the axis
            self.specPlot.setLabel('bottom', "Frequency", units='Hz')
            # pyqtgraph automatically scales the axis and adjusts the SI prefix (in this case kHz)
            self.specPlot.getAxis('left').setTextPen(blackPen)
            self.specPlot.setLabel('left', "PSD", units='V**2/Hz')
            self.specPlot.setXRange(self.si.minFs,self.si.maxFs,padding=0)
            self.specPlot.setLogMode(False, True)
            # self.specPlot.setYRange(self.si.minFs,self.si.maxFs,padding=0)
            self.specPlot.setTitle(self.si.chnName,color='k',size='16pt')
        if self.btnZoom.text() == "Close zoom":
            self.updateZoomPlot()

        if self.init == 0 and self.argv.show:
            self.throwAlert("Data has been plotted.")

    def printVal(self):
        print(self.sender().value())

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
            self.callmovePlot(1, 0)
        elif self.init == 0 and cbox.isChecked():
            cbox.setChecked(False)

    def prep_filter_ws(self):
        """
        Does filtering for one window of size window_size
        """
        fs = self.edf_info.fs
        if len(self.filteredData) == 0 or self.filteredData.shape != self.ci.data_to_plot[:,self.count*fs:(self.count + self.window_size)*fs].shape:
            self.filteredData = np.zeros((self.ci.nchns_to_plot,self.window_size * fs))
        filt_window_size = filterData(
            self.ci.data_to_plot[:, self.count * fs:(self.count + self.window_size)*fs], fs, self.fi)
        filt_window_size = np.array(filt_window_size)
        self.filteredData = filt_window_size

    def changeFilter(self):
        """
        Opens the FilterOptions window
        """
        if self.init == 1:
            self.filter_win_open = 1
            self.filter_ops = FilterOptions(self.fi, self)
            self.filter_ops.show()

    def changePredictions(self):
        """
        Take loaded model and data and compute predictions
        """
        if self.init == 1:
            self.preds_win_open = 1
            self.pred_ops = PredictionOptions(self.pi, self)
            self.pred_ops.show()

    def makeSpecPlot(self):
        """
        Creates the spectrogram plot.
        """
        self.specPlot = self.plotLayout.addPlot(row=1, col=0)
        fs = self.edf_info.fs
        f, Pxx_den = signal.welch(self.si.data[(1) * fs:(4) * fs], fs)
        self.specPlot.clear()
        self.spec_plot_lines = []
        pen = QPen(QColor(0,0,0))
        # pen = pg.mkPen(color=self.ci.colors[i], width=2, style=QtCore.Qt.SolidLine)
        self.spec_plot_lines.append(self.specPlot.plot(f,Pxx_den, clickable=False, pen=pen))


        self.specPlot.setMouseEnabled(x=False, y=False)
        qGraphicsGridLayout = self.plotLayout.ci.layout
        qGraphicsGridLayout.setRowStretchFactor(0, 2)
        qGraphicsGridLayout.setRowStretchFactor(1, 1)
        # pg.setConfigOptions(imageAxisOrder='row-major')
        # self.img = pg.ImageItem() # Item for displaying image data
        # self.specPlot.addItem(self.img)
        # self.hist = pg.HistogramLUTItem() # Add a histogram with which to control the gradient of the image
        # self.hist.setImageItem(self.img) # Link the histogram to the image
        # self.plotLayout.addItem(self.hist, row = 1, col = 1) # To make visible, add the histogram
        # self.hist.setLevels(0,200)
        redBrush = QBrush(QColor(217, 43, 24,50))
        nchns = self.ci.nchns_to_plot
        self.selectTimeRect = pg.LinearRegionItem(values=(fs, 4 * fs),
                        brush=redBrush, movable=True, orientation=pg.LinearRegionItem.Vertical)
        self.selectTimeRect.setSpan((self.si.chnPlotted + 2) / (nchns + 3),(self.si.chnPlotted + 3) / (nchns + 3))
        self.selectTimeRect.setBounds([0,fs * self.window_size])
        self.mainPlot.addItem(self.selectTimeRect)
        self.selectTimeRect.sigRegionChangeFinished.connect(self.specTimeSelectChanged)

    def specTimeSelectChanged(self):
        """
        Function called when the user changes the region that selects where in
        time to compute the power spectrum
        """
        fs = self.edf_info.fs
        bounds = self.selectTimeRect.getRegion()
        bounds = bounds + self.count * fs
        # f, Pxx_den = signal.welch(self.si.data[int(bounds[0]):int(bounds[1])], fs)
        f, Pxx_den = signal.periodogram(self.si.data[int(bounds[0]):int(bounds[1])], fs)
        pen = pg.mkPen(color=(178, 7, 245), width=3, style=QtCore.Qt.SolidLine)
        # pen = pg.mkPen(color=self.ci.colors[i], width=2, style=QtCore.Qt.SolidLine)
        self.spec_plot_lines[0].setData(f,Pxx_den, clickable=False, pen=pen)

    def updateSpecChn(self):
        self.mainPlot.removeItem(self.selectTimeRect)
        redBrush = QBrush(QColor(217, 43, 24,50))
        nchns = self.ci.nchns_to_plot
        fs = self.edf_info.fs
        self.selectTimeRect = pg.LinearRegionItem(values=(fs, 4 * fs),
                        brush=redBrush, movable=True, orientation=pg.LinearRegionItem.Vertical)
        self.selectTimeRect.setSpan((self.si.chnPlotted + 2) / (nchns + 3),(self.si.chnPlotted + 3) / (nchns + 3))
        self.mainPlot.addItem(self.selectTimeRect)
        self.selectTimeRect.sigRegionChangeFinished.connect(self.specTimeSelectChanged)

    def removeSpecPlot(self):
        """
        Removes the spectrogram plot.
        """
        self.plotLayout.removeItem(self.specPlot)
        # self.plotLayout.removeItem(self.hist)
        self.mainPlot.removeItem(self.selectTimeRect)

    def loadSpec(self):
        """
        Opens the SpecOptions window
        """
        if self.init == 1:
            if self.btnZoom.text() == "Close zoom":
                self.throwAlert("Please close the zoom plot before opening the spectrogram.")
            else:
                self.spec_win_open = 1
                self.spec_ops = SpecOptions(self.si, self)
                self.spec_ops.show()

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
        self.fig = Figure(figsize=(width, height), dpi=dpi,
                          constrained_layout=False)
        self.gs = self.fig.add_gridspec(1, 1, wspace=0.0, hspace=0.0)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

class QHLine(QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)

def get_args():
    p = ap.ArgumentParser()

    # Add arguments
    p.add_argument("--fn", type=str,
                    help="Name of EDF file to load.")
    p.add_argument("--predictions-file", type=str, help="Name of prediction file.")
    p.add_argument("--montage-file", type=str, help="Text file with list of montage to load.")
    p.add_argument("--location", type=int, default=0,
                    help="Time in seconds to plot.")
    p.add_argument("--export-png-file", type=str,
                    help="Where to save image.")
    p.add_argument("--window-width", type=int, default=10,
                   choices=[5, 10, 15, 20, 25, 30],
                    help="The width of signals on the plot.")
    p.add_argument("--filter", nargs=6, type=float, default=[0,30,2,0,0,0],
                    help="1 or 0 to set the filter. Low pass, high pass, notch, and bandpass frequencies. Set to 0 to turn off each filter.")
    p.add_argument("--show", type=int, default=1, choices=[0,1],
                    help="Whether or not to show the GUI.")
    p.add_argument("--print-annotations",type=int, default=1, choices=[0,1])
    p.add_argument("--line-thickness",type=float, default=0.5)
    p.add_argument("--font-size",type=int, default=12)
    p.add_argument("--plot-title", type=str, default="")
    p.add_argument("--save-edf-fn", type=str, default=None)
    p.add_argument("--anonymize-edf", type=int, default=1, choices=[0,1])

    return p.parse_args()

def check_args(args):

    mandatory_args = {'fn', 'montage_file', 'show'}
    if args.show == 0:
        if not mandatory_args.issubset(set(dir(args))):
            raise Exception(("You're missing essential arguments!"))

        if args.fn is None:
            raise Exception("--fn must be specified")
        if args.montage_file is None:
            raise Exception("--montage_file must be specified")

    if not args.fn is None and args.montage_file is None:
        raise Exception("--montage_file must be specified if --fn is specified")

    if args.fn is None and not args.montage_file is None:
        raise Exception("--fn must be specified if --montage-file is specified")

    if not args.fn is None:
        if not path.exists(args.fn):
            raise Exception("The --fn that you specifed does not exist.")

    if not args.montage_file is None:
        if not path.exists(args.montage_file):
            raise Exception("The --montage_file that you specifed does not exist.")
        elif not args.montage_file[len(args.montage_file) - 4:] == ".txt":
            raise Exception("The --montage_file must be a .txt file.")

    if not args.predictions_file is None:
        if not path.exists(args.predictions_file):
            raise Exception("The --predictions_file that you specifed does not exist.")
        elif not args.montage_file[len(args.montage_file) - 3:] == ".pt":
            raise Exception("The --predictions_file must be a .pt file.")

    if not args.line_thickness is None:
        if args.line_thickness < 0.1 or args.line_thickness > 3:
            raise Exception("Please choose a line thickness between 0.1 and 3.")

    if not args.font_size is None:
        if args.font_size < 5 or args.line_thickness > 20:
            raise Exception("Please choose a font size between 5 and 20.")

    if not args.save_edf_fn is None:
        if not mandatory_args.issubset(set(dir(args))):
            raise Exception(("You're missing essential arguments!"))


if __name__ == '__main__':
    args = get_args()
    check_args(args)
    app = QApplication(sys.argv)
    ex = MainPage(args, app)
    sys.exit(app.exec_())
