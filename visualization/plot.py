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
from signalStats_info import SignalStatsInfo

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
import mne
import math

from PyQt5.QtCore import Qt, QTime, QUrl

from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QMenu,
                             QVBoxLayout, QSizePolicy, QMessageBox, QWidget,
                             QPushButton, QCheckBox, QLabel, QInputDialog,
                             QSlider, QGridLayout, QDockWidget, QListWidget,
                             QStatusBar, QListWidgetItem, QLineEdit, QSpinBox,
                             QTimeEdit, QComboBox, QFrame, QGroupBox, QStyle)
from PyQt5.QtGui import QIcon, QBrush, QColor, QPen, QFont, QDesktopServices
import pyqtgraph as pg
from pyqtgraph.dockarea import *
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
        self.width = sizeObject.width() * 0.9
        self.height = sizeObject.height() * 0.9
        self.app = app
        self.initUI()

    def initUI(self):
        """
        Setup the UI
        """
        self.app.setStyleSheet(open('visualization/ui_files/gui_stylesheet.css').read())
        layout = QGridLayout()
        layout.setSpacing(10)
        grid_lt = QGridLayout()
        self.grid_rt = QGridLayout()

        #---- left side of the screen ----#

        self.buttonSelectFile = QPushButton('Select file', self)
        self.buttonSelectFile.setToolTip('Click to select EDF file')
        grid_lt.addWidget(self.buttonSelectFile, 0, 0, 1, 2)

        self.lblFn = QLabel("No file loaded.",self)
        grid_lt.addWidget(self.lblFn, 1, 0, 1, 2)

        self.buttonChgSig = QPushButton("Change signals", self)
        self.buttonChgSig.setToolTip("Click to change signals")
        grid_lt.addWidget(self.buttonChgSig, 2, 1)

        self.cbox_filter = QCheckBox("Filter signals", self)
        self.cbox_filter.setToolTip("Click to filter")
        grid_lt.addWidget(self.cbox_filter, 3, 0)

        self.buttonChgFilt = QPushButton("Change Filter", self)
        self.buttonChgFilt.setToolTip("Click to change filter")
        grid_lt.addWidget(self.buttonChgFilt, 3, 1)

        test01 = QLabel("", self)
        grid_lt.addWidget(test01, 4, 0)

        grid_lt.addWidget(QHLine(), 5, 0, 1, 2)

        test0 = QLabel("", self)
        grid_lt.addWidget(test0, 6, 0)

        self.buttonPredict = QPushButton("Load model / predictions", self)
        self.buttonPredict.setToolTip("Click load data, models, and predictions")
        grid_lt.addWidget(self.buttonPredict, 7, 0, 1, 2)

        self.predLabel = QLabel("", self)
        grid_lt.addWidget(self.predLabel, 8, 0, 1, 1)

        self.threshLblVal = QLabel("Change threshold of prediction:  (threshold = 0.5)", self)
        grid_lt.addWidget(self.threshLblVal, 9, 0,1,2)


        self.threshSlider = QSlider(Qt.Horizontal, self)
        self.threshSlider.setMinimum(0)
        self.threshSlider.setMaximum(100)
        self.threshSlider.setValue(50)
        grid_lt.addWidget(self.threshSlider, 10, 0, 1, 2)

        test = QLabel("", self)
        grid_lt.addWidget(test, 11, 0)

        grid_lt.addWidget(QHLine(), 12, 0, 1, 2)

        test11 = QLabel("", self)
        grid_lt.addWidget(test11, 13, 0)

        self.btnZoom = QPushButton("Open zoom", self)
        self.btnZoom.setToolTip("Click to open the zoom window")
        grid_lt.addWidget(self.btnZoom, 15, 0)

        self.buttonChgSpec = QPushButton("Power spectrum", self)
        self.buttonChgSpec.setToolTip("Click to plot the spectrogram of a signal")
        grid_lt.addWidget(self.buttonChgSpec, 15, 1)

        labelAmp = QLabel("Change amplitude:", self)
        grid_lt.addWidget(labelAmp, 16, 0)

        self.buttonAmpInc = QPushButton("+", self)
        self.buttonAmpInc.setToolTip("Click to increase signal amplitude")
        grid_lt.addWidget(self.buttonAmpInc, 16, 1)

        self.buttonAmpDec = QPushButton("-", self)
        self.buttonAmpDec.setToolTip("Click to decrease signal amplitude")
        grid_lt.addWidget(self.buttonAmpDec, 17, 1)

        labelWS = QLabel("Window size:", self)
        grid_lt.addWidget(labelWS, 18, 0)

        self.wsComboBox = QComboBox()
        self.wsComboBox.addItems(["1s","5s","10s","15s","20s","25s","30s"])
        self.wsComboBox.setCurrentIndex(2)
        grid_lt.addWidget(self.wsComboBox, 18, 1)


        test2 = QLabel("", self)
        grid_lt.addWidget(test2, 20, 0)

        grid_lt.addWidget(QHLine(), 21, 0, 1, 2)

        test11 = QLabel("", self)
        grid_lt.addWidget(test11, 22, 0)

        self.buttonPrint = QPushButton("Export to .png", self)
        self.buttonPrint.setToolTip("Click to print a copy of the graph")
        grid_lt.addWidget(self.buttonPrint, 23, 0)

        self.buttonSaveEDF = QPushButton("Save to .edf", self)
        self.buttonSaveEDF.setToolTip("Click to save current signals to an .edf file")
        grid_lt.addWidget(self.buttonSaveEDF, 23, 1)

        test3 = QLabel("", self)
        grid_lt.addWidget(test3, 24, 0)
        test4 = QLabel("", self)
        grid_lt.addWidget(test4, 25, 0)
        test5 = QLabel("", self)
        grid_lt.addWidget(test5, 26, 0)
        test6 = QLabel("", self)
        grid_lt.addWidget(test6, 27, 0)

        self.btnHelp = QPushButton('Help')
        self.btnHelp.setIcon(self.style().standardIcon(getattr(QStyle, 'SP_TitleBarContextHelpButton')))
        grid_lt.addWidget(self.btnHelp,28,0)

        #---- end left side ----#


        #---- Right side of the screen ----#
        self.plotLayout = pg.GraphicsLayoutWidget()
        self.mainPlot = self.plotLayout.addPlot(row=0, col=0)
        self.mainPlot.setMouseEnabled(x=False, y=False)
        self.plotLayout.setBackground('w')
        # self.grid_rt.addWidget(self.plotLayout,0,0,6,8)
        self.plot_area = DockArea()
        self.main_dock = Dock("Main plot", size=(500,200))
        self.main_dock.hideTitleBar()
        self.topoplot_dock = Dock("Topoplot", size=(250,200))
        self.m = PlotCanvas(self, width=7, height=7)
        # self.topoplot_dock.hide()
        self.topoplot_dock.addWidget(self.m)
        self.plot_area.addDock(self.main_dock, 'left')     ## place d4 at right edge of dock area
        self.plot_area.addDock(self.topoplot_dock, 'right', self.main_dock)
        self.topoplot_dock.hide()
        self.main_dock.addWidget(self.plotLayout)
        self.grid_rt.addWidget(self.plot_area,0,0,6,8)


        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(3000)
        self.slider.setValue(0)
        # self.slider.setTickPosition(QSlider.TicksBelow)
        # self.slider.setTickInterval(100)
        self.grid_rt.addWidget(self.slider, 6, 0, 1, 8)

        self.btnOpenAnnDock = QPushButton("Statistics / annotations", self)
        self.btnOpenAnnDock.setToolTip("Click to open annotations dock")
        self.grid_rt.addWidget(self.btnOpenAnnDock, 7, 0)
        self.btnOpenAnnDock.hide()

        self.buttonLt10s = QPushButton("<10", self)
        self.buttonLt10s.setToolTip("Click to go back")
        self.grid_rt.addWidget(self.buttonLt10s, 7, 1)

        self.buttonLt1s = QPushButton("<<1", self)
        self.buttonLt1s.setToolTip("Click to go back")
        self.grid_rt.addWidget(self.buttonLt1s, 7, 2)

        self.buttonChgCount = QPushButton("Jump to...", self)
        self.buttonChgCount.setToolTip("Click to select time for graph")
        self.grid_rt.addWidget(self.buttonChgCount, 7, 3, 1, 2)

        self.buttonRt1s = QPushButton("1>>", self)
        self.buttonRt1s.setToolTip("Click to advance")
        self.grid_rt.addWidget(self.buttonRt1s, 7, 5)

        self.buttonRt10s = QPushButton("10>", self)
        self.buttonRt10s.setToolTip("Click to advance")
        self.grid_rt.addWidget(self.buttonRt10s, 7, 6)

        self.time_lbl = QLabel("0:00:00", self)
        self.grid_rt.addWidget(self.time_lbl, 7, 7)

        #---- Right side dock ----#
        self.dockWidth = self.width * 0.23

        # Annotation dock
        self.scroll = QDockWidget()
        self.btnOpenEditAnn = QPushButton("Open annotation editor", self)
        self.btnOpenEditAnn.setToolTip("Click to open annotation editor")
        self.scroll.setTitleBarWidget(self.btnOpenEditAnn)
        self.ann_qlist = QListWidget()
        self.scroll.setWidget(self.ann_qlist)
        self.scroll.setFloating(False)
        self.scroll.setFixedWidth(self.dockWidth)

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
        self.annTimeEditCount = QSpinBox(self)
        self.annDuration = QSpinBox(self)
        self.btnAnnEdit = QPushButton("Update", self)
        self.btnAnnEdit.setToolTip("Click to modify selected annotation")
        self.btnAnnDel = QPushButton("Delete", self)
        self.btnAnnDel.setToolTip("Click to delete selected annotation")
        self.btnAnnCreate = QPushButton("Create", self)
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
        self.annEditMainWidget.setFixedWidth(self.dockWidth)
        self.annEditDock.setWidget(self.annEditMainWidget)

        # Stats dock
        self.statsDock = QDockWidget()
        self.btnOpenStats = QPushButton("Open signal stats", self)
        self.btnOpenStats.setToolTip("Click to open stats")
        self.statsDock.setTitleBarWidget(self.btnOpenStats)
        self.statsDock.setFixedWidth(self.dockWidth)
        
        """
        self.gridLayout = QtWidgets.QGridLayout()
        ud = 0
        all_lbl = QtWidgets.QLabel(self)
        all_lbl.setText("Total")
        self.gridLayout.addWidget(all_lbl,ud,2,1,1)
        region_lbl = QtWidgets.QLabel(self)
        region_lbl.setText("Region")
        self.gridLayout.addWidget(region_lbl,ud,3,1,1)
        ud += 1
        mean_l = QtWidgets.QLabel(self)
        mean_l.setText("")
        self.mean_lbl = QtWidgets.QLabel(self)
        self.mean_lbl.setText("")
        self.gridLayout.addWidget(self.mean_lbl, ud, 1, 1, 2)
        ud += 1
        self.var_lbl = QtWidgets.QLabel(self)
        self.var_lbl.setText("Var: ")
        self.gridLayout.addWidget(self.var_lbl, ud, 1, 1, 2)
        ud += 1
        self.line_len_lbl = QtWidgets.QLabel(self)
        self.line_len_lbl.setText("Line length: ")
        self.gridLayout.addWidget(self.line_len_lbl, ud, 1, 1, 2)
        ud += 1
        self.mean_sel_lbl = QtWidgets.QLabel(self)
        self.mean_sel_lbl.setText("Region mean: ")
        self.gridLayout.addWidget(self.mean_sel_lbl, ud, 1, 1, 2)
        ud += 1
        self.var_sel_lbl = QtWidgets.QLabel(self)
        self.var_sel_lbl.setText("Region var: ")
        self.gridLayout.addWidget(self.var_sel_lbl, ud, 1, 1, 2)
        ud += 1
        self.line_len_sel_lbl = QtWidgets.QLabel(self)
        self.line_len_sel_lbl.setText("Region line length: ")
        self.gridLayout.addWidget(self.line_len_sel_lbl, ud, 1, 1, 2)
        ud += 1
        self.gridLayout.addWidget(QHLine(), ud, 1, 1, 2)
        ud += 1
        self.alpha_lbl = QtWidgets.QLabel(self)
        self.alpha_lbl.setText("Alpha power: ")
        self.gridLayout.addWidget(self.alpha_lbl, ud, 1, 1, 2)
        ud += 1
        self.beta_lbl = QtWidgets.QLabel(self)
        self.beta_lbl.setText("Beta power: ")
        self.gridLayout.addWidget(self.beta_lbl, ud, 1, 1, 2)
        ud += 1
        self.gamma_lbl = QtWidgets.QLabel(self)
        self.gamma_lbl.setText("Gamma power: ")
        self.gridLayout.addWidget(self.gamma_lbl, ud, 1, 1, 2)
        ud += 1
        self.delta_lbl = QtWidgets.QLabel(self)
        self.delta_lbl.setText("Delta power: ")
        self.gridLayout.addWidget(self.delta_lbl, ud, 1, 1, 2)
        ud += 1
        self.theta_lbl = QtWidgets.QLabel(self)
        self.theta_lbl.setText("Theta power: ")
        self.gridLayout.addWidget(self.theta_lbl, ud, 1, 1, 2)
        ud += 1
        self.qscroll = QtWidgets.QScrollArea(self)
        self.qscroll.setWidgetResizable(True)
        self.qscroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.qscroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.chn_qlist = QListWidget()
        self.qscroll.setWidget(self.chn_qlist)
        self.gridLayout.addWidget(self.qscroll, 0, 0, ud, 1)
        """
        ### playing with UI ####
        self.gridLayout = QtWidgets.QGridLayout()
        ud = 0
        all_lbl = QtWidgets.QLabel(self)
        all_lbl.setText("Overall")
        self.gridLayout.addWidget(all_lbl,ud,2,1,1)
        region_lbl = QtWidgets.QLabel(self)
        region_lbl.setText("Region")
        self.gridLayout.addWidget(region_lbl,ud,3,1,1)
        ud += 1
        mean_l = QtWidgets.QLabel(self)
        mean_l.setText("Mean:")
        self.gridLayout.addWidget(mean_l, ud, 1, 1, 1)
        self.mean_lbl = QtWidgets.QLabel(self)
        self.mean_lbl.setText("")
        self.gridLayout.addWidget(self.mean_lbl, ud, 2, 1, 1)
        self.mean_sel_lbl = QtWidgets.QLabel(self)
        self.mean_sel_lbl.setText("")
        self.gridLayout.addWidget(self.mean_sel_lbl, ud, 3, 1, 1)
        ud += 1
        var_l = QtWidgets.QLabel(self)
        var_l.setText("Var:")
        self.gridLayout.addWidget(var_l, ud, 1, 1, 1)
        self.var_lbl = QtWidgets.QLabel(self)
        self.var_lbl.setText("")
        self.gridLayout.addWidget(self.var_lbl, ud, 2, 1, 1)
        self.var_sel_lbl = QtWidgets.QLabel(self)
        self.var_sel_lbl.setText("")
        self.gridLayout.addWidget(self.var_sel_lbl, ud, 3, 1, 2)
        ud += 1
        line_len_l = QtWidgets.QLabel(self)
        line_len_l.setText("Line\nlength:")
        self.gridLayout.addWidget(line_len_l, ud, 1, 1, 1)
        self.line_len_lbl = QtWidgets.QLabel(self)
        self.line_len_lbl.setText("")
        self.gridLayout.addWidget(self.line_len_lbl, ud, 2, 1, 1)
        self.line_len_sel_lbl = QtWidgets.QLabel(self)
        self.line_len_sel_lbl.setText("")
        self.gridLayout.addWidget(self.line_len_sel_lbl, ud, 3, 1, 1)
        ud += 1
        self.gridLayout.addWidget(QHLine(), ud, 1, 1, 3)
        ud += 1
        alpha_l = QtWidgets.QLabel(self)
        alpha_l.setText("Alpha:")
        self.gridLayout.addWidget(alpha_l, ud, 1, 1,1)
        self.alpha_lbl = QtWidgets.QLabel(self)
        self.alpha_lbl.setText("")
        self.gridLayout.addWidget(self.alpha_lbl, ud, 2, 1, 1)
        self.alpha_sel_lbl = QtWidgets.QLabel(self)
        self.alpha_sel_lbl.setText("")
        self.gridLayout.addWidget(self.alpha_sel_lbl, ud, 3, 1, 1)
        ud += 1
        beta_l = QtWidgets.QLabel(self)
        beta_l.setText("Beta:")
        self.gridLayout.addWidget(beta_l, ud, 1, 1, 1)
        self.beta_lbl = QtWidgets.QLabel(self)
        self.beta_lbl.setText("")
        self.gridLayout.addWidget(self.beta_lbl, ud, 2, 1, 1)
        self.beta_sel_lbl = QtWidgets.QLabel(self)
        self.beta_sel_lbl.setText("")
        self.gridLayout.addWidget(self.beta_sel_lbl, ud, 3, 1, 2)
        ud += 1
        gamma_l = QtWidgets.QLabel(self)
        gamma_l.setText("Gamma:")
        self.gridLayout.addWidget(gamma_l, ud, 1, 1, 1)
        self.gamma_lbl = QtWidgets.QLabel(self)
        self.gamma_lbl.setText("")
        self.gridLayout.addWidget(self.gamma_lbl, ud, 2, 1, 1)
        self.gamma_sel_lbl = QtWidgets.QLabel(self)
        self.gamma_sel_lbl.setText("")
        self.gridLayout.addWidget(self.gamma_sel_lbl, ud, 3, 1, 1)
        ud += 1
        delta_l = QtWidgets.QLabel(self)
        delta_l.setText("Delta:")
        self.gridLayout.addWidget(delta_l, ud, 1, 1, 1)
        self.delta_lbl = QtWidgets.QLabel(self)
        self.delta_lbl.setText("")
        self.gridLayout.addWidget(self.delta_lbl, ud, 2, 1, 1)
        self.delta_sel_lbl = QtWidgets.QLabel(self)
        self.delta_sel_lbl.setText("")
        self.gridLayout.addWidget(self.delta_sel_lbl, ud, 3, 1, 1)
        ud += 1
        theta_l = QtWidgets.QLabel(self)
        theta_l.setText("Theta:")
        self.gridLayout.addWidget(theta_l, ud, 1, 1, 1)
        self.theta_lbl = QtWidgets.QLabel(self)
        self.theta_lbl.setText("")
        self.gridLayout.addWidget(self.theta_lbl, ud, 2, 1, 1)
        self.theta_sel_lbl = QtWidgets.QLabel(self)
        self.theta_sel_lbl.setText("")
        self.gridLayout.addWidget(self.theta_sel_lbl, ud, 3, 1, 1)
        ud += 1
        self.qscroll = QtWidgets.QScrollArea(self)
        self.qscroll.setWidgetResizable(True)
        self.qscroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.qscroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.chn_qlist = QListWidget()
        self.qscroll.setWidget(self.chn_qlist)
        self.gridLayout.addWidget(self.qscroll, 0, 0, ud, 1)

        ### end playing with UI ####
        
        self.statsMainWidget = QWidget()
        self.statsMainWidget.setLayout(self.gridLayout)
        self.statsDock.setWidget(self.statsMainWidget)

        self.scroll.hide()
        self.annEditDock.hide()
        self.statsDock.hide()
        self.addDockWidget(Qt.RightDockWidgetArea, self.scroll)
        self.addDockWidget(Qt.RightDockWidgetArea, self.annEditDock)
        self.splitDockWidget(self.annEditDock, self.scroll, Qt.Vertical)
        self.addDockWidget(Qt.RightDockWidgetArea, self.statsDock)
        self.splitDockWidget(self.scroll, self.statsDock, Qt.Vertical)
        
        #---- end right side dock ----#

        #---- end right side of screen ----#

        wid = QWidget(self)
        self.setCentralWidget(wid)
        wid.setLayout(layout)
        layout.addLayout(grid_lt, 0, 0, 3, 1)
        layout.addLayout(self.grid_rt, 0, 1, 4, 3)

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.setSignalsSlots()
        self.init_values()

        if self.argv.show:
            self.show()
            if not self.argv.fn is None:
                self.load_data(self.argv.fn)
        else:
            fn = self.argv.fn
            self.argv_pred_fn = self.argv.predictions_file
            self.argv_mont_fn = self.argv.montage_file
            self.load_data(fn)

    def setSignalsSlots(self):
        """ Sets signals and slots for the main window. 
        """
        # ---- left side of the screen ---- #
        self.buttonSelectFile.clicked.connect(self.load_data)
        self.buttonChgSig.clicked.connect(self.chgSig)
        self.cbox_filter.toggled.connect(self.filterChecked)
        self.buttonChgFilt.clicked.connect(self.changeFilter)
        self.buttonPredict.clicked.connect(self.changePredictions)
        self.threshSlider.sliderReleased.connect(self.changeThreshSlider)
        self.btnZoom.clicked.connect(self.openZoomPlot)
        self.buttonChgSpec.clicked.connect(self.loadSpec)
        self.buttonAmpInc.clicked.connect(self.incAmp)
        self.buttonAmpDec.clicked.connect(self.decAmp)
        self.wsComboBox.currentIndexChanged['int'].connect(self.chgWindow_size)
        self.buttonPrint.clicked.connect(self.print_graph)
        self.buttonSaveEDF.clicked.connect(self.save_to_edf)
        self.btnHelp.clicked.connect(self.openHelp)

        # ---- right side of the screen ---- #
        self.slider.sliderReleased.connect(self.valuechange)
        self.btnOpenAnnDock.clicked.connect(self.openAnnDock)
        self.buttonLt10s.clicked.connect(self.leftPlot10s)
        self.buttonLt1s.clicked.connect(self.leftPlot1s)
        self.buttonChgCount.clicked.connect(self.getCount)
        self.buttonRt1s.clicked.connect(self.rightPlot1s)
        self.buttonRt10s.clicked.connect(self.rightPlot10s)

        # ---- right side dock ---- #
        self.btnOpenEditAnn.clicked.connect(self.openAnnEditor)
        self.ann_qlist.itemClicked.connect(self.ann_clicked)
        self.annTimeEditTime.timeChanged.connect(self.updateCountTime)
        self.annTimeEditCount.valueChanged.connect(self.updateNormalTime)
        self.btnAnnEdit.clicked.connect(self.annEditorUpdate)
        self.btnAnnDel.clicked.connect(self.annEditorDel)
        self.btnAnnCreate.clicked.connect(self.annEditorCreate)
        self.btnOpenStats.clicked.connect(self.openStatWindow)
        self.chn_qlist.itemClicked.connect(self.statChnClicked)

    def init_values(self):
        """ Set some initial values and create Info objects. 
        """
        self.count = 0  # the current location in time we are plotting
        self.init = 0  # if any data has been loaded in yet
        self.window_size = 10  # number of seconds to display at a time
        self.filter_checked = 0  # whether or not to plot filtered data
        self.ylim = [150, 100]  # ylim for unfiltered and filtered data
        self.max_channels = 30 # maximum channels you can plot at once
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
        self.ssi = SignalStatsInfo() # holds info for stats window

    def closeEvent(self, event):
        """ Called when the main window is closed to act as a destructor and close
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
        """ Function to properly initialize everything when new data
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

        if not self.topoplot_dock.isHidden():
            self.close_topoplot()
        
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

    def ann_clicked(self):
        """ Moves the plot when annotations in the dock are clicked.
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

    def openAnnEditor(self):
        """ Create and open the annotation editor.
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
                self.showAnnStatsDock()

    def openAnnDock(self):
        """ Opens the annotation and stats dock when the button below
            plot is clicked.
        """
        self.scroll.show()
        self.statsDock.show()
        self.statsMainWidget.hide()
        self.btnOpenAnnDock.hide()
        self.openAnnEditor()

    def populateAnnDock(self):
        """ Fills the annotation dock with annotations if they exist.
        """
        self.ann_qlist.clear()
        ann = self.edf_info.annotations
        if len(ann[0]) > 0:
            for i in range(len(ann[0])):
                self.ann_qlist.addItem(ann[2][i])
    
    def showAnnStatsDock(self):
        """ Properly show the stats and annotation dock. 
        """
        ann = self.edf_info.annotations
        if len(ann[0]) == 0:
            self.annEditDock.hide()
            if self.btnOpenStats.text() == "Open signal stats":
                self.scroll.hide()
                self.statsDock.hide()
                self.btnOpenAnnDock.show()
            else:
                self.scroll.show()
                self.btnOpenAnnDock.hide()
        else:
            self.scroll.show()
            self.statsDock.show()
            self.statsMainWidget.hide()
            self.btnOpenAnnDock.hide()

    def annEditorUpdate(self):
        """ Called when the update annotation button is pressed. 
        """
        annTxt = self.annTxtEdit.text()
        loc = self.annTimeEditCount.value()
        dur = self.annDuration.value()
        self.edf_info.annotations[0][self.ann_qlist.currentRow()] = loc
        self.edf_info.annotations[1][self.ann_qlist.currentRow()] = dur
        self.edf_info.annotations[2][self.ann_qlist.currentRow()] = annTxt
        self.populateAnnDock()
        self.callmovePlot(0,0)

    def annEditorDel(self):
        """ Called when the delete selected annotation button is pressed. 
        """
        self.edf_info.annotations = np.delete(self.edf_info.annotations,self.ann_qlist.currentRow(),axis = 1)
        self.btnAnnEdit.setEnabled(False)
        self.btnAnnDel.setEnabled(False)
        # self.annEditDock.hide()
        self.populateAnnDock()
        #if len(self.edf_info.annotations[0]) == 0:
        #    self.scroll.show()
        #    self.btnOpenAnnDock.hide()
        #self.annEditDock.show()
        self.callmovePlot(0,0)

    def annEditorCreate(self):
        """ Called when the create new annotation button is pressed. 
        """
        annTxt = self.annTxtEdit.text()
        if len(annTxt) > 0:
            self.annTxtEdit.setText("")
            loc = self.annTimeEditCount.value()
            dur = self.annDuration.value()
            i = 0
            while i < len(self.edf_info.annotations[0]):
                if int(float(self.edf_info.annotations[0][i])) > loc:
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
            #name = QFileDialog.getOpenFileName(
            #    self, 'Open file', '.', 'EDF files (*.edf)')
            #name = name[0]
            name = "/Users/daniellecurrey/Desktop/GUI/Random_edf_files/00013145_s004_t002.edf"
        if name == None or len(name) == 0:
            return
        else:
            self.edf_file_name_temp = name
            loader = EdfLoader()
            try:
                self.edf_info_temp = loader.load_metadata(name)
            except:
                self.throwAlert("The .edf file is invalid.")
                return
            self.edf_info_temp.annotations = np.array(
                self.edf_info_temp.annotations)

            # edf_montages = EdfMontage(self.edf_info_temp)
            # fs_idx = edf_montages.getIndexForFs(self.edf_info_temp.labels2chns)

            # self.data_temp = loader.load_buffers(self.edf_info_temp)
            # data_for_preds = self.data_temp
            # self.edf_info_temp.fs, self.data_temp = loadSignals(
            #    self.data_temp, self.edf_info_temp.fs)
            try:
                if len(self.edf_info_temp.fs) > 1:
                    self.edf_info_temp.fs = np.max(self.edf_info_temp.fs)
                elif len(self.edf_info_temp.fs) == 1:
                    self.edf_info_temp.fs = self.edf_info_temp.fs[0]
            except:
                pass

            # setting temporary variables that will be overwritten if
            # the user selects signals to plot
            self.max_time_temp = int(
                self.edf_info_temp.nsamples[0] / self.edf_info_temp.fs)
            self.ci_temp = ChannelInfo()  # holds channel information
            self.ci_temp.chns2labels = self.edf_info_temp.chns2labels
            self.ci_temp.labels2chns = self.edf_info_temp.labels2chns
            self.ci_temp.fs = self.edf_info_temp.fs
            self.ci_temp.max_time = self.max_time_temp
            self.ci_temp.edf_fn = name
            self.fn_full_temp = name
            if len(name.split('/')[-1]) < 40:
                self.fn_temp = name.split('/')[-1]
            else:
                self.fn_temp = name.split('/')[-1][0:37] + "..."

            self.chn_win_open = 1
            self.predicted = 0  # whether or not predictions have been made
            self.chn_ops = ChannelOptions(self.ci_temp, self)
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
        self.showAnnStatsDock()

        nchns = self.ci.nchns_to_plot
        self.plot_lines = []
        if not self.init and self.argv.location < self.max_time - self.window_size:
            self.count = self.argv.location

        ann = self.edf_info.annotations
        print(self.pi.pred_by_chn)
        print(self.predicted)
        if self.pi.pred_by_chn and self.predicted:
            self.add_topoplot()
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
        elif right == 1 and self.count + num_move + self.window_size <= self.ci.data_to_plot.shape[1] / fs:#self.data.shape[1] / fs:
            self.count = self.count + num_move
        self.slider.setValue(self.count)
        t = getTime(self.count)
        self.time_lbl.setText(t)

        # update the topoplot
        if not self.topoplot_dock.isHidden():
            self.update_topoplot()

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
        if self.btnOpenStats.text() == "Close signal stats":
            self.statTimeSelectChanged()

        if self.init == 0 and self.argv.show:
            self.throwAlert("Data has been plotted.")

    def close_topoplot(self):
        """ Function to close the topoplot. 
        """
        self.topoplot_dock.hide()

    def add_topoplot(self):
        """ Function called when pred options loads and pred_by_chn == 1
        """
        self.topoplot_dock.show()
        self.update_topoplot()

    def update_topoplot(self):
        """ Update the topoplot if pred_by_chn == 1
        """
        # TODO: make window be able to open and close
        # TODO: create plot axes
        # TODO: use mne to create plot

        # How will this work?
        # 1) create some fake data for the file, say 1s each
        # 2) once you load it in, plot for each second of data
        # 3) assume it is by second (but should probably make this custom)
        # 4) set this up so this plot opens when predictions are loaded
        # 5) consider making this custom so that it can be toggled on / off by user

        #def topoplot(scores, label_list, title=None, fn='',
        #     plot_hemisphere=False, plot_lobe=False, zone=None,
        #     lobe_correct=None, lat_correct=None):
        # open dock window
        # self.topoplot_dock.show()

        # clear figure
        self.m.fig.clf()
        scores = self.pi.preds_to_plot
        #scores = np.zeros((73,18))
        #for i in range(len(scores[0])):
        #    for j in range(len(scores[1])):
        #        scores[i][j] = np.random.random()
        

        curr_score = scores[self.count,:]
        #for i in range(len(curr_score)):
        #    curr_score[i] = np.random.random()

        # Create the layout
        layout = mne.channels.read_layout('EEG1005')
        # positions = []
        pos2d = []
        layout_names = [name.upper() for name in layout.names]
        for ch in self.ci.labels_to_plot:
            if ch != "Notes":
                if '-' in ch:
                    anode, cathode = ch.split('-')
                    anode_idx = layout_names.index(anode)
                    cathode_idx = layout_names.index(cathode)
                    anode_pos = layout.pos[anode_idx, 0:2]
                    cathode_pos = layout.pos[cathode_idx, 0:2]
                    pos2d.append([(a + c) / 2 for a, c in zip(anode_pos, cathode_pos)])
                else:
                    idx = layout_names.index(ch)
                    # positions.append(layout.pos[idx, :])
                    pos2d.append(layout.pos[idx, 0:2])
        # positions = np.asarray(positions)
        pos2d = np.asarray(pos2d)
        # Scale locations from [-1, 1]
        pos2d = 2 * (pos2d - 0.5)

        # fig = plt.figure()
        #ax = plt.gca()
        # self.ax = self.m.fig.gca()
        self.ax = self.m.fig.add_subplot(self.m.gs[0])
        im, cn = mne.viz.plot_topomap(curr_score, pos2d, sphere=1,
                                  axes=self.ax, vmin=0, vmax=1, show=False,
                                  outlines='head')
        self.m.draw()



    def filterChecked(self):
        """ Function for when the filterbox is checked

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
            # if you start / stop filtering, need to update the stats
            if self.btnOpenStats.text() == "Close signal stats":
                self.statChnClicked()
            # if data was already filtered do not uncheck box
            ann = self.edf_info.annotations
            if len(ann[0]) > 0 and ann[2][0] == "filtered":
                self.filter_checked = 1
                cbox.setChecked(True)
            self.callmovePlot(1, 0)
        elif self.init == 0 and cbox.isChecked():
            cbox.setChecked(False)

    def prep_filter_ws(self):
        """ Does filtering for one window of size window_size
        """
        fs = self.edf_info.fs
        if len(self.filteredData) == 0 or self.filteredData.shape != self.ci.data_to_plot[:,self.count*fs:(self.count + self.window_size)*fs].shape:
            self.filteredData = np.zeros((self.ci.nchns_to_plot,self.window_size * fs))
        filt_window_size = filterData(
            self.ci.data_to_plot[:, self.count * fs:(self.count + self.window_size)*fs], fs, self.fi)
        filt_window_size = np.array(filt_window_size)
        self.filteredData = filt_window_size

    def changeFilter(self):
        """ Opens the FilterOptions window
        """
        if self.init == 1:
            self.filter_win_open = 1
            self.filter_ops = FilterOptions(self.fi, self)
            self.filter_ops.show()

    def changePredictions(self):
        """ Take loaded model and data and compute predictions
        """
        if self.init == 1:
            self.preds_win_open = 1
            self.pred_ops = PredictionOptions(self.pi, self)
            self.pred_ops.show()

    def makeSpecPlot(self):
        """ Creates the spectrogram plot.
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
        """ Function called when the user changes the region that selects where in
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
        """ Updates spectrogram plot. 
        """
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
        """ Removes the spectrogram plot.
        """
        self.plotLayout.removeItem(self.specPlot)
        # self.plotLayout.removeItem(self.hist)
        self.mainPlot.removeItem(self.selectTimeRect)

    def loadSpec(self):
        """ Opens the SpecOptions window
        """
        if self.init == 1:
            if self.btnZoom.text() == "Close zoom":
                self.throwAlert("Please close the zoom plot before opening the spectrogram.")
            else:
                self.spec_win_open = 1
                self.spec_ops = SpecOptions(self.si, self)
                self.spec_ops.show()

    def openStatWindow(self):
        """ Opens the statistics window in the sidebar. 
        """
        if self.btnOpenStats.text() == "Open signal stats":
            self.btnOpenStats.setText("Close signal stats")
            self.statsMainWidget.show()
            self.populateStatList()
            self.chn_qlist.setCurrentRow(0)
            self.ssi.chn = 0
            self.ssi.fs = self.edf_info.fs
            self.createStatSelectTimeRect(self.ssi.chn)
            self.statChnClicked()
        else:
            self.btnOpenStats.setText("Open signal stats")
            self.removeStatSelectTimeRect()
            self.statsMainWidget.hide()
            if self.btnOpenEditAnn.text() == "Open annotation editor":
                self.populateAnnDock()
                self.showAnnStatsDock()
    
    def populateStatList(self):
        """ Fill the stats window with channels.
        """
        # Remove old channels if they exist
        self.chn_qlist.clear()
        chns = self.ci.labels_to_plot
        self.ssi.chn_items = []
        for i in range(1, len(chns)):
            self.ssi.chn_items.append(QListWidgetItem(chns[i], self.chn_qlist))
            self.chn_qlist.addItem(self.ssi.chn_items[i - 1])

    def statChnClicked(self):
        """ When a channel is clicked.
        """
        self.removeStatSelectTimeRect()
        self.ssi.chn = self.chn_qlist.currentRow()
        self.createStatSelectTimeRect(self.ssi.chn)
        mean_str, var_str, line_len_str = self.get_stats(0,self.max_time * self.edf_info.fs)

        mean_str = "" + "{:.2f}".format(mean_str)
        self.mean_lbl.setText(mean_str)

        var_str = "" + "{:.2f}".format(var_str)
        self.var_lbl.setText(var_str)

        line_len_str = "" + "{:.2f}".format(line_len_str)
        self.line_len_lbl.setText(line_len_str)
        self.set_fs_band_lbls()

    def createStatSelectTimeRect(self, chn):
        """ Create the rectangle selector item.
        """
        redBrush = QBrush(QColor(217, 43, 24,50))
        self.statSelectTimeRect = pg.LinearRegionItem(values=(self.edf_info.fs, 4 * self.edf_info.fs),
                        brush=redBrush, movable=True, orientation=pg.LinearRegionItem.Vertical)
        self.statSelectTimeRect.setSpan((chn + 2) / (self.ci.nchns_to_plot + 3),(chn + 3) / (self.ci.nchns_to_plot + 3))
        self.mainPlot.addItem(self.statSelectTimeRect)
        self.statSelectTimeRect.sigRegionChangeFinished.connect(self.statTimeSelectChanged)
        self.statTimeSelectChanged()

    def removeStatSelectTimeRect(self):
        """ Remove the rectangle selector item.
        """
        self.mainPlot.removeItem(self.statSelectTimeRect)

    def statTimeSelectChanged(self):
        """ Called when the stats bar is moved. 
        """
        bounds = self.statSelectTimeRect.getRegion()
        bounds = bounds + self.count * self.edf_info.fs
        mean_str, var_str, line_len_str = self.get_stats(int(bounds[0]), int(bounds[1]))
        mean_str = "" + "{:.2f}".format(mean_str)
        self.mean_sel_lbl.setText(mean_str)
        var_str = "" + "{:.2f}".format(var_str)
        self.var_sel_lbl.setText(var_str)
        line_len_str = "" + "{:.2f}".format(line_len_str)
        self.line_len_sel_lbl.setText(line_len_str)

        alpha, beta, theta, gamma, delta = self.get_power_band_stats(int(bounds[0]), int(bounds[1]))
        alpha_str = "" + "{:.2e}".format(alpha)
        self.alpha_sel_lbl.setText(alpha_str)
        beta_str = "" + "{:.2e}".format(beta)
        self.beta_sel_lbl.setText(beta_str)
        theta_str = "" + "{:.2e}".format(theta)
        self.theta_sel_lbl.setText(theta_str)
        gamma_str = "" + "{:.2e}".format(gamma)
        self.gamma_sel_lbl.setText(gamma_str)
        delta_str = "" + "{:.2e}".format(delta)
        self.delta_sel_lbl.setText(delta_str)

    def get_stats(self, s, f):
        """ Get mean, var, and line length.

        Args:
            chn: the channel to compute stats
            s: start time in samples
            f: end time in samples
        Returns:
            The mean
            var
            line length (for the part of the signal specified)
        """

        if self.filter_checked == 1:
            self.prep_filter_ws()
            array_sum = np.sum(self.filteredData)
            mean_str = self.filteredData[self.ssi.chn,s:f].mean()
            var_str = self.filteredData[self.ssi.chn,s:f].var()
            line_len_str = np.sqrt(np.sum(np.diff(self.filteredData[self.ssi.chn,s:f]) ** 2 + 1))
        else:
            mean_str = self.ci.data_to_plot[self.ssi.chn,s:f].mean()
            var_str = self.ci.data_to_plot[self.ssi.chn,s:f].var()
            line_len_str = np.sqrt(np.sum(np.diff(self.ci.data_to_plot[self.ssi.chn,s:f]) ** 2 + 1))
        
        return mean_str, var_str, line_len_str

    def get_power_band_stats(self, s, f):
        """ Get power band stats

        Args:
            chn: the channel to compute stats
            s: start time in samples
            f: end time in samples
        Returns:
            alpha, beta, gamma, theta, delta 
            (for the part of the signal specified)
        """
        data = self.ci.data_to_plot[self.ssi.chn,:]
        lp = 0
        hp = 0
        if self.filter_checked == 1:
            print("in stat - filter checked")
            lp = self.fi.lp
            hp = self.fi.hp
        alpha, beta, theta, gamma, delta = self.ssi.get_power(data, s, f, hp, lp)
        return alpha, beta, theta, gamma, delta        

    def set_fs_band_lbls(self):
        """ Sets alpha, beta, gamma, delta, theta lbls for stats.
        """
        alpha, beta, theta, gamma, delta = self.get_power_band_stats(0, self.max_time * self.edf_info.fs)
        alpha_str = "" + "{:.2e}".format(alpha)
        self.alpha_lbl.setText(alpha_str)
        beta_str = "" + "{:.2e}".format(beta)
        self.beta_lbl.setText(beta_str)
        theta_str = "" + "{:.2e}".format(theta)
        self.theta_lbl.setText(theta_str)
        gamma_str = "" + "{:.2e}".format(gamma)
        self.gamma_lbl.setText(gamma_str)
        delta_str = "" + "{:.2e}".format(delta)
        self.delta_lbl.setText(delta_str)

    def throwAlert(self, msg):
        """ Throws an alert to the user.
        """
        alert = QMessageBox()
        alert.setIcon(QMessageBox.Information)
        alert.setText(msg)
        # alert.setInformativeText(msg)
        alert.setWindowTitle("Warning")
        alert.exec_()


class QHLine(QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)

class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi,
                          constrained_layout=False)
        self.gs = self.fig.add_gridspec(1, 1, wspace=0.0, hspace=0.0)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

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
            raise Exception("--montage-file must be specified")

    if not args.fn is None and args.montage_file is None:
        raise Exception("--montage-file must be specified if --fn is specified")

    if args.fn is None and not args.montage_file is None:
        raise Exception("--fn must be specified if --montage-file is specified")

    if not args.fn is None:
        if not path.exists(args.fn):
            raise Exception("The --fn that you specifed does not exist.")

    if not args.montage_file is None:
        if not path.exists(args.montage_file):
            raise Exception("The --montage-file that you specifed does not exist.")
        elif not args.montage_file[len(args.montage_file) - 4:] == ".txt":
            raise Exception("The --montage-file must be a .txt file.")

    if not args.predictions_file is None:
        if not path.exists(args.predictions_file):
            raise Exception("The --predictions_file that you specifed does not exist.")
        elif not args.predictions_file[len(args.predictions_file) - 3:] == ".pt":
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
