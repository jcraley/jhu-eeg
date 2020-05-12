from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import (QVBoxLayout, QMessageBox, QWidget, QListWidget,
                                QPushButton, QCheckBox, QLabel, QGridLayout,
                                QScrollArea, QListWidgetItem, QAbstractItemView)

import numpy as np
from preds_info import PredsInfo
from channel_info import ChannelInfo
from organize_channels import OrganizeChannels

class ChannelOptions(QWidget):
    def __init__(self,data,parent,data_for_preds = []):
        super().__init__()
        self.left = 10
        self.top = 10
        self.title = 'Select signals'
        self.width = 300
        self.height = 400
        self.unprocessed_data = data_for_preds
        # if loading new data make copies in case user cancels loading channels
        self.new_load = 0
        if len(self.unprocessed_data) != 0:
            self.pi = PredsInfo()
            self.new_load = 1
        else:
            self.pi = parent.pi
        self.data = data
        self.parent = parent
        self.organize_win_open = 0
        self.setupUI()

    def setupUI(self):

        layout = QGridLayout()
        grid_lt = QGridLayout()
        grid_rt = QGridLayout()

        self.scroll = QScrollArea()
        self.scroll.setMinimumWidth(120)
        self.scroll.setMinimumHeight(200) # would be better if resizable
        self.scroll.setWidgetResizable(True)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.chn_qlist = QListWidget()
        self.chn_qlist.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.scroll.setWidget(self.chn_qlist)

        self.populateChnList()
        self.data.convertedChnNames = []
        self.data.convertChnNames()
        self.ar = self.data.canDoAR()
        bip = self.data.canDoBIP()
        self.data.total_nchns = len(self.data.chns2labels)

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        lblInfo = QLabel("Select channels to plot: ")
        grid_lt.addWidget(lblInfo,0,0)

        if self.ar:
            self.cbox_ar = QCheckBox("Average reference",self)
            self.cbox_ar.toggled.connect(self.arChecked)
            grid_lt.addWidget(self.cbox_ar,1,0)

            self.cbox_bip = QCheckBox("Bipolar",self)
            self.cbox_bip.toggled.connect(self.bipChecked)
            grid_lt.addWidget(self.cbox_bip,2,0)
        elif bip:
            self.cbox_bip = QCheckBox("Bipolar",self)
            self.cbox_bip.toggled.connect(self.bipChecked)
            grid_lt.addWidget(self.cbox_bip,2,0)

        lbl = QLabel("")
        grid_lt.addWidget(lbl, 3,0)

        btnOrganize = QPushButton('Organize', self)
        btnOrganize.clicked.connect(self.organize)
        grid_lt.addWidget(btnOrganize,4,0)

        btnExit = QPushButton('Ok', self)
        btnExit.clicked.connect(self.okayPressed)
        grid_lt.addWidget(btnExit,4,1)

        grid_rt.addWidget(self.scroll,0,1)

        layout.addLayout(grid_lt,0,0)
        layout.addLayout(grid_rt,0,1)
        self.setLayout(layout)

        self.show()

    def populateChnList(self):
        """
        Fills the list with all of the channels in the edf file.
        PREDICTION channels are ignored and saved into self.pi
        """
        chns = self.data.chns2labels
        lbls = self.data.labels2chns
        if len(chns) == 0:
            self.parent.throwAlert("There are no named channels in the file.")
            self.closeWindow()
        else:
            self.chn_items = []
            for i in range(len(chns)):
                if chns[i].find("PREDICTIONS") == -1:
                    self.chn_items.append(QListWidgetItem(chns[i], self.chn_qlist))
                    #self.chn_items[i].setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                    #self.chn_items[i].setCheckState(Qt.Unchecked)
                    self.chn_qlist.addItem(self.chn_items[i])
                elif len(self.unprocessed_data) > 0:
                    self.data.pred_chn_data.append(self.unprocessed_data[i])
                    lbls.pop(chns[i])
                    chns.pop(i)

            if len(self.unprocessed_data) > 0 and len(self.data.pred_chn_data) != 0:
                self.data.pred_chn_data = np.array(self.data.pred_chn_data)
                self.data.pred_chn_data = self.data.pred_chn_data.T
                if self.data.pred_chn_data.shape[1] > 1:
                    self.pi.pred_by_chn = 1
                else:
                    self.data.pred_chn_data = np.squeeze(self.data.pred_chn_data)
                if len(self.data.pred_chn_data) > 0:
                    self.pi.preds = self.data.pred_chn_data
                    self.pi.preds_to_plot = self.data.pred_chn_data
                    self.pi.preds_loaded = 1
                    self.pi.plot_loaded_preds = 1
                    self.pi.preds_fn = "loaded from .edf file"
                    self.pi.pred_width = (self.data.fs * self.data.max_time) / self.pi.preds.shape[0]
                    self.parent.predicted = 1

            if len(self.data.list_of_chns) != 0:
                for k in range(len(self.data.list_of_chns)):
                    self.chn_items[self.data.list_of_chns[k]].setSelected(True)
            self.scroll.show()

    def arChecked(self):
        c = self.sender()
        chns = self.data.getARchns()
        if c.isChecked():
            if self.cbox_bip.isChecked():
                self.cbox_bip.setChecked(0)
            # select all AR channels, deselect all others
            for k in range(len(chns)):
                if chns[k]:
                    self.chn_items[k].setSelected(True)
                else:
                    self.chn_items[k].setSelected(False)
        else:
            for k in range(len(chns)):
                if chns[k]:
                    self.chn_items[k].setSelected(False)

    def bipChecked(self):
        c = self.sender()
        chns = self.data.getBIPchns()
        if self.ar:
            chns = self.data.getARchns()
            if c.isChecked():
                if self.cbox_ar.isChecked():
                    self.cbox_ar.setChecked(0)
                for k in range(len(chns)):
                    if chns[k]:
                        self.chn_items[k].setSelected(True)
                    else:
                        self.chn_items[k].setSelected(False)
            else:
                for k in range(len(chns)):
                    if chns[k]:
                        self.chn_items[k].setSelected(False)
        elif c.isChecked():
            # select all bipolar channels, deselect all others
            for k in range(len(chns)):
                if chns[k]:
                    self.chn_items[k].setSelected(True)
                else:
                    self.chn_items[k].setSelected(False)
        else:
            for k in range(len(chns)):
                if chns[k]:
                    self.chn_items[k].setSelected(True)

    def overwriteTempInfo(self):
        """
        If temporary data was created in case the user cancels loading channels,
        it is now overwritten.

        Things to be overwritten:
            - parent.edf_info
            - parent.data
            - parent.fs
            - parent.max_time
            - parent.pi
            - parent.ci
        """
        self.parent.edf_info = self.parent.edf_info_temp
        self.parent.data = self.parent.data_temp
        self.parent.max_time = self.parent.max_time_temp
        self.parent.pi.write_data(self.pi)
        self.parent.ci.write_data(self.data)
        self.data = self.parent.ci

    def organize(self):
        """
        Function to open the window to change signal order
        """
        if not self.parent.organize_win_open:
            ret = self.check()
            if ret == 0:
                self.parent.organize_win_open = 1
                self.parent.chn_org = OrganizeChannels(self.data, self.parent)
                self.parent.chn_org.show()
                self.closeWindow()

    def check(self):
        """
        Function to check the clicked channels and exit.

        returns:
            -1 if there are no selected channels, 0 otherwise
        """
        selectedListItems = self.chn_qlist.selectedItems()
        idxs = []
        for k in range(len(self.chn_items)):
            if self.chn_items[k] in selectedListItems:
            # if self.chn_items[k].checkState():
                idxs.append(self.data.labels2chns[self.data.chns2labels[k]])
        if len(idxs) == 0:
            self.parent.throwAlert("Please select channels to plot.")
            return -1
        else:
            # Overwrite if needed, and prepare to plot
            if self.new_load:
                self.overwriteTempInfo()
            data = self.parent.data
            plot_bip_from_ar = 0
            if self.ar and self.cbox_bip.isChecked():
                plot_bip_from_ar = 1
            self.data.prepareToPlot(idxs, data, self.parent, plot_bip_from_ar)
        return 0

    def okayPressed(self):
        ret = self.check()
        if ret == 0:
            self.parent.callInitialMovePlot()
            self.closeWindow()

    def closeWindow(self):
        self.parent.chn_win_open = 0
        self.close()

    def closeEvent(self, event):
        """
        Called when the window is closed.
        """
        self.parent.chn_win_open = 0
        event.accept()
