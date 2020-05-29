from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import (QVBoxLayout, QMessageBox, QWidget, QListWidget,
                                QPushButton, QCheckBox, QLabel, QGridLayout,
                                QScrollArea, QListWidgetItem, QAbstractItemView,
                                QFileDialog)

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
        self.width = parent.width / 5
        self.height = parent.height / 2.5
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
        self.bip = self.data.canDoBIP()
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
        elif self.bip:
            self.cbox_bip = QCheckBox("Bipolar",self)
            self.cbox_bip.toggled.connect(self.bipChecked)
            grid_lt.addWidget(self.cbox_bip,1,0)

        self.cbox_txtfile = QCheckBox("",self)
        self.cbox_txtfile.toggled.connect(self.txtFileChecked)
        grid_lt.addWidget(self.cbox_txtfile,3,0)

        self.btn_loadtxtfile = QPushButton("Load text file",self)
        self.btn_loadtxtfile.clicked.connect(self.loadTxtFile)
        grid_lt.addWidget(self.btn_loadtxtfile,3,0)

        self.btn_cleartxtfile = QPushButton("Clear text file",self)
        self.btn_cleartxtfile.clicked.connect(self.clearTxtFile)
        self.btn_cleartxtfile.setVisible(0)
        grid_lt.addWidget(self.btn_cleartxtfile,4,0)

        if len(self.data.txtFile_fn) > 0:
            if self.data.use_loaded_txt_file:
                self.cbox_txtfile.setChecked(1)
            self.btn_loadtxtfile.setVisible(0)
            self.btn_cleartxtfile.setVisible(1)
            self.cbox_txtfile.setVisible(1)
            self.cbox_txtfile.setText(self.data.txtFile_fn)

        lbl = QLabel("")
        grid_lt.addWidget(lbl, 5,0)

        btnOrganize = QPushButton('Organize', self)
        btnOrganize.clicked.connect(self.organize)
        grid_lt.addWidget(btnOrganize,6,0)

        btnExit = QPushButton('Ok', self)
        btnExit.clicked.connect(self.okayPressed)
        grid_lt.addWidget(btnExit,7,0)

        grid_rt.addWidget(self.scroll,0,1)

        layout.addLayout(grid_lt,0,0)
        layout.addLayout(grid_rt,0,1)
        self.setLayout(layout)

        if self.parent.argv.show and self.parent.argv.montage_file is None:
            self.show()
        elif not self.parent.argv.montage_file is None:
            self.loadTxtFile(self.parent.argv.montage_file)
            self.okayPressed()

    def populateChnList(self):
        """
        Fills the list with all of the channels in the edf file.
        PREDICTION channels are ignored and saved into self.pi
        """
        chns = self.data.chns2labels
        lbls = self.data.labels2chns
        self.data.pred_chn_data = []
        if len(self.unprocessed_data) > 0: # reset predicted
            self.parent.predicted = 0
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
        chns = self.data.getChns(self.data.labelsAR)
        if c.isChecked():
            self.cbox_txtfile.setChecked(0)
            self.cbox_bip.setChecked(0)
            # select all AR channels, deselect all others
            self._selectChns(chns, 0)
        else:
            self._selectChns(chns, 1)

    def bipChecked(self):
        c = self.sender()
        chns = self.data.getChns(self.data.labelsBIP)
        if self.ar:
            chns = self.data.getChns(self.data.labelsAR)
            if c.isChecked():
                self.cbox_ar.setChecked(0)
                self.cbox_txtfile.setChecked(0)
                self._selectChns(chns, 0)
            else:
                self._selectChns(chns, 1)
        elif c.isChecked():
            self.cbox_txtfile.setChecked(0)
            # select all bipolar channels, deselect all others
            self._selectChns(chns, 0)
        else:
            self._selectChns(chns, 1)

    def txtFileChecked(self):
        c = self.sender()
        chns = self.data.getChns(self.data.labelsFromTxtFile)
        if c.isChecked():
            if self.ar:
                self.cbox_ar.setChecked(0)
                self.cbox_bip.setChecked(0)
            if self.bip:
                self.cbox_bip.setChecked(0)
            self._selectChns(chns, 0)
            self.data.use_loaded_txt_file = 1
        else:
            self._selectChns(chns, 1)
            self.data.use_loaded_txt_file = 0

    def _selectChns(self, chns, deselectOnly):
        """
        Selects given channels.

        input:
            deselectOnly - whether to only deselect given channels
        """
        if deselectOnly:
            for k in range(len(chns)):
                if chns[k]:
                    self.chn_items[k].setSelected(0)
        else:
            for k in range(len(chns)):
                if chns[k]:
                    self.chn_items[k].setSelected(1)
                else:
                    self.chn_items[k].setSelected(0)

    def loadTxtFile(self, name = ""):
        if self.parent.argv.montage_file is None:
            name = QFileDialog.getOpenFileName(self, 'Open file','.','Text files (*.txt)')
            name = name[0]
        if name == None or len(name) == 0:
            return
        else:
            if self._check_chns(name):
                self.data.txtFile_fn = name.split('/')[-1]
                if len(name.split('/')[-1]) > 15:
                    self.data.txtFile_fn = name.split('/')[-1][0:15] + "..."
                self.btn_loadtxtfile.setVisible(0)
                self.btn_cleartxtfile.setVisible(1)
                self.cbox_txtfile.setVisible(1)
                self.cbox_txtfile.setChecked(1)
                self.cbox_txtfile.setText(self.data.txtFile_fn)
            else:
                # throw error
                self.parent.throwAlert("The channels in this file do not match"
                    + " those of the .edf file you have loaded. Please check your file.")

    def _check_chns(self, txt_fn):
        """
        Function to check that this file has the appropriate channel names.
        Sets self.data.labelsFromTxtFile if valid.

        inputs:
            txt_fn: the file name to be loaded
        returns:
            1 for sucess, 0 for at least one of the channels was not found in
            the .edf file
        """
        text_file = open(txt_fn, "r")
        lines = text_file.readlines()
        l = 0
        while l < len(lines):
            loc = lines[l].find("\n")
            if loc != -1:
                lines[l] = lines[l][0:loc]
            if len(lines[l]) == 0:
                lines.pop(l)
            else:
                l += 1
        text_file.close()
        ret = 1
        for l in lines:
            if not l in self.data.convertedChnNames:
                ret = 0
        if ret:
            for i in range(len(lines)):
                self.data.labelsFromTxtFile.append(lines[len(lines) - 1 - i])
        return ret

    def clearTxtFile(self):
        self.cbox_txtfile.setVisible(0)
        self.btn_loadtxtfile.setVisible(1)
        self.btn_cleartxtfile.setVisible(0)
        self.cbox_txtfile.setChecked(0)
        self.data.labelsFromTxtFile = []
        self.data.txtFile_fn = ""

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
            - parent.predicted
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
                if self.parent.si.plotSpec:
                    self.parent.si.plotSpec = 0
                    self.parent.removeSpecPlot()
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
