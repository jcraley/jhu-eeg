from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import (QVBoxLayout, QMessageBox, QWidget, QListWidget,
                                QPushButton, QCheckBox, QLabel, QGridLayout,
                                QScrollArea, QListWidgetItem, QAbstractItemView,
                                QFileDialog, QStyle)

import numpy as np
from preds_info import PredsInfo
from channel_info import ChannelInfo
from organize_channels import OrganizeChannels
from copy import deepcopy
import pyedflib

class ChannelOptions(QWidget):
    def __init__(self,data,parent):
        super().__init__()
        self.left = 10
        self.top = 10
        self.title = 'Select signals'
        self.width = parent.width / 5
        self.height = parent.height / 2.5
        # self.unprocessed_data = data_for_preds
        # if loading new data make copies in case user cancels loading channels
        self.new_load = 0
        # if len(self.unprocessed_data) != 0:
        if data.edf_fn != parent.ci.edf_fn:
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
        self.ar1020 = self.data.canDoBIP_AR(1,0)
        self.bip1020 = self.data.canDoBIP_AR(0,0)
        self.ar1010 = self.data.canDoBIP_AR(1,1)
        self.bip1010 = self.data.canDoBIP_AR(0,1)
        self.data.total_nchns = len(self.data.chns2labels)

        self.setWindowTitle(self.title)
        self.setGeometry(self.parent.width / 3, self.parent.height / 3, 
                                self.width, self.height)

        lblInfo = QLabel("Select channels to plot: ")
        grid_lt.addWidget(lblInfo,0,0)

        self.scroll_chn_cbox = QScrollArea()
        #self.scroll_chn_cbox.setMinimumWidth(120)
        #self.scroll_chn_cbox.setMinimumHeight(200)
        self.scroll_chn_cbox.setWidgetResizable(True)
        self.scroll_chn_cbox.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_chn_cbox.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        if self.ar1020:
            self.cbox_ar = QCheckBox("Average reference (10-20)",self)
            self.cbox_ar.toggled.connect(self.arChecked)
            grid_lt.addWidget(self.cbox_ar,1,0)

            self.cbox_bip = QCheckBox("Bipolar (10-20)",self)
            self.cbox_bip.toggled.connect(self.bipChecked)
            grid_lt.addWidget(self.cbox_bip,2,0)
        elif self.bip1020:
            self.cbox_bip = QCheckBox("Bipolar (10-20)",self)
            self.cbox_bip.toggled.connect(self.bipChecked)
            grid_lt.addWidget(self.cbox_bip,1,0)

        if self.ar1010:
            self.cbox_ar1010 = QCheckBox("Average reference (10-10)",self)
            self.cbox_ar1010.toggled.connect(self.arChecked1010)
            grid_lt.addWidget(self.cbox_ar1010,3,0)

            self.cbox_bip1010 = QCheckBox("Bipolar (10-10)",self)
            self.cbox_bip1010.toggled.connect(self.bipChecked1010)
            grid_lt.addWidget(self.cbox_bip1010,4,0)
        elif self.bip1010:
            self.cbox_bip1010 = QCheckBox("Bipolar (10-10)",self)
            self.cbox_bip1010.toggled.connect(self.bipChecked1010)
            grid_lt.addWidget(self.cbox_bip1010,3,0)

        self.chn_cbox_list = QWidget()
        # self.chn_cbox_list.setObjectName("chn_cbox_list")
        self.scroll_chn_cbox.setWidget(self.chn_cbox_list)
        self.chn_cbox_layout = QVBoxLayout()
        self.chn_cbox_list.setLayout(self.chn_cbox_layout)
        self.cbox_list_items = []
        for k in self.data.labelsFromTxtFile.keys():
            self.addTxtFile(k)
        self.uncheck_txt_files()

        grid_lt.addWidget(self.scroll_chn_cbox,5,0)
        #self.cbox_txtfile = QCheckBox("",self)
        #self.cbox_txtfile.toggled.connect(self.txtFileChecked)
        #grid_lt.addWidget(self.cbox_txtfile,6,0)

        self.btn_loadtxtfile = QPushButton("Load text file",self)
        self.btn_loadtxtfile.clicked.connect(self.loadTxtFile)
        grid_lt.addWidget(self.btn_loadtxtfile,6,0)

        #self.btn_cleartxtfile = QPushButton("Clear text file",self)
        #self.btn_cleartxtfile.clicked.connect(self.clearTxtFile)
        #self.btn_cleartxtfile.setVisible(0)
        #grid_lt.addWidget(self.btn_cleartxtfile,6,0)

        if len(self.data.txtFile_fn) > 0:
            if self.data.use_loaded_txt_file:
                self.cbox_txtfile.setChecked(1)
            self.btn_loadtxtfile.setVisible(0)
            self.btn_cleartxtfile.setVisible(1)
            self.cbox_txtfile.setVisible(1)
            self.cbox_txtfile.setText(self.data.txtFile_fn)

        lbl = QLabel("")
        grid_lt.addWidget(lbl, 7,0)

        btnOrganize = QPushButton('Organize', self)
        btnOrganize.clicked.connect(self.organize)
        grid_lt.addWidget(btnOrganize,8,0)

        btnExit = QPushButton('Ok', self)
        btnExit.clicked.connect(self.okayPressed)
        grid_lt.addWidget(btnExit,9,0)

        grid_rt.addWidget(self.scroll,0,1)

        layout.addLayout(grid_lt,0,0)
        layout.addLayout(grid_rt,0,1)
        self.setLayout(layout)

        if (not self.parent.argv.montage_file is None) and self.parent.init == 0:
            self.loadTxtFile(self.parent.argv.montage_file)
            self.okayPressed()
        else:
            self.show()

    def populateChnList(self):
        """
        Fills the list with all of the channels in the edf file.
        PREDICTION channels are ignored and saved into self.pi
        """
        chns = self.data.chns2labels
        lbls = self.data.labels2chns
        self.data.pred_chn_data = []
        f = pyedflib.EdfReader(self.data.edf_fn)
        # if len(self.unprocessed_data) > 0: # reset predicted
        #    self.parent.predicted = 0
        if len(chns) == 0:
            self.parent.throwAlert("There are no named channels in the file.")
            self.closeWindow()
        else:
            self.chn_items = []
            for i in range(len(chns)):
                if chns[i].find("PREDICTIONS") == -1:
                    self.chn_items.append(QListWidgetItem(chns[i], self.chn_qlist))
                    self.chn_qlist.addItem(self.chn_items[i])
                # elif len(self.unprocessed_data) > 0:
                # load in the prediction channels if they exist
                # if they do, then the file was saved which
                # means that there are a reasonable amount of channels
                elif self.new_load:
                    # self.data.pred_chn_data.append(self.unprocessed_data[i])
                    self.data.pred_chn_data.append(f.readSignal(i))
                    lbls.pop(chns[i])
                    chns.pop(i)

            # if len(self.unprocessed_data) > 0 and len(self.data.pred_chn_data) != 0:
            if self.new_load and len(self.data.pred_chn_data) != 0:
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

            # select the previously selected channels if they exist
            if len(self.data.list_of_chns) != 0 and not self.new_load:
                for k in range(len(self.data.list_of_chns)):
                    self.chn_items[self.data.list_of_chns[k]].setSelected(True)
            self.scroll.show()

    def arChecked(self):
        c = self.sender()
        chns = self.data.getChns(self.data.labelsAR1020)
        if c.isChecked():
            # self.cbox_txtfile.setChecked(0)
            self.uncheck_txt_files()
            self.cbox_bip.setChecked(0)
            if self.ar1010:
                self.cbox_ar1010.setChecked(0)
                self.cbox_bip1010.setChecked(0)
            elif self.bip1010:
                self.cbox_bip1010.setChecked(0)
            # select all AR channels, deselect all others
            self._selectChns(chns, 0)
        else:
            self._selectChns(chns, 1)

    def bipChecked(self):
        c = self.sender()
        chns = self.data.getChns(self.data.labelsBIP1020)
        if c.isChecked():
            if self.ar1010:
                self.cbox_ar1010.setChecked(0)
                self.cbox_bip1010.setChecked(0)
            elif self.bip1010:
                self.cbox_bip1010.setChecked(0)
        if self.ar1020:
            chns = self.data.getChns(self.data.labelsAR1020)
            if c.isChecked():
                self.cbox_ar.setChecked(0)
                # self.cbox_txtfile.setChecked(0)
                self.uncheck_txt_files()
                self._selectChns(chns, 0)
            else:
                self._selectChns(chns, 1)
        elif c.isChecked():
            # self.cbox_txtfile.setChecked(0)
            self.uncheck_txt_files()
            # select all bipolar channels, deselect all others
            self._selectChns(chns, 0)
        else:
            self._selectChns(chns, 1)

    def arChecked1010(self):
        c = self.sender()
        chns = self.data.getChns(self.data.labelsAR1010)
        if c.isChecked():
            # self.cbox_txtfile.setChecked(0)
            self.uncheck_txt_files()
            self.cbox_bip.setChecked(0)
            self.cbox_ar.setChecked(0)
            self.cbox_bip1010.setChecked(0)
            # select all AR channels, deselect all others
            self._selectChns(chns, 0)
        else:
            self._selectChns(chns, 1)

    def bipChecked1010(self):
        c = self.sender()
        chns = self.data.getChns(self.data.labelsBIP1010)
        if self.ar1020:
            chns = self.data.getChns(self.data.labelsAR1010)
            if c.isChecked():
                self.cbox_ar.setChecked(0)
                # self.cbox_txtfile.setChecked(0)
                self.uncheck_txt_files()
                if self.ar1010:
                    self.cbox_ar1010.setChecked(0)
                self.cbox_bip.setChecked(0)
                self._selectChns(chns, 0)
            else:
                self._selectChns(chns, 1)
        elif c.isChecked():
            # self.cbox_txtfile.setChecked(0)
            self.uncheck_txt_files()
            self.cbox_bip.setChecked(0)
            # select all bipolar channels, deselect all others
            self._selectChns(chns, 0)
        else:
            self._selectChns(chns, 1)

    def uncheck_txt_files(self):
        """ Deselect all text files. 
        """
        for child in self.chn_cbox_list.children():
            for ch in child.children():
                if isinstance(ch, QCheckBox):
                    ch.setChecked(0)

    def txtFileChecked(self):
        c = self.sender()
        name = c.text()
        chns = self.data.getChns(self.data.labelsFromTxtFile[name])
        if c.isChecked():
            if self.ar1020:
                self.cbox_ar.setChecked(0)
                self.cbox_bip.setChecked(0)
                if self.ar1010:
                    self.cbox_ar1010.setChecked(0)
                    self.cbox_bip1010.setChecked(0)
            if self.bip1020:
                self.cbox_bip.setChecked(0)
                if self.bip1010:
                    self.cbox_bip1010.setChecked(0)
            for child in self.chn_cbox_list.children():
                for ch in child.children():
                    if isinstance(ch, QCheckBox):
                        if ch.text() != name:
                            ch.setChecked(0)
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

    def test_delete_txt(self):
        # get parent widget
        c = self.sender()
        parent_wid = c.parent()
        qscroll_wid = parent_wid.parent()
        for child in parent_wid.children():
            if isinstance(child, QCheckBox):
                print("check box")
                print(child.text())
        parent_wid.hide()

    def addTxtFile(self, name):
        """ Called to load in the new text file.
        """
        # add text file to the list
        main_wid = QWidget()
        wid = QGridLayout()
        wid_name = QCheckBox(name)
        wid_name.toggled.connect(self.txtFileChecked)
        wid_name.setChecked(1)
        wid.addWidget(wid_name,0,0)
        wid_btn = QPushButton()
        # wid_btn.clicked.connect(self.test_delete_txt)
        wid_btn.setIcon(self.style().standardIcon(getattr(QStyle, 'SP_DialogDiscardButton')))
        wid.addWidget(wid_btn, 0,1)
        main_wid.setLayout(wid)
        self.chn_cbox_layout.addWidget(main_wid)

    def loadTxtFile(self, name = ""):
        if self.parent.argv.montage_file is None or self.parent.init:
            name = QFileDialog.getOpenFileName(self, 'Open file','.','Text files (*.txt)')
            name = name[0]
        if name == None or len(name) == 0:
            return
        else:
            short_name = name.split('/')[-1]
            if len(name.split('/')[-1]) > 15:
                short_name = name.split('/')[-1][0:15] + "..."
            if short_name in self.data.labelsFromTxtFile.keys():
                self.parent.throwAlert("Each loaded text file must have a unique " 
                 + "name (first 14 characters). Please rename your file.")
                return
            if self._check_chns(name, short_name):
                #self.btn_loadtxtfile.setVisible(0)
                #self.btn_cleartxtfile.setVisible(1)
                #self.cbox_txtfile.setVisible(1)
                #self.cbox_txtfile.setChecked(1)
                #self.cbox_txtfile.setText(self.data.txtFile_fn)
                self.addTxtFile(short_name)
            else:
                # throw error
                self.parent.throwAlert("The channels in this file do not match"
                    + " those of the .edf file you have loaded. Please check your file.")

    def _check_chns(self, txt_fn, txt_fn_short):
        """
        Function to check that this file has the appropriate channel names.
        Sets self.data.labelsFromTxtFile if valid.

        inputs:
            txt_fn: the file name to be loaded
            txt_fn_short: the name to be used in the dict
        returns:
            1 for sucess, 0 for at least one of the channels was not found in
            the .edf file
        """
        try:
            text_file = open(txt_fn, "r")
        except:
            self.parent.throwAlert("The .txt file is invalid.")
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
            if not l in self.data.convertedChnNames and not l in self.data.labels2chns:
                ret = 0
        if ret:
            self.data.labelsFromTxtFile[txt_fn_short] = []
            for i in range(len(lines)):
                if not lines[len(lines) - 1 - i] in self.data.convertedChnNames:
                    lines[len(lines) - 1 - i] = self.data.convertTxtChnNames(lines[len(lines) - 1 - i])
                self.data.labelsFromTxtFile[txt_fn_short].append(lines[len(lines) - 1 - i])
        return ret

    def clearTxtFile(self):
        self.cbox_txtfile.setVisible(0)
        self.btn_loadtxtfile.setVisible(1)
        self.btn_cleartxtfile.setVisible(0)
        self.cbox_txtfile.setChecked(0)
        self.data.labelsFromTxtFile = []
        self.data.txtFile_fn = ""

    def check_multi_chn_preds(self):
        """ Check if plotting predictions by channel. If so, check
            whether the number of channels match. 

            Sets parent.predicted to 1 if correct, 0 if incorrect.
        """
        if self.parent.pi.pred_by_chn and self.parent.predicted:
            if self.parent.ci.nchns_to_plot != self.parent.pi.preds_to_plot.shape[1]:
                self.parent.predicted = 0

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
            - parent.count (set to 0 if new load)
        """
        self.parent.edf_info = self.parent.edf_info_temp
        # self.parent.data = self.parent.data_temp
        self.parent.max_time = self.parent.max_time_temp
        self.parent.pi.write_data(self.pi)
        self.parent.ci.write_data(self.data)
        self.data = self.parent.ci
        self.parent.sei.fn = self.parent.fn_full_temp
        # if len(self.unprocessed_data) > 0: # new load
        if self.new_load:
            self.parent.count = 0
            self.parent.lblFn.setText("Plotting: " + self.parent.fn_temp)

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
        num_chns = 0
        txt_file_name = ""
        for k in range(len(self.chn_items)):
            if self.chn_items[k] in selectedListItems:
                idxs.append(self.data.labels2chns[self.data.chns2labels[k]])
        if len(idxs) > self.parent.max_channels:
            self.parent.throwAlert("You may select at most " + 
                                    str(self.parent.max_channels) + " to plot. " + 
                                    "You have selected " + str(len(idxs)) + ".")
            return -1
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
            # data = self.parent.data
            plot_bip_from_ar = 0
            if (self.ar1020 and self.cbox_bip.isChecked() or
                    self.ar1010 and self.cbox_bip1010.isChecked()):
                plot_bip_from_ar = 1
            mont_type = 5
            if self.ar1020 and self.cbox_ar.isChecked():
                mont_type = 0
            elif (self.ar1020 or self.bip1020) and self.cbox_bip.isChecked():
                mont_type = 1
            elif self.ar1010 and self.cbox_ar1010.isChecked():
                mont_type = 2
            elif self.bip1010 and self.cbox_bip1010.isChecked():
                mont_type = 3
            else:
                # check if cbox_txtfile.isChecked()
                for child in self.chn_cbox_list.children():
                    for ch in child.children():
                        if isinstance(ch, QCheckBox) and ch.isChecked():
                            txt_file_name = ch.text()
                            mont_type = 4
            self.data.prepareToPlot(idxs, self.parent, mont_type, plot_bip_from_ar, txt_file_name)
            # check if multi-chn pred and number of chns match
            self.check_multi_chn_preds()
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
