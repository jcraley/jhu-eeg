""" Module for spectrogram options window """
from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import (QVBoxLayout, QMessageBox, QWidget, QListWidget,
                                QPushButton, QCheckBox, QLabel, QGridLayout,
                                QScrollArea, QListWidgetItem, QAbstractItemView,
                                QFileDialog, QSpinBox, QComboBox, QDoubleSpinBox)
from PyQt5.QtGui import QFont
from matplotlib.backends.qt_compat import QtWidgets

import numpy as np

class SpecOptions(QWidget):
    """ Class for spectrogram options window """
    def __init__(self,data,parent):
        """ Constructor """
        super().__init__()
        self.left = 10
        self.top = 10
        self.title = 'Select signal for spectrogram'
        self.width = parent.width / 6
        self.height = parent.height / 2.5
        self.data = data
        self.parent = parent
        self.setupUI()

    def setup_ui(self):
        """ Setup the UI """
        self.setWindowTitle(self.title)
        center_point = QtWidgets.QDesktopWidget().availableGeometry().center()

        self.setGeometry(center_point.x() - self.width / 2, center_point.y() - self.height / 2, self.width, self.height)

        grid = QGridLayout()

        self.chn_combobox = QComboBox()
        self.chn_combobox.addItems(["<select channel>"])

        lbl_info = QLabel("To hide the plot, click \n\"Clear\" and then \"Ok\".")
        grid.addWidget(lbl_info,3,0)
        my_font=QFont()
        my_font.setBold(True)
        lbl_info.setFont(my_font)

        lbl_chn = QLabel("Select a channel for \nspectrogram plotting: ")
        grid.addWidget(lbl_chn,1,0)
        # grid.addWidget(self.scroll,1,0)
        grid.addWidget(self.chn_combobox,1,1,1,3)

        lblfsaxis = QLabel("x-axis (Hz): ")
        grid.addWidget(lblfsaxis,2,0)
        self.btn_get_min_fs = QDoubleSpinBox(self)
        self.btn_get_min_fs.setRange(0, self.parent.edf_info.fs / 2)
        self.btn_get_min_fs.setValue(self.data.minFs)
        self.btn_get_min_fs.setDecimals(3)
        grid.addWidget(self.btn_get_min_fs,2,1)
        lblfsto = QLabel(" to ")
        grid.addWidget(lblfsto,2,2)
        self.btn_get_max_fs = QDoubleSpinBox(self)
        self.btn_get_max_fs.setDecimals(3)
        self.btn_get_max_fs.setRange(0, self.parent.edf_info.fs / 2)
        self.btn_get_max_fs.setValue(self.data.maxFs)
        grid.addWidget(self.btn_get_max_fs,2,3)

        self.btn_clear = QPushButton('Clear', self)
        grid.addWidget(self.btn_clear,3,1,1,2)
        self.btn_exit = QPushButton('Ok', self)
        grid.addWidget(self.btn_exit,3,3)

        self.setLayout(grid)

        self.set_sig_slots()

    def set_sig_slots(self):
        """ Set signals and slots """
        self.populate_chn_list()
        self.btn_exit.clicked.connect(self.check)
        self.btn_clear.clicked.connect(self.clear_spec)

        self.show()

    def populate_chn_list(self):
        """ Fills the list with all of the channels to be loaded.
        """
        self.chn_items = []
        self.labels_flipped = []
        labels_to_plot = self.parent.ci.labels_to_plot
        if len(labels_to_plot) == 0:
            self.closeWindow()
        else:
            for i in range(len(labels_to_plot) - 1):
                #self.labels_flipped.append(self.data.labels_to_plot[len(self.data.labels_to_plot) - 1 - i])
                self.labels_flipped.append(labels_to_plot[i+1])
                self.chn_combobox.addItems([labels_to_plot[len(labels_to_plot) - 1 - i]])
                # self.chn_items.append(QListWidgetItem(labels_to_plot[len(labels_to_plot) - 1 - i], self.chn_qlist))
                # self.chn_qlist.addItem(self.chn_items[i])
                #if i == self.data.chnPlotted:
                    # self.chn_items[i].setSelected(1)
            if self.data.chnPlotted != -1:
                self.chn_combobox.setCurrentIndex(len(labels_to_plot) - self.data.chnPlotted - 1)
            else:
                self.chn_combobox.setCurrentIndex(0)
            # self.scroll.show()
            # self.labels_flipped.append(labels_to_plot[i+1])
            # self.chn_combobox.addItems(self.labels_flipped)

    def clear_spec(self):
        self.chn_combobox.setCurrentIndex(0)
        self.data.chnPlotted = -1
        #selectedListItem = self.chn_qlist.selectedItems()
        #if len(selectedListItem) == 1:
        #    selectedListItem[0].setSelected(0)

    def check(self):
        """ Function to check the clicked channel and exit.
        """
        # selectedListItem = self.chn_qlist.selectedItems()
        # selectedIdx = self.chn_qlist.selectedIndexes()[0].row()
        row = self.chn_combobox.currentIndex()
        if row != 0:
            # row = self.chn_combobox.currentIndex()
            # row = self.chn_qlist.selectedIndexes()[0].row()
            nchns = self.parent.ci.nchns_to_plot
            self.data.chnPlotted = nchns - row
            self.data.chnName = self.labels_flipped[len(self.labels_flipped) - row]
            self.data.data = self.parent.ci.data_to_plot[self.data.chnPlotted,:]
            if not self.data.plotSpec:
                self.data.plotSpec = 1
                self.parent.makeSpecPlot()
            else:
                self.parent.updateSpecChn()
        elif self.data.plotSpec:
            self.data.plotSpec = 0
            self.parent.removeSpecPlot()
            self.data.chnPlotted = -1
        if self.btn_get_max_fs.value() > self.btn_get_min_fs.value():
            self.data.maxFs = self.btn_get_max_fs.value()
            self.data.minFs = self.btn_get_min_fs.value()
            self.closeWindow()
        elif self.data.plotSpec:
            self.parent.throwAlert("Maximum frequency must be greater than minimum frequency.")
        else:
            self.closeWindow()

    def close_window(self):
        self.parent.spec_win_open = 0
        self.parent.callmovePlot(0,0)
        self.close()

    def closeEvent(self, event):
        """
        Called when the window is closed.
        """
        self.parent.spec_win_open = 0
        event.accept()
