from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import (QVBoxLayout, QMessageBox, QWidget, QListWidget,
                                QPushButton, QCheckBox, QLabel, QGridLayout,
                                QScrollArea, QListWidgetItem, QAbstractItemView,
                                QFileDialog, QSpinBox, QComboBox, QDoubleSpinBox)
from PyQt5.QtGui import QFont
from matplotlib.backends.qt_compat import QtWidgets

import numpy as np

class SpecOptions(QWidget):
    def __init__(self,data,parent):
        super().__init__()
        self.left = 10
        self.top = 10
        self.title = 'Select signal for spectrogram'
        self.width = parent.width / 6
        self.height = parent.height / 2.5
        self.data = data
        self.parent = parent
        self.setupUI()

    def setupUI(self):
        self.setWindowTitle(self.title)
        centerPoint = QtWidgets.QDesktopWidget().availableGeometry().center()

        self.setGeometry(centerPoint.x() - self.width / 2, centerPoint.y() - self.height / 2, self.width, self.height)

        grid = QGridLayout()

        self.chnComboBox = QComboBox()
        self.chnComboBox.addItems(["<select channel>"])

        lblInfo = QLabel("To hide the plot, click \n\"Clear\" and then \"Ok\".")
        grid.addWidget(lblInfo,3,0)
        myFont=QFont()
        myFont.setBold(True)
        lblInfo.setFont(myFont)

        lblChn = QLabel("Select a channel for \nspectrogram plotting: ")
        grid.addWidget(lblChn,1,0)
        # grid.addWidget(self.scroll,1,0)
        grid.addWidget(self.chnComboBox,1,1,1,3)

        lblfsaxis = QLabel("y-axis (V**2/Hz): ")
        grid.addWidget(lblfsaxis,2,0)
        self.btnGetMinFs = QDoubleSpinBox(self)
        self.btnGetMinFs.setRange(0, self.parent.edf_info.fs / 2)
        self.btnGetMinFs.setValue(self.data.minFs)
        self.btnGetMinFs.setDecimals(3)
        grid.addWidget(self.btnGetMinFs,2,1)
        lblfsto = QLabel(" to ")
        grid.addWidget(lblfsto,2,2)
        self.btnGetMaxFs = QDoubleSpinBox(self)
        self.btnGetMaxFs.setDecimals(3)
        self.btnGetMaxFs.setRange(0, self.parent.edf_info.fs / 2)
        self.btnGetMaxFs.setValue(self.data.maxFs)
        grid.addWidget(self.btnGetMaxFs,2,3)

        self.btnClear = QPushButton('Clear', self)
        grid.addWidget(self.btnClear,3,1,1,2)
        self.btnExit = QPushButton('Ok', self)
        grid.addWidget(self.btnExit,3,3)

        self.setLayout(grid)

        self.setSigSlots()

    def setSigSlots(self):
        # set signals and slots
        self.populateChnList()
        self.btnExit.clicked.connect(self.check)
        self.btnClear.clicked.connect(self.clearSpec)

        self.show()

    def populateChnList(self):
        """
        Fills the list with all of the channels to be loaded.
        """
        #chns = self.data.chns2labels
        #lbls = self.data.labels2chns
        self.chn_items = []
        self.labels_flipped = []
        labels_to_plot = self.parent.ci.labels_to_plot
        if len(labels_to_plot) == 0:
            self.closeWindow()
        else:
            for i in range(len(labels_to_plot) - 1):
                #self.labels_flipped.append(self.data.labels_to_plot[len(self.data.labels_to_plot) - 1 - i])
                self.labels_flipped.append(labels_to_plot[i+1])
                self.chnComboBox.addItems([labels_to_plot[len(labels_to_plot) - 1 - i]])
                # self.chn_items.append(QListWidgetItem(labels_to_plot[len(labels_to_plot) - 1 - i], self.chn_qlist))
                # self.chn_qlist.addItem(self.chn_items[i])
                #if i == self.data.chnPlotted:
                    # self.chn_items[i].setSelected(1)
            if self.data.chnPlotted != -1:
                self.chnComboBox.setCurrentIndex(len(labels_to_plot) - self.data.chnPlotted - 1)
            else:
                self.chnComboBox.setCurrentIndex(0)
            # self.scroll.show()
            # self.labels_flipped.append(labels_to_plot[i+1])
            # self.chnComboBox.addItems(self.labels_flipped)

    def clearSpec(self):
        self.chnComboBox.setCurrentIndex(0)
        self.data.chnPlotted = -1
        #selectedListItem = self.chn_qlist.selectedItems()
        #if len(selectedListItem) == 1:
        #    selectedListItem[0].setSelected(0)

    def check(self):
        """
        Function to check the clicked channel and exit.
        """
        # selectedListItem = self.chn_qlist.selectedItems()
        # selectedIdx = self.chn_qlist.selectedIndexes()[0].row()
        row = self.chnComboBox.currentIndex()
        if row != 0:
            # row = self.chnComboBox.currentIndex()
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
        if self.btnGetMaxFs.value() > self.btnGetMinFs.value():
            self.data.maxFs = self.btnGetMaxFs.value()
            self.data.minFs = self.btnGetMinFs.value()
            self.closeWindow()
        elif self.data.plotSpec:
            self.parent.throwAlert("Maximum frequency must be greater than minimum frequency.")
        else:
            self.closeWindow()

    def closeWindow(self):
        self.parent.spec_win_open = 0
        self.parent.callmovePlot(0,0)
        self.close()

    def closeEvent(self, event):
        """
        Called when the window is closed.
        """
        self.parent.spec_win_open = 0
        event.accept()
