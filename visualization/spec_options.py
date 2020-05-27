from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import (QVBoxLayout, QMessageBox, QWidget, QListWidget,
                                QPushButton, QCheckBox, QLabel, QGridLayout,
                                QScrollArea, QListWidgetItem, QAbstractItemView,
                                QFileDialog)

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

        grid = QGridLayout()

        self.scroll = QScrollArea()
        self.scroll.setMinimumWidth(120)
        self.scroll.setMinimumHeight(200) # would be better if resizable
        self.scroll.setWidgetResizable(True)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.chn_qlist = QListWidget()
        self.chn_qlist.setSelectionMode(QAbstractItemView.SingleSelection)
        self.scroll.setWidget(self.chn_qlist)

        self.populateChnList()

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        lblInfo = QLabel("Select a channel for \n spectrogram plotting: ")
        grid.addWidget(lblInfo,0,0,1,2)

        btnExit = QPushButton('Ok', self)
        btnExit.clicked.connect(self.check)
        grid.addWidget(btnExit,2,1)

        btnClear = QPushButton('Clear', self)
        btnClear.clicked.connect(self.clearSpec)
        grid.addWidget(btnClear,2,0)

        grid.addWidget(self.scroll,1,0)

        self.setLayout(grid)

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
                self.chn_items.append(QListWidgetItem(labels_to_plot[len(labels_to_plot) - 1 - i], self.chn_qlist))
                self.chn_qlist.addItem(self.chn_items[i])
                if i == self.data.chnPlotted:
                    self.chn_items[i].setSelected(1)
            self.scroll.show()

    def clearSpec(self):
        selectedListItem = self.chn_qlist.selectedItems()
        if len(selectedListItem) == 1:
            selectedListItem[0].setSelected(0)

    def check(self):
        """
        Function to check the clicked channel and exit.
        """
        # TODO: get selected chn from parent.data
        selectedListItem = self.chn_qlist.selectedItems()
        if len(selectedListItem) == 1:
            row = self.chn_qlist.currentRow()
            self.data.chnPlotted = row
            nchns = self.parent.ci.nchns_to_plot
            self.data.data = self.parent.ci.data_to_plot[nchns - row,:]
            if not self.data.plotSpec:
                self.data.plotSpec = 1
                self.parent.makeSpecPlot()
        elif self.data.plotSpec:
            self.data.plotSpec = 0
            self.parent.removeSpecPlot()
            self.data.chnPlotted = -1
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
