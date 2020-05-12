from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import (QVBoxLayout, QMessageBox, QWidget, QListWidget,
                                QPushButton, QCheckBox, QLabel, QGridLayout,
                                QScrollArea, QListWidgetItem, QAbstractItemView)

import numpy as np
from channel_info import ChannelInfo

class OrganizeChannels(QWidget):
    def __init__(self,data,parent):
        super().__init__()
        self.left = 10
        self.top = 10
        self.title = 'Organize signals'
        self.width = 200
        self.height = 400
        self.data = data
        self.parent = parent
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
        #self.chn_qlist.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.chn_qlist.setSelectionMode(QAbstractItemView.SingleSelection)
        self.chn_qlist.setDragEnabled(True)
        self.chn_qlist.setDragDropMode(QAbstractItemView.InternalMove)
        #self.chn_qlist.viewport.setAcceptDrops(True)
        self.chn_qlist.setDropIndicatorShown(True)

        self.scroll.setWidget(self.chn_qlist)

        self.populateChnList()

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        lblInfo = QLabel("Drag and drop channels \n to change their order: ")
        grid_lt.addWidget(lblInfo,0,0)

        btnExit = QPushButton('Ok', self)
        btnExit.clicked.connect(self.updateChnOrder)
        grid_lt.addWidget(btnExit,2,0)

        grid_lt.addWidget(self.scroll,1,0)

        layout.addLayout(grid_lt,0,0)
        layout.addLayout(grid_rt,0,1)
        self.setLayout(layout)

        self.show()

    def populateChnList(self):
        """
        Fills the list with all of the channels to be loaded.
        """
        #chns = self.data.chns2labels
        #lbls = self.data.labels2chns
        self.chn_items = []
        self.labels_flipped = []
        if len(self.data.labels_to_plot) == 0:
            self.closeWindow()
        else:
            for i in range(len(self.data.labels_to_plot) - 1):
                #self.labels_flipped.append(self.data.labels_to_plot[len(self.data.labels_to_plot) - 1 - i])
                self.labels_flipped.append(self.data.labels_to_plot[i+1])
                self.chn_items.append(QListWidgetItem(self.data.labels_to_plot[len(self.data.labels_to_plot) - 1 - i], self.chn_qlist))
                self.chn_qlist.addItem(self.chn_items[i])
            self.scroll.show()

    def updateChnOrder(self):
        """
        Function to check the clicked channels and exit.
        """
        temp_labels = ["Notes"]
        temp_colors = []
        temp_data = np.zeros(self.data.data_to_plot.shape)
        for i in range(len(self.data.colors)):
            temp_labels.append(self.data.labels_to_plot[i+1])
            temp_colors.append(self.data.colors[i])
            temp_data[i,:] += self.data.data_to_plot[i,:]

        for k in range(len(self.chn_items)):
            row = self.chn_qlist.row(self.chn_items[k])
            temp_labels[len(self.chn_items) - row] = self.chn_items[k].text()
            temp_colors[len(self.chn_items) - row - 1] = self.data.colors[len(self.chn_items) - k - 1]
            temp_data[len(self.chn_items) - row - 1,:] = self.data.data_to_plot[len(self.chn_items) - k - 1,:]
        self.data.labels_to_plot = temp_labels
        self.data.colors = temp_colors
        self.data.data_to_plot = temp_data
        self.parent.callInitialMovePlot()
        self.closeWindow()

    def closeWindow(self):
        self.parent.organize_win_open = 0
        self.close()

    def closeEvent(self, event):
        """
        Called when the window is closed.
        """
        self.parent.organize_win_open = 0
        event.accept()
