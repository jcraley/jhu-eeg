from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import (QFileDialog, QVBoxLayout, QMessageBox, QWidget,
                                QPushButton, QCheckBox, QLabel, QInputDialog,
                                QSlider, QGridLayout, QSpinBox)


class FilterOptions(QWidget):
    def __init__(self,data,parent):
        super().__init__()
        self.left = 10
        self.top = 10
        self.title = 'Filter Options'
        self.width = 300
        self.height = 300
        self.data = data
        self.parent = parent
        self.setupUI()

    def setupUI(self):

        layout = QGridLayout()
        layout.setSpacing(4)

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        btnExit = QPushButton('Ok', self)
        btnExit.clicked.connect(self.change)
        layout.addWidget(btnExit,3,3)

        self.cbox_lp= QCheckBox("Lowpass",self)
        self.cbox_lp.toggled.connect(self.lp_filterChecked)
        self.cbox_lp.setToolTip("Click to filter")
        if self.data.do_lp:
            self.cbox_lp.setChecked(True)
        layout.addWidget(self.cbox_lp,0,0)

        self.btnGetLP = QSpinBox(self)
        self.btnGetLP.setValue(self.data.lp)
        self.btnGetLP.setRange(0, self.data.fs / 2)
        layout.addWidget(self.btnGetLP,0,1)

        lp_hz_lbl = QLabel("Hz",self)
        layout.addWidget(lp_hz_lbl,0,2)

        self.cbox_hp= QCheckBox("Highpass",self)
        self.cbox_hp.toggled.connect(self.hp_filterChecked)
        self.cbox_hp.setToolTip("Click to filter")
        if self.data.do_hp:
            self.cbox_hp.setChecked(True)
        layout.addWidget(self.cbox_hp,1,0)

        self.btnGetHP = QSpinBox(self)
        self.btnGetHP.setValue(self.data.hp)
        self.btnGetHP.setRange(0, self.data.fs / 2)
        layout.addWidget(self.btnGetHP,1,1)

        hp_hz_lbl = QLabel("Hz",self)
        layout.addWidget(hp_hz_lbl,1,2)

        self.cbox_notch = QCheckBox("Notch",self)
        self.cbox_notch.toggled.connect(self.notch_filterChecked)
        self.cbox_notch.setToolTip("Click to filter")
        if self.data.do_notch:
            self.cbox_notch.setChecked(True)
        layout.addWidget(self.cbox_notch,2,0)

        self.btnGetNotch = QSpinBox(self)
        self.btnGetNotch.setValue(self.data.notch)
        self.btnGetNotch.setRange(0, self.data.fs / 2)
        layout.addWidget(self.btnGetNotch,2,1)

        notch_hz_lbl = QLabel("Hz",self)
        layout.addWidget(notch_hz_lbl,2,2)

        self.setLayout(layout)

        self.show()

    def lp_filterChecked(self):
        c = self.sender()
        if c.isChecked():
            self.data.do_lp = 1
        else:
            self.data.do_lp = 0

    def hp_filterChecked(self):
        c = self.sender()
        if c.isChecked():
            self.data.do_hp = 1
        else:
            self.data.do_hp = 0

    def notch_filterChecked(self):
        c = self.sender()
        if c.isChecked():
            self.data.do_notch = 1
        else:
            self.data.do_notch = 0

    def change(self):
        """
        Checks to make sure values are legal and updates the PredsInfo object
        """
        hp = self.btnGetHP.value()
        lp = self.btnGetLP.value()
        if (lp > 0 and lp < self.data.fs / 2 and hp > 0 and hp < self.data.fs / 2):
            if lp - hp > 0:
                if self.data.do_lp:
                    self.data.lp = self.btnGetLP.value()
                if self.data.do_hp:
                    self.data.hp = self.btnGetHP.value()
        if self.btnGetNotch.value() > 0 and self.btnGetNotch.value() < self.data.fs / 2:
            if self.data.do_notch:
                self.data.notch = self.btnGetNotch.value()
        else:
            self.data.do_notch = 0
        self.parent.callmovePlot(0,0,0)
        self.closeWindow()

    def closeWindow(self):
        self.parent.filter_win_open = 0
        self.close()
