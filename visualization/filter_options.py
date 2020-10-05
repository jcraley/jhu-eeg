from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import (QFileDialog, QVBoxLayout, QMessageBox, QWidget,
                                QPushButton, QCheckBox, QLabel, QInputDialog,
                                QSlider, QGridLayout, QSpinBox, QDoubleSpinBox)

from matplotlib.backends.qt_compat import QtWidgets


class FilterOptions(QWidget):
    def __init__(self,data,parent):
        super().__init__()
        self.left = 10
        self.top = 10
        self.title = 'Filter Options'
        self.width = parent.width / 5
        self.height = parent.height / 3
        self.data = data
        self.parent = parent
        self.setupUI()

    def setupUI(self):

        layout = QGridLayout()
        self.setWindowTitle(self.title)
        centerPoint = QtWidgets.QDesktopWidget().availableGeometry().center()
        self.setGeometry(centerPoint.x() - self.width / 2, centerPoint.y() - self.height / 2, self.width, self.height)

        self.btnExit = QPushButton('Ok', self)
        layout.addWidget(self.btnExit,4,3)

        self.cbox_lp = QCheckBox("Lowpass",self)
        self.cbox_lp.setToolTip("Click to filter")
        if self.data.do_lp:
            self.cbox_lp.setChecked(True)
        layout.addWidget(self.cbox_lp,0,0)

        self.btnGetLP = QDoubleSpinBox(self)
        self.btnGetLP.setValue(self.data.lp)
        self.btnGetLP.setRange(0, self.data.fs / 2)
        layout.addWidget(self.btnGetLP,0,1)

        lp_hz_lbl = QLabel("Hz",self)
        layout.addWidget(lp_hz_lbl,0,2)

        self.cbox_hp= QCheckBox("Highpass",self)
        self.cbox_hp.setToolTip("Click to filter")
        if self.data.do_hp:
            self.cbox_hp.setChecked(True)
        layout.addWidget(self.cbox_hp,1,0)

        self.btnGetHP = QDoubleSpinBox(self)
        self.btnGetHP.setValue(self.data.hp)
        self.btnGetHP.setRange(0, self.data.fs / 2)
        layout.addWidget(self.btnGetHP,1,1)

        hp_hz_lbl = QLabel("Hz",self)
        layout.addWidget(hp_hz_lbl,1,2)

        self.cbox_notch = QCheckBox("Notch",self)
        self.cbox_notch.setToolTip("Click to filter")
        if self.data.do_notch:
            self.cbox_notch.setChecked(True)
        layout.addWidget(self.cbox_notch,2,0)

        self.btnGetNotch = QDoubleSpinBox(self)
        self.btnGetNotch.setValue(self.data.notch)
        self.btnGetNotch.setRange(0, self.data.fs / 2)
        layout.addWidget(self.btnGetNotch,2,1)

        notch_hz_lbl = QLabel("Hz",self)
        layout.addWidget(notch_hz_lbl,2,2)

        self.cbox_bp = QCheckBox("Bandpass",self)
        self.cbox_bp.setToolTip("Click to filter")
        if self.data.do_bp:
            self.cbox_bp.setChecked(True)
        layout.addWidget(self.cbox_bp,3,0)

        self.btnGetBP1 = QDoubleSpinBox(self)
        self.btnGetBP1.setValue(self.data.bp1)
        self.btnGetBP1.setRange(0, self.data.fs / 2)
        layout.addWidget(self.btnGetBP1,3,1)

        bp_to_lbl = QLabel("to",self)
        layout.addWidget(bp_to_lbl,3,2)

        self.btnGetBP2 = QDoubleSpinBox(self)
        self.btnGetBP2.setValue(self.data.bp2)
        self.btnGetBP2.setRange(0, self.data.fs / 2)
        layout.addWidget(self.btnGetBP2,3,3)

        notch_hz_lbl = QLabel("Hz",self)
        layout.addWidget(notch_hz_lbl,3,4)

        self.setLayout(layout)

        self.setSignalsSlots()

        self.show()

    def setSignalsSlots(self):
        """
        Setup signals and slots.
        """
        self.btnExit.clicked.connect(self.change)
        self.cbox_lp.toggled.connect(self.lp_filterChecked)
        self.cbox_hp.toggled.connect(self.hp_filterChecked)
        self.cbox_notch.toggled.connect(self.notch_filterChecked)
        self.cbox_bp.toggled.connect(self.bp_filterChecked)

    def lp_filterChecked(self):
        c = self.sender()
        if c.isChecked():
            self.data.do_lp = 1
            self.cbox_bp.setChecked(0)
        else:
            self.data.do_lp = 0

    def hp_filterChecked(self):
        c = self.sender()
        if c.isChecked():
            self.data.do_hp = 1
            self.cbox_bp.setChecked(0)
        else:
            self.data.do_hp = 0

    def notch_filterChecked(self):
        c = self.sender()
        if c.isChecked():
            self.data.do_notch = 1
        else:
            self.data.do_notch = 0

    def bp_filterChecked(self):
        c = self.sender()
        if c.isChecked():
            self.data.do_bp = 1
            self.cbox_lp.setChecked(0)
            self.cbox_hp.setChecked(0)
        else:
            self.data.do_bp = 0

    def change(self):
        """
        Checks to make sure values are legal and updates the FilterInfo object
        """
        hp = self.btnGetHP.value()
        lp = self.btnGetLP.value()
        bp1 = self.btnGetBP1.value()
        bp2 = self.btnGetBP2.value()
        if (lp > 0 and lp < self.data.fs / 2 and hp > 0 and hp < self.data.fs / 2):
            if lp - hp > 0:
                if self.data.do_lp:
                    self.data.lp = self.btnGetLP.value()
                if self.data.do_hp:
                    self.data.hp = self.btnGetHP.value()
        if bp1 > 0 and bp2 > 0 and bp1 < bp2:
            if self.data.do_bp:
                self.data.bp1 = bp1
                self.data.bp2 = bp2
        else:
            self.data.do_bp = 0
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
