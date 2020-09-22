from PyQt5.QtCore import Qt
from PyQt5 import uic

from PyQt5.QtWidgets import QWidget#, QDialogButtonBox, QFileDialog

# import numpy as np
# from matplotlib.backends.backend_qt5agg import FigureCanvas
# from matplotlib.figure import Figure

# from plot_utils import checkAnnotations
from ui_files.saveEdfOps import Ui_saveToEdf
from anonymizer import Anonymizer
# import sys

class SaveEdfOptions(QWidget):
    def __init__(self,data,parent):
        super().__init__()
        self.data = data
        self.parent = parent
        self.seo_ui = Ui_saveToEdf()
        self.seo_ui.setupUi(self)
        self.setupUI() # Show the GUI

    def setupUI(self):
        # self.sio_ui.resize(self.parent.width / 1.5, self.parent.height / 1.5)

        # Reset fields
        self.data.ptId = "X X X X" + " " * 730
        self.data.recInfo = "Startdate X X X X" + " " * 63
        self.data.startDate = "01.01.01"
        self.data.startTime = "01.01.01"

        self.setSigSlots()

    def setSigSlots(self):
        # set signals and slots
        self.seo_ui.cbox_orig.toggled.connect(self.cboxOrigChecked)
        self.seo_ui.cbox_anon.toggled.connect(self.cboxAnonChecked)
        self.seo_ui.btn_editFields.clicked.connect(self.openAnonEditor)
        self.seo_ui.btn_anonAndSave.clicked.connect(self.saveAndClose)

        self.seo_ui.cbox_anon.setChecked(1)

        self.show()

        """if (not self.parent.argv.export_png_file is None) and self.parent.init == 0:
            self.data.plotAnn = self.parent.argv.print_annotations
            self.data.linethick = self.parent.argv.line_thickness
            self.data.fontSize = self.parent.argv.font_size
            self.data.plotTitle = 1
            self.data.title = self.parent.argv.plot_title
            self.makePlot()
            self.printPlot()
        else:
            self.show()"""

    def cboxOrigChecked(self):
        if self.seo_ui.cbox_orig.isChecked():
            self.seo_ui.cbox_anon.setChecked(0)

    def cboxAnonChecked(self):
        if self.seo_ui.cbox_anon.isChecked():
            self.seo_ui.cbox_orig.setChecked(0)

    def openAnonEditor(self):
        # TODO: this should open the anon editor and also close this window
        self.parent.anon_win_open = 1
        self.parent.anon_ops = Anonymizer(self.data,self.parent)
        self.closeWindow()

    def saveAndClose(self):
        # TODO: do something when window is closed
        if self.seo_ui.cbox_orig.isChecked():
            with open(self.data.fn, 'rb') as f:
                file = f.read(200)
            self.data.ptId = file[8:88].decode("utf-8")
            self.data.recInfo = file[88:168].decode("utf-8")
            self.data.startDate = file[168:176].decode("utf-8")
            self.data.startTime = file[176:184].decode("utf-8")
        self.parent.save_sig_to_edf()
        self.closeWindow()

    def closeWindow(self):
        self.parent.saveedf_win_open = 0
        self.close()

    def closeEvent(self, event):
        # Called when the window is closed.
        self.parent.saveedf_win_open = 0
        event.accept()
