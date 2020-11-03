import PyQt5
from PyQt5.QtCore import Qt, QTime, QDate

from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog,
                             QMessageBox, QWidget, QPushButton, QLabel,
                             QGridLayout, QLineEdit, QSpinBox,
                             QTimeEdit, QFrame, QTextEdit, QDateEdit, QGroupBox,
                             QRadioButton, QHBoxLayout, QCheckBox)
from PyQt5.QtGui import QFont
from matplotlib.backends.qt_compat import QtCore, QtWidgets

import sys

from anonymize_edf import anonymizeFile

class Anonymizer(QWidget):

    def __init__(self, data, parent):
        super().__init__()
        self.left = 10
        self.top = 10
        self.title = 'EDF Anonymizer'
        sizeObject = QtWidgets.QDesktopWidget().screenGeometry(-1)
        self.width = sizeObject.width() * 0.2
        self.height = sizeObject.height() * 0.2
        self.field_defaults = [bytes(" " * 80, 'utf-8'), bytes(" " * 80, 'utf-8'),
                                bytes("01.01.01", 'utf-8'), bytes("01.01.01", 'utf-8')]

        self.data = data
        self.parent = parent

        self.MONTHS = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP",
                        "OCT","NOV","DEC"]

        self.initUI()

    def initUI(self):
        """
        Setup the UI
        """
        layout = QGridLayout()

        font = QFont()
        font.setPointSize(16)
        lblInfo = QLabel("Update fields and click anonymize " +
                        "and save to select location for new file.")
        lblInfo.setAlignment(Qt.AlignCenter)
        lblInfo.setFont(font)
        layout.addWidget(lblInfo,0,0,1,3)

        # ---- Add in all subfields ---- #
        ptId_recInfo_fields = QGridLayout()

        lbloriginal = QLabel("Original")
        lbloriginal.setAlignment(Qt.AlignCenter)
        ptId_recInfo_fields.addWidget(lbloriginal,0,2)
        lblnew = QLabel("Modified")
        lblnew.setAlignment(Qt.AlignCenter)
        ptId_recInfo_fields.addWidget(lblnew,0,3)
        self.cbox_copyoriginal = QCheckBox("Copy original values")
        self.cbox_copyoriginal.setEnabled(0)
        ptId_recInfo_fields.addWidget(self.cbox_copyoriginal,1,3)
        self.cbox_setdefaults = QCheckBox("Set default values")
        self.cbox_setdefaults.setEnabled(0)
        ptId_recInfo_fields.addWidget(self.cbox_setdefaults,2,3)

        lblPtId = QLabel("Patient ID:")
        ptId_recInfo_fields.addWidget(lblPtId,3,0,5,1)
        lblRecInfo = QLabel("Recording information:")
        ptId_recInfo_fields.addWidget(lblRecInfo,8,0,5,1)

        lblhospcode = QLabel("Hospital code:")
        lblsex = QLabel("Sex:")
        lbldob = QLabel("Date of birth:")
        lblptname = QLabel("Patient name:")
        lblother = QLabel("Other:")
        self.ptId_lbls = [lblhospcode, lblsex, lbldob, lblptname,lblother]
        hospcodeedit = QLineEdit()
        groupBoxSexEdit = QGroupBox()
        self.radioPtIdF = QRadioButton("F")
        self.radioPtIdM = QRadioButton("M")
        self.radioPtIdX = QRadioButton("X")
        self.radioPtIdX.setChecked(True)
        hbox = QHBoxLayout()
        hbox.addWidget(self.radioPtIdF)
        hbox.addWidget(self.radioPtIdM)
        hbox.addWidget(self.radioPtIdX)
        groupBoxSexEdit.setLayout(hbox)
        groupBoxDateEdit = QGroupBox()
        self.radioPtIdDate = QRadioButton("")
        self.radioPtIdDateX = QRadioButton("X")
        self.radioPtIdDateX.setChecked(True)
        self.dobedit = QDateEdit(QDate(2001, 1, 1))
        self.dobedit.setDisplayFormat("MM/dd/yyyy")
        hbox2 = QHBoxLayout()
        hbox2.addWidget(self.radioPtIdDate)
        hbox2.addWidget(self.dobedit)
        hbox2.addWidget(self.radioPtIdDateX)
        groupBoxDateEdit.setLayout(hbox2)
        ptnameedit = QLineEdit()
        ptidotheredit = QLineEdit()
        self.ptId_fields = [hospcodeedit,groupBoxSexEdit,groupBoxDateEdit,ptnameedit,ptidotheredit]
        hospcode = QLineEdit()
        groupBoxSex = QGroupBox()
        self.radioPtIdF2 = QRadioButton("F")
        self.radioPtIdM2 = QRadioButton("M")
        self.radioPtIdX2 = QRadioButton("X")
        self.radioPtIdF2.setChecked(True)
        hbox3 = QHBoxLayout()
        hbox3.addWidget(self.radioPtIdF2)
        hbox3.addWidget(self.radioPtIdM2)
        hbox3.addWidget(self.radioPtIdX2)
        groupBoxSex.setLayout(hbox3)
        groupBoxDate = QGroupBox()
        self.radioPtIdDate2 = QRadioButton("")
        self.radioPtIdDateX2 = QRadioButton("X")
        self.radioPtIdDate2.setChecked(True)
        self.dob = QDateEdit(QDate(2001, 1, 1))
        self.dob.setDisplayFormat("MM/dd/yyyy")
        hbox4 = QHBoxLayout()
        hbox4.addWidget(self.radioPtIdDate2)
        hbox4.addWidget(self.dob)
        hbox4.addWidget(self.radioPtIdDateX2)
        groupBoxDate.setLayout(hbox4)
        ptname = QLineEdit()
        ptidother = QLineEdit()
        self.oldptId_fields = [hospcode,groupBoxSex,groupBoxDate,ptname,ptidother]

        for i,l in enumerate(self.ptId_lbls):
            ptId_recInfo_fields.addWidget(l,i + 3,1)
            ptId_recInfo_fields.addWidget(self.oldptId_fields[i],i + 3,2)
            ptId_recInfo_fields.addWidget(self.ptId_fields[i],i + 3,3)
            self.oldptId_fields[i].setDisabled(1)
            self.ptId_fields[i].setDisabled(1)

        ptId_recInfo_fields.addWidget(QHLine(), len(self.ptId_lbls) + 3, 0, 1, 4)

        lblstartdate = QLabel("Startdate:")
        lblhospadmincode = QLabel("Hospital admin code:")
        lbltechcode = QLabel("Technician code:")
        lblequip = QLabel("Equipment code:")
        lblother = QLabel("Other:")
        self.recInfo_lbls = [lblstartdate, lblhospadmincode, lbltechcode, lblequip, lblother]
        groupBoxDateEditrecInfo = QGroupBox()
        self.recInfoDate = QRadioButton("")
        self.recInfoDateX = QRadioButton("X")
        self.recInfoDateX.setChecked(True)
        self.startdateedit = QDateEdit(QDate(2001, 1, 1))
        self.startdateedit.setDisplayFormat("MM/dd/yyyy")
        hb = QHBoxLayout()
        hb.addWidget(self.recInfoDate)
        hb.addWidget(self.startdateedit)
        hb.addWidget(self.recInfoDateX)
        groupBoxDateEditrecInfo.setLayout(hb)
        hospadmincodeedit = QLineEdit()
        techcodeedit = QLineEdit()
        equipcodeedit = QLineEdit()
        recinfootheredit = QLineEdit()
        self.recInfo_fields = [groupBoxDateEditrecInfo, hospadmincodeedit, techcodeedit, equipcodeedit, recinfootheredit]
        groupBoxDaterecInfo = QGroupBox()
        self.recInfoDate2 = QRadioButton("")
        self.recInfoDateX2 = QRadioButton("X")
        self.recInfoDate2.setChecked(True)
        self.startdate = QDateEdit(QDate(2001, 1, 1))
        self.startdate.setDisplayFormat("MM/dd/yyyy")
        hb = QHBoxLayout()
        hb.addWidget(self.recInfoDate2)
        hb.addWidget(self.startdate)
        hb.addWidget(self.recInfoDateX2)
        groupBoxDaterecInfo.setLayout(hb)
        hospadmincode = QLineEdit()
        techcode = QLineEdit()
        equipcode = QLineEdit()
        recinfoother = QLineEdit()
        self.oldrecInfo_fields = [groupBoxDaterecInfo, hospadmincode, techcode, equipcode, recinfoother]
        for i,l in enumerate(self.recInfo_lbls):
            ptId_recInfo_fields.addWidget(l,i + len(self.oldptId_fields) + 4,1)
            ptId_recInfo_fields.addWidget(self.oldrecInfo_fields[i],i + len(self.oldptId_fields) + 4,2)
            ptId_recInfo_fields.addWidget(self.recInfo_fields[i],i + len(self.oldptId_fields) + 4,3)
            self.recInfo_fields[i].setDisabled(1)
            self.oldrecInfo_fields[i].setDisabled(1)

        ptId_recInfo_fields.addWidget(QHLine(), len(self.ptId_lbls) + len(self.oldptId_fields) + 4, 0, 1, 4)
        #layout.addLayout(ptIdFields, 3, 0, 1,3)
        #layout.addLayout(recInfoFields, 4, 0, 1,3)
        layout.addLayout(ptId_recInfo_fields, 3, 0, 1,3)

        lblStartDate = QLabel("Start date:")
        lblStartTime = QLabel("Start time:")
        oldinputStartDate = QDateEdit(QDate(2001, 1, 1))
        oldinputStartDate.setDisplayFormat("MM/dd/yyyy")
        oldinputStartTime = QTimeEdit(QTime(1, 1, 1))
        oldinputStartTime.setDisplayFormat("hh:mm:ss")
        inputStartDate = QDateEdit(QDate(2001, 1, 1))
        inputStartDate.setDisplayFormat("MM/dd/yyyy")
        inputStartTime = QTimeEdit(QTime(1, 1, 1))
        inputStartTime.setDisplayFormat("hh:mm:ss")

        self.dateTime_lbls = [lblStartDate,lblStartTime]
        self.olddateTimefield_inputs = [oldinputStartDate,oldinputStartTime]
        self.dateTimefield_inputs = [inputStartDate,inputStartTime]
        for i,l in enumerate(self.dateTime_lbls):
            ptId_recInfo_fields.addWidget(l,i + len(self.oldptId_fields) + len(self.ptId_lbls) + 6,1)
            self.olddateTimefield_inputs[i].setDisabled(1)
            self.dateTimefield_inputs[i].setDisabled(1)
            ptId_recInfo_fields.addWidget(self.olddateTimefield_inputs[i],i + len(self.oldptId_fields) + len(self.ptId_lbls) + 6,2)
            ptId_recInfo_fields.addWidget(self.dateTimefield_inputs[i],i + len(self.oldptId_fields) + len(self.ptId_lbls) + 6,3)

        # Set defaults
        self.ptId_fields[0].setText("X")

        self.ptId_fields[3].setText("X")
        self.ptId_fields[4].setText("")

        self.recInfo_fields[1].setText("X")
        self.recInfo_fields[2].setText("X")
        self.recInfo_fields[3].setText("X")
        self.recInfo_fields[4].setText("")

        # ---- -------------------- ---- #

        # self.btn_loadfile = QPushButton("Select file",self)
        # layout.addWidget(self.btn_loadfile,1,1)

        self.lblFn = QLabel("No file loaded.")
        self.lblFn.setFont(font)
        layout.addWidget(self.lblFn,1,1,1,1)
        layout.addWidget(QHLine(), 2, 0, 1, 3)

        layout.addWidget(QHLine(), len(self.dateTime_lbls) + 5, 0, 1, 3)
        self.btn_anonfile = QPushButton("Anonymize and save file",self)
        layout.addWidget(self.btn_anonfile,len(self.dateTime_lbls) + 6,0,1,3)
        self.btn_anonfile.setDisabled(True)

        self.setWindowTitle(self.title)
        # wid = QWidget(self)
        # self.setCentralWidget(wid)
        self.setLayout(layout)

        self.setSignalsSlots()
        self.show()

    def setSignalsSlots(self):
        """
        Set up the signals and slots.
        """
        # self.btn_loadfile.clicked.connect(self.loadFile)
        self.btn_anonfile.clicked.connect(self.anonFile)
        self.cbox_copyoriginal.toggled.connect(self.copyOriginal)
        self.cbox_setdefaults.toggled.connect(self.setDefaults)
        self.loadFile()

    def loadFile(self):
        """
        Function to select .edf file
        """
        """name = QFileDialog.getOpenFileName(self, 'Open file', '.', 'EDF files (*.edf)')
        name = name[0]
        if name == None or len(name) == 0:
            return
        else:"""
        name = self.data.fn
        self.input_fn = name
        self.input_fn_text = name.split('/')[-1]
        if len(name.split('/')[-1]) > 15:
            self.input_fn_text = name.split('/')[-1][0:15] + "..."
        self.lblFn.setText(self.input_fn_text)
        self.btn_anonfile.setDisabled(0)
        self.cbox_copyoriginal.setEnabled(1)
        self.cbox_setdefaults.setEnabled(1)
        self.cbox_setdefaults.setChecked(1)
        # Setup all fields to anonymize
        for i in range(len(self.ptId_fields)):
            self.ptId_fields[i].setDisabled(0)
        for i in range(len(self.recInfo_fields)):
            self.recInfo_fields[i].setDisabled(0)
        for i in range(len(self.dateTime_lbls)):
            self.dateTimefield_inputs[i].setDisabled(0)
        # Open the file
        with open(self.input_fn, 'rb') as f:
            file = f.read(200)
        file = bytearray(file)
        ptIdText = file[8:88].decode("utf-8").split(" ")
        recInfoText = file[88:168].decode("utf-8").split(" ")
        # Error checking
        if len(ptIdText) > 1:
            if not (ptIdText[1] in {"X","F","M"}):
                ptIdText[1] = "X"
            if len(ptIdText) > 2:
                if self._validDate(ptIdText[2]) == -1:
                    ptIdText[2] = "01-JAN-2001"
        if len(ptIdText) == 0:
            ptIdText = ["X","X","01-JAN-2001","X"]
        elif len(ptIdText) == 1:
            ptIdText.append("X")
            ptIdText.append("01-JAN-2001")
            ptIdText.append("X")
        elif len(ptIdText) == 2:
            ptIdText.append("01-JAN-2001")
            ptIdText.append("X")
        elif len(ptIdText) == 3:
            ptIdText.append("X")
        elif len(ptIdText) >= 4:
            if ptIdText[0] == " " or ptIdText[0] == "":
                ptIdText[0] = "X"
            if ptIdText[3] == " " or ptIdText[3] == "":
                ptIdText[3] = "X"

        if len(recInfoText) > 1:
            if self._validDate(recInfoText[1]) == -1:
                recInfoText[1] = "01-JAN-2001"
        if len(recInfoText) == 0:
            recInfoText = ["Startdate","01-JAN-2001","X","X","X"]
        elif len(recInfoText) == 1:
            recInfoText.append("01-JAN-2001")
            recInfoText.append("X")
            recInfoText.append("X")
            recInfoText.append("X")
        elif len(recInfoText) == 2:
            recInfoText.append("X")
            recInfoText.append("X")
            recInfoText.append("X")
        elif len(recInfoText) == 3:
            recInfoText.append("X")
            recInfoText.append("X")
        elif len(recInfoText) == 4:
            recInfoText.append("X")
        elif len(recInfoText) >= 5:
            if recInfoText[2] == " " or recInfoText[2] == "":
                recInfoText[2] = "X"
            if recInfoText[3] == " " or recInfoText[3] == "":
                recInfoText[3] = "X"
            if recInfoText[4] == " " or recInfoText[4] == "":
                recInfoText[4] = "X"
        self.oldptId_fields[0].setText(ptIdText[0])
        if ptIdText[1] == "F":
            self.radioPtIdF2.setChecked(1)
        elif ptIdText[1] == "M":
            self.radioPtIdM2.setChecked(1)
        else:
            self.radioPtIdX2.setChecked(1)
        if ptIdText[2] == "X":
            self.radioPtIdDateX2.setChecked(1)
        else:
            self.radioPtIdDate2.setChecked(1)
            yr = int(ptIdText[2].split("-")[2])
            mth = self.MONTHS.index(ptIdText[2].split("-")[1]) + 1
            day = int(ptIdText[2].split("-")[0])
            self.dob.setDate(QDate(yr,mth,day))
        self.oldptId_fields[3].setText(ptIdText[3])
        if len(ptIdText) > 4:
            self.oldptId_fields[4].setText("".join(ptIdText[4:]))
        else:
            self.oldptId_fields[4].setText("")

        if recInfoText[1] == "X":
            self.recInfoDateX2.setChecked(1)
        else:
            self.recInfoDate2.setChecked(1)
            yr = int(recInfoText[1].split("-")[2])
            mth = self.MONTHS.index(recInfoText[1].split("-")[1]) + 1
            day = int(recInfoText[1].split("-")[0])
            self.startdate.setDate(QDate(yr,mth,day))
        self.oldrecInfo_fields[1].setText(recInfoText[2])
        self.oldrecInfo_fields[2].setText(recInfoText[3])
        self.oldrecInfo_fields[3].setText(recInfoText[4])
        if len(recInfoText) > 5:
            self.oldrecInfo_fields[4].setText("".join(recInfoText[5:]))
        else:
            self.oldrecInfo_fields[4].setText("")

        yrs = int(file[174:176].decode("utf-8"))
        if yrs > 20:
            yrs = yrs + 1900
        else:
            yrs = yrs + 2000
        mths = int(file[171:173].decode("utf-8"))
        dys = int(file[168:170].decode("utf-8"))
        hrs = int(file[176:178].decode("utf-8"))
        min = int(file[179:181].decode("utf-8"))
        sec = int(file[182:184].decode("utf-8"))
        self.olddateTimefield_inputs[0].setDate(QDate(yrs,mths,dys))
        self.olddateTimefield_inputs[1].setTime(QTime(hrs,min,sec))


    def anonFile(self):
        """
        Chooses location to create file and anonymizes selected edf file.
        """
        # file = QFileDialog.getSaveFileName(self, 'Save File')
        # if file == None or len(file[0]) == 0:
        #    pass
        #else:
        # self.output_fn = file[0].split('.')[0] + ".edf"
        # Get info from user
        ptId = ""
        recInfo = ""
        startTime = ""
        startDate = self.dateTimefield_inputs[0].date().toString("dd.MM.yy")
        startTime = self.dateTimefield_inputs[1].time().toString("hh.mm.ss")

        if self.radioPtIdF.isChecked():
            sex = "F"
        elif self.radioPtIdM.isChecked():
            sex = "M"
        else:
            sex = "X"
        if self.radioPtIdDateX.isChecked():
            ptIdDate = "X"
        else:
             ptIdDate = self.dobedit.date().toString("dd-MMM-yyyy").upper()

        if self.ptId_fields[0].text() == "" or self.ptId_fields[0].text() == " ":
            self.ptId_fields[0].setText("X")
        if self.ptId_fields[3].text() == "" or self.ptId_fields[3].text() == " ":
            self.ptId_fields[3].setText("X")

        ptId = (self.ptId_fields[0].text().replace(" ","_") + " " + sex + " " + ptIdDate + " "
                + self.ptId_fields[3].text().replace(" ","_") + " " + self.ptId_fields[4].text().replace(" ","_"))

        if len(ptId) < 80:
            ptId = ptId + " " * (80 - len(ptId))
        elif len(ptId) > 80:
            self.throwAlert("The patient ID fields must be less than 80 characters. You have "
                    + str(len(ptId)) + " characters. Please edit your fields and try again.")
            return

        if self.recInfoDateX.isChecked():
            recInfoDate = "X"
        else:
            recInfoDate = self.startdateedit.date().toString("dd-MMM-yyyy").upper()

        if self.recInfo_fields[1].text() == "" or self.recInfo_fields[1].text() == " ":
            self.recInfo_fields[1].setText("X")
        if self.recInfo_fields[2].text() == "" or self.recInfo_fields[2].text() == " ":
            self.recInfo_fields[2].setText("X")
        if self.recInfo_fields[3].text() == "" or self.recInfo_fields[3].text() == " ":
            self.recInfo_fields[3].setText("X")
        recInfo = ("Startdate " + recInfoDate + " " + self.recInfo_fields[1].text().replace(" ","_")
                    + " " + self.recInfo_fields[2].text().replace(" ","_") + " " + self.recInfo_fields[3].text().replace(" ","_")
                    + " " + self.recInfo_fields[4].text().replace(" ","_"))
        if len(recInfo) < 80:
            recInfo = recInfo + " " * (80 - len(recInfo))
        elif len(recInfo) > 80:
            self.throwAlert("The recording information fields must be less than 80 characters. You have "
                + str(len(recInfo)) + " characters. Please edit your fields and try again.")
            return

        self.data.ptId = ptId
        self.data.recInfo = recInfo
        self.data.startDate = startDate
        self.data.startTime = startTime
        self.parent.save_sig_to_edf()
        self.closeWindow()
        """
        anonymizeFile(self.input_fn, self.output_fn, ptId, recInfo, startDate, startTime)
        self.throwAlert("Done", "File saved to: \n" + self.output_fn)
        self.lblFn.setText("No file loaded.")
        self.btn_anonfile.setDisabled(1)

        self.cbox_copyoriginal.setChecked(0)
        self.cbox_copyoriginal.setEnabled(0)
        self.cbox_setdefaults.setEnabled(0)
        self.cbox_setdefaults.setChecked(0)

        # Reset defaults
        self.ptId_fields[0].setText("X")
        self.radioPtIdX.setChecked(1)
        self.radioPtIdDateX.setChecked(1)
        self.dobedit.setDate(QDate(2001, 1, 1))
        self.ptId_fields[3].setText("X")
        self.ptId_fields[4].setText("")

        self.recInfoDate.setChecked(1)
        self.startdateedit.setDate(QDate(2001, 1, 1))
        self.recInfo_fields[1].setText("X")
        self.recInfo_fields[2].setText("X")
        self.recInfo_fields[3].setText("X")
        self.recInfo_fields[4].setText("")

        self.dateTimefield_inputs[0].setDate(QDate(2001, 1, 1))
        self.dateTimefield_inputs[1].setTime(QTime(1,1,1))

        self.oldptId_fields[0].setText("X")
        self.radioPtIdX2.setChecked(1)
        self.radioPtIdDateX2.setChecked(1)
        self.dob.setDate(QDate(2001, 1, 1))
        self.oldptId_fields[3].setText("X")
        self.oldptId_fields[4].setText("")

        self.recInfoDate2.setChecked(1)
        self.startdate.setDate(QDate(2001, 1, 1))
        self.oldrecInfo_fields[1].setText("X")
        self.oldrecInfo_fields[2].setText("X")
        self.oldrecInfo_fields[3].setText("X")
        self.oldrecInfo_fields[4].setText("")

        self.olddateTimefield_inputs[0].setDate(QDate(2001, 1, 1))
        self.olddateTimefield_inputs[1].setTime(QTime(1,1,1))


        # Disable editing blocks
        for i in range(len(self.ptId_fields)):
            self.ptId_fields[i].setDisabled(1)
        for i in range(len(self.recInfo_fields)):
            self.recInfo_fields[i].setDisabled(1)
        for i in range(len(self.dateTime_lbls)):
            self.dateTimefield_inputs[i].setDisabled(1)"""

    def closeWindow(self):
        self.parent.anon_win_open = 0
        self.close()

    def closeEvent(self, event):
        # Called when the window is closed.
        self.parent.anon_win_open = 0
        event.accept()

    def copyOriginal(self):
        """
        Copy original values to modified fields.
        """
        if self.cbox_copyoriginal.isChecked():
            if self.cbox_setdefaults.isChecked():
                self.cbox_setdefaults.setChecked(0)
            self.ptId_fields[0].setText(self.oldptId_fields[0].text())
            if self.radioPtIdF2.isChecked():
                self.radioPtIdF.setChecked(1)
            elif self.radioPtIdM2.isChecked():
                self.radioPtIdM.setChecked(1)
            else:
                self.radioPtIdX.setChecked(1)
            if self.radioPtIdDateX2.isChecked():
                self.radioPtIdDateX.setChecked(1)
            else:
                self.radioPtIdDate.setChecked(1)
            self.dobedit.setDate(self.dob.date())
            self.ptId_fields[3].setText(self.oldptId_fields[3].text())
            self.ptId_fields[4].setText(self.oldptId_fields[4].text())

            if self.recInfoDateX2.isChecked():
                self.recInfoDateX.setChecked(1)
            else:
                self.recInfoDate.setChecked(1)
            self.startdateedit.setDate(self.startdate.date())
            self.recInfo_fields[1].setText(self.oldrecInfo_fields[1].text())
            self.recInfo_fields[2].setText(self.oldrecInfo_fields[2].text())
            self.recInfo_fields[3].setText(self.oldrecInfo_fields[3].text())
            self.recInfo_fields[4].setText(self.oldrecInfo_fields[4].text())

            self.dateTimefield_inputs[0].setDate(self.olddateTimefield_inputs[0].date())
            self.dateTimefield_inputs[1].setTime(self.olddateTimefield_inputs[1].time())

    def setDefaults(self):
        """
        Set all edit fields to default values.
        """
        if self.cbox_setdefaults.isChecked():
            if self.cbox_copyoriginal.isChecked():
                self.cbox_copyoriginal.setChecked(0)
            self.ptId_fields[0].setText("X")
            self.radioPtIdX.setChecked(1)
            self.radioPtIdDateX.setChecked(1)
            self.ptId_fields[3].setText("X")
            self.ptId_fields[4].setText("")

            self.recInfoDateX.setChecked(1)
            self.recInfo_fields[1].setText("X")
            self.recInfo_fields[2].setText("X")
            self.recInfo_fields[3].setText("X")
            self.recInfo_fields[4].setText("")

            self.dateTimefield_inputs[0].setDate(QDate(2001,1,1))
            self.dateTimefield_inputs[1].setTime(QTime(1,1,1))

    def _validDate(self,datetext):
        date = datetext.split("-")
        if len(date) != 3:
            return -1
        if len(date[2]) != 4:
            return -1
        if len(date[0]) != 2:
            return -1
        if self.MONTHS.index(date[1]) == -1:
            return -1
        return 0

    def throwAlert(self, msg, text = ""):
        """
        Throws an alert to the user.
        """
        alert = QMessageBox()
        alert.setIcon(QMessageBox.Information)
        alert.setText(msg)
        alert.setInformativeText(text)
        alert.setWindowTitle("Warning")
        alert.exec_()
        self.resize( self.sizeHint() )

class QHLine(QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)
"""
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainPage(app)
    sys.exit(app.exec_())
"""
