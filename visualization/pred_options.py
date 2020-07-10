from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import (QFileDialog, QVBoxLayout, QMessageBox, QWidget,
                                QPushButton, QCheckBox, QLabel, QInputDialog,
                                QSlider, QGridLayout, QSpinBox)


class PredictionOptions(QWidget):
    def __init__(self,pi,parent):
        super().__init__()
        self.left = 10
        self.top = 10
        self.title = 'Prediction Options'
        self.width = parent.width / 3
        self.height = parent.height / 4
        self.data = pi
        self.parent = parent
        self.nchns = self.parent.ci.nchns_to_plot
        self.setupUI()

    def setupUI(self):

        layout = QGridLayout()
        layout.setSpacing(4)

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        btnExit = QPushButton('Ok', self)
        btnExit.clicked.connect(self.check)
        layout.addWidget(btnExit,3,3)

        self.cbox_preds = QCheckBox("Plot predictions from file",self)

        self.cbox_model = QCheckBox("Plot model predictions",self)
        self.cbox_model.toggled.connect(self.model_filterChecked)
        self.cbox_model.setToolTip("Click to plot model predictions")
        #model_preds_valid = 0
        #if self.data.plot_model_preds == 1:
        #    model_preds_valid = self.data.check_preds_shape(self.data.model_preds, 1,
        #                        self.parent.max_time, self.parent.edf_info.fs, self.nchns)
        if self.data.plot_model_preds == 1:
            self.cbox_model.setChecked(True)
        elif self.data.plot_model_preds == 1:
            self.cbox_model.setChecked(False)
            self.data.plot_model_preds = 0

        layout.addWidget(self.cbox_model,0,0)

        buttonLoadPtFile = QPushButton("Load preprocessed data",self)
        buttonLoadPtFile.clicked.connect(self.loadPtData)
        buttonLoadPtFile.setToolTip("Click to load preprocessed data (as a torch tensor)")
        layout.addWidget(buttonLoadPtFile,0,1)

        self.labelLoadPtFile = QLabel("No data loaded.",self)
        if self.data.data_loaded == 1:
            self.labelLoadPtFile.setText(self.data.data_fn)
        layout.addWidget(self.labelLoadPtFile,0,2)

        buttonLoadModel = QPushButton("Load model",self)
        buttonLoadModel.clicked.connect(self.loadModel)
        buttonLoadModel.setToolTip("Click to load model")
        layout.addWidget(buttonLoadModel,1,1)

        self.labelLoadModel = QLabel("No model loaded.",self)
        if self.data.model_loaded == 1:
            self.labelLoadModel.setText(self.data.model_fn)
        layout.addWidget(self.labelLoadModel,1,2)

        self.cbox_preds.toggled.connect(self.preds_filterChecked)
        self.cbox_preds.setToolTip("Click to plot predictions from file")
        #loaded_preds_valid = 0
        #if self.data.plot_loaded_preds == 1:
        #    loaded_preds_valid = self.data.check_preds_shape(self.data.preds, 0,
        #                        self.parent.max_time, self.parent.edf_info.fs, self.nchns)
        if self.data.plot_loaded_preds == 1:
            self.cbox_preds.setChecked(True)
        elif self.data.plot_loaded_preds == 1:
            self.cbox_preds.setChecked(False)
            self.data.plot_loaded_preds = 0

        layout.addWidget(self.cbox_preds,2,0)

        buttonLoadPreds = QPushButton("Load predictions",self)
        buttonLoadPreds.clicked.connect(self.loadPreds)
        buttonLoadPreds.setToolTip("Click to load predictions")
        layout.addWidget(buttonLoadPreds,2,1)

        self.labelLoadPreds = QLabel("No predictions loaded.", self)
        if self.data.preds_loaded == 1:
            self.labelLoadPreds.setText(self.data.preds_fn)
        layout.addWidget(self.labelLoadPreds,2,2)

        self.setLayout(layout)

        self.show()

    def model_filterChecked(self):
        c = self.sender()
        if c.isChecked():
            if self.cbox_preds.isChecked():
                self.cbox_preds.setChecked(False)
                self.data.plot_loaded_preds = 0
            self.data.plot_model_preds = 1
        else:
            self.data.plot_model_preds = 0

    def preds_filterChecked(self):
        c = self.sender()
        if c.isChecked():
            if self.cbox_model.isChecked():
                self.cbox_model.setChecked(False)
                self.data.plot_model_preds = 0
            self.data.plot_loaded_preds = 1
        else:
            self.data.plot_loaded_preds = 0

    def loadPtData(self):
        """
        Load data for prediction
        """
        ptfile_fn = QFileDialog.getOpenFileName(self, 'Open file','.','Pytorch files (*.pt)')
        if ptfile_fn[0] == None or len(ptfile_fn[0]) == 0:
            return
        else:
            if len(ptfile_fn[0].split('/')[-1]) < 18:
                self.labelLoadPtFile.setText(ptfile_fn[0].split('/')[-1])
                self.data.data_fn = ptfile_fn[0].split('/')[-1]
            else:
                self.labelLoadPtFile.setText(ptfile_fn[0].split('/')[-1][0:15] + "...")
                self.data.data_fn = ptfile_fn[0].split('/')[-1][0:15] + "..."
            self.data.set_data(ptfile_fn[0])
            self.cbox_model.setChecked(True)
            self.cbox_preds.setChecked(False)

    def loadModel(self):
        """
        Load model for prediction
        """
        model_fn = QFileDialog.getOpenFileName(self, 'Open model','.','Pytorch files (*.pt)')
        if model_fn[0] == None or len(model_fn[0]) == 0:
            return
        else:
            if len(model_fn[0].split('/')[-1]) < 18:
                self.labelLoadModel.setText(model_fn[0].split('/')[-1])
                self.data.model_fn = model_fn[0].split('/')[-1]
            else:
                self.labelLoadModel.setText(model_fn[0].split('/')[-1][0:15] + "...")
                self.data.model_fn = model_fn[0].split('/')[-1][0:15] + "..."
            self.data.set_model(model_fn[0])
            self.cbox_model.setChecked(True)
            self.cbox_preds.setChecked(False)

    def loadPreds(self):
        """
        Loads predictions
        """
        preds_fn = QFileDialog.getOpenFileName(self, 'Open predictions','.','Pytorch files (*.pt)')
        if preds_fn[0] == None or len(preds_fn[0]) == 0:
            return
        else:
            if self.data.set_preds(preds_fn[0], self.parent.max_time,self.parent.edf_info.fs,self.nchns) == -1:
                self.parent.throwAlert("Predictions are not an even multiple of the samples in the .edf" +
                                "file you loaded or are the incorrect shape. Please check your file.")
            else:
                if len(preds_fn[0].split('/')[-1]) < 18:
                    self.labelLoadPreds.setText(preds_fn[0].split('/')[-1])
                    self.data.preds_fn = preds_fn[0].split('/')[-1]
                else:
                    self.labelLoadPreds.setText(preds_fn[0].split('/')[-1][0:15] + "...")
                    self.data.preds_fn = preds_fn[0].split('/')[-1][0:15] + "..."
                self.cbox_model.setChecked(False)
                self.cbox_preds.setChecked(True)

    def check(self):
        # check and return
        """
        Take loaded model and data and compute predictions
        """
        if self.data.plot_loaded_preds == 0 and self.data.plot_model_preds == 0:
            self.parent.predicted = 0
            self.closeWindow()
            self.parent.throwAlert("You have not chosen to plot any predictions.")
            self.parent.callmovePlot(0,0,0)
        elif self.data.plot_loaded_preds:
            if not self.data.preds_loaded:
                self.parent.throwAlert("Please load predictions.")
            else:
                loaded_preds_valid = self.data.check_preds_shape(self.data.preds, 0,
                                        self.parent.max_time, self.parent.edf_info.fs, self.nchns)
                if not loaded_preds_valid:
                    self.parent.predicted = 1
                    self.data.preds_to_plot = self.data.preds
                    self.parent.predLabel.setText("Predictions plotted.")
                    self.parent.callmovePlot(0,0,0)
                    self.closeWindow()
                elif loaded_preds_valid == -1:
                    self.parent.throwAlert("Predictions are not an even multiple of the samples in the .edf" +
                                    "file you loaded or are the incorrect shape. Please check your file.")
        else:
            if self.data.ready:
                preds_ret = self.data.predict(self.parent.max_time,self.parent.edf_info.fs,self.nchns)
                if preds_ret == -2:
                    self.parent.throwAlert("An error occured when trying to call the predict() " +
                                "function using your model. Please check your model and data.")
                elif preds_ret == -1:
                    self.parent.throwAlert("Predictions are not an even multiple of the samples in the .edf" +
                                    "file you loaded or are the incorrect shape. Please check your file.")
                else:
                    self.parent.predicted = 1
                    self.data.preds_to_plot = self.data.model_preds
                    self.parent.predLabel.setText("Predictions plotted.")
                    self.parent.callmovePlot(0,0,0)
                    self.closeWindow()
            elif not self.data.data_loaded:
                self.parent.throwAlert('Please load data.')
            else:
                self.parent.throwAlert('Please load a model.')

    def closeWindow(self):
        # For when zoom plot is open, if new predictions are loaded they will
        # make the roi box invisible
        if self.parent.btnZoom.text() == "Close zoom":
            self.parent.openZoomPlot()
            self.parent.openZoomPlot()
        self.parent.preds_win_open = 0
        self.close()
