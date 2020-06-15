from PyQt5.QtCore import Qt
from PyQt5 import uic

from PyQt5.QtWidgets import QWidget, QDialogButtonBox, QFileDialog

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

from plot_utils import checkAnnotations
import sys

class SaveImgOptions(QWidget):
    def __init__(self,data,parent):
        super().__init__()
        self.data = data
        self.parent = parent
        self.sio_ui = uic.loadUi("./visualization/ui_files/saveImg.ui", self) # Load the .ui file
        self.setupUI() # Show the GUI

    def setupUI(self):
        self.sio_ui.resize(self.parent.width / 1.5, self.parent.height / 1.5)

        self.m = PlotCanvas(self, width=7, height=7)
        self.sio_ui.plot_layout.addWidget(self.m)

        # set signals and slots - in .ui file
        self.sio_ui.okBtn.button(QDialogButtonBox.Ok).clicked.connect(self.printPlot)
        self.sio_ui.okBtn.button(QDialogButtonBox.Cancel).clicked.connect(self.closeWindow)

        self.ann_list = []
        self.aspan_list = []

        self.nchns = self.data.ci.nchns_to_plot
        self.fs = self.data.fs
        self.plotData = self.data.data
        self.count = self.data.count
        self.window_size = self.data.window_size
        self.y_lim = self.data.y_lim
        self.predicted = self.data.predicted
        self.thresh = self.data.thresh

        # add items to the combo boxes
        self.sio_ui.textSizeInput.addItems(["6pt","8pt","10pt", "12pt", "14pt", "16pt"])
        self.sio_ui.textSizeInput.setCurrentIndex(3)

        self.sio_ui.lineThickInput.addItems(["0.25px","0.5px","0.75px","1px","1.25px","1.5px","1.75px","2px"])
        self.sio_ui.lineThickInput.setCurrentIndex(1)

        self.sio_ui.annCbox.setChecked(1)

        self.makePlot()

        if (not self.parent.argv.export_png_file is None) and self.parent.init == 0:
            self.data.plotAnn = self.parent.argv.print_annotations
            self.data.linethick = self.parent.argv.line_thickness
            self.data.fontSize = self.parent.argv.font_size
            self.data.plotTitle = 1
            self.data.title = self.parent.argv.plot_title
            self.makePlot()
            self.printPlot()
        else:
            self.show()

    def annChecked(self):
        cbox = self.sender()
        if cbox.isChecked():
            self.data.plotAnn = 1
        else:
            self.data.plotAnn = 0
        self.makePlot()

    def titleChecked(self):
        if self.sio_ui.titleCbox.isChecked():
            self.data.title = self.sio_ui.titleInput.text()
        else:
            self.data.title = ""
        self.makePlot()

    def titleChanged(self):
        self.titleChecked()

    def chgLineThick(self):
        thickness = self.sio_ui.lineThickInput.currentText()
        thickness = float(thickness.split("px")[0])
        self.data.linethick = thickness
        self.makePlot()

    def chgTextSize(self):
        fontSize = self.sio_ui.textSizeInput.currentText()
        fontSize = int(fontSize.split("pt")[0])
        self.data.fontSize = fontSize
        self.makePlot()

    def makePlot(self):
        """ Makes the plot with the given specifications """
        self.m.fig.clf()
        self.ax = self.m.fig.add_subplot(self.m.gs[0])

        del(self.ax.lines[:])
        for i, a in enumerate(self.ann_list):
            a.remove()
        self.ann_list[:] = []
        for aspan in self.aspan_list:
            aspan.remove()
        self.aspan_list[:] = []

        for i in range(self.nchns):
            if self.data.plotAnn:
                self.ax.plot(self.plotData[i, self.count * self.fs:(self.count + 1) * self.fs * self.window_size]
                             + (i + 1) * self.y_lim, '-', linewidth=self.data.linethick, color=self.data.ci.colors[i])
                self.ax.set_ylim([-self.y_lim, self.y_lim * (self.nchns + 1)])
                self.ax.set_yticks(np.arange(0, (self.nchns + 2)*self.y_lim, step=self.y_lim))
                self.ax.set_yticklabels(
                    self.data.ci.labels_to_plot, fontdict=None, minor=False, fontsize=self.data.fontSize)
            else:
                self.ax.plot(self.plotData[i, self.count * self.fs:(self.count + 1) * self.fs * self.window_size]
                             + (i) * self.y_lim, '-', linewidth=self.data.linethick, color=self.data.ci.colors[i])
                self.ax.set_ylim([-self.y_lim, self.y_lim * (self.nchns)])
                self.ax.set_yticks(np.arange(0, (self.nchns + 1)*self.y_lim, step=self.y_lim))
                self.ax.set_yticklabels(
                    self.data.ci.labels_to_plot[1:], fontdict=None, minor=False, fontsize=self.data.fontSize)

            width = 1 / (self.nchns + 2)
            if self.predicted == 1:
                starts, ends, chns = self.data.pi.compute_starts_ends_chns(self.thresh,
                                                                      self.count, self.window_size, fs, nchns)
                for k in range(len(starts)):
                    if self.pi.pred_by_chn:
                        if chns[k][i]:
                            if i == plotData.shape[0] - 1:
                                self.aspan_list.append(self.ax.axvspan(starts[k] - self.count * self.fs, ends[k] - self.count * self.fs,
                                                                       ymin=width*(i+1.5), ymax=1, color='paleturquoise', alpha=1))
                            else:
                                self.aspan_list.append(self.ax.axvspan(starts[k] - self.count * self.fs, ends[k] - self.count * self.fs,
                                                                       ymin=width*(i+1.5), ymax=width*(i+2.5), color='paleturquoise', alpha=1))
                            x_vals = range(
                                int(starts[k]) - self.count * self.fs, int(ends[k]) - self.count * self.fs)
                            self.ax.plot(x_vals, self.plotData[i, int(starts[k]):int(ends[k])] + i*self.y_lim + self.y_lim,
                                         '-', linewidth=self.linethick * 2, color=self.data.ci.colors[i])
                    else:
                        self.aspan_list.append(self.ax.axvspan(
                            starts[k] - self.count * self.fs, ends[k] - self.count * self.fs, color='paleturquoise', alpha=0.5))

        self.ax.set_xlim([0, self.fs*self.window_size])
        step_size = self.fs  # Updating the x labels with scaling
        step_width = 1
        if self.window_size >= 15 and self.window_size <= 25:
            step_size = step_size * 2
            step_width = step_width * 2
        elif self.window_size > 25:
            step_size = step_size * 3
            step_width = step_width * 3
        self.ax.set_xticks(np.arange(0, self.window_size *
                                     self.fs + 1, step=step_size))
        self.ax.set_xticklabels(np.arange(self.count, self.count + self.window_size + 1,
                                          step=step_width), fontdict=None, minor=False, fontsize=self.data.fontSize)
        self.ax.set_xlabel("Time (s)", fontsize=self.data.fontSize)
        self.ax.set_title(self.data.title, fontsize=self.data.fontSize)

        if self.data.plotAnn:
            ann, idx_w_ann = checkAnnotations(
                self.count, self.window_size, self.parent.edf_info)
            # font_size = 10 - self.window_size / 5
            font_size = self.data.fontSize - 4
            # Add in annotations
            if len(ann) != 0:
                ann = np.array(ann).T
                txt = ""
                int_prev = int(float(ann[0, 0]))
                for i in range(ann.shape[1]):
                    int_i = int(float(ann[0, i]))
                    if int_prev == int_i:
                        txt = txt + "\n" + ann[2, i]
                    else:
                        if idx_w_ann[int_prev - self.count] and int_prev % 2 == 1:
                            self.ann_list.append(self.ax.annotate(txt, xy=(
                                (int_prev - self.count)*self.fs, -self.y_lim / 2 + self.y_lim), color='black', size=font_size))
                        else:
                            self.ann_list.append(self.ax.annotate(txt, xy=(
                                (int_prev - self.count)*self.fs, -self.y_lim / 2), color='black', size=font_size))
                        txt = ann[2, i]
                    int_prev = int_i
                if txt != "":
                    if idx_w_ann[int_i - self.count] and int_i % 2 == 1:
                        self.ann_list.append(self.ax.annotate(txt, xy=(
                            (int_i - self.count)*self.fs, -self.y_lim / 2 + self.y_lim), color='black', size=font_size))
                    else:
                        self.ann_list.append(self.ax.annotate(
                            txt, xy=((int_i - self.count)*self.fs, -self.y_lim / 2), color='black', size=font_size))

        self.m.draw()

    def printPlot(self):
        """ Saves the plot and exits """
        if (not self.parent.argv.export_png_file is None) and self.parent.init == 0:
            self.ax.figure.savefig(self.parent.argv.export_png_file, bbox_inches='tight', dpi=300)
            if self.parent.argv.show == 0:
                sys.exit()
            else:
                self.closeWindow()
        else:
            file = QFileDialog.getSaveFileName(self, 'Save File')
            if len(file[0]) == 0 or file[0] == None:
                return
            else:
                self.ax.figure.savefig(file[0] + ".png", bbox_inches='tight', dpi=300)
                self.closeWindow()

    def resetInitialState(self):
        self.data.plotAnn = 1
        self.data.linethick = 0.5
        self.data.fontSize = 12
        self.data.plotTitle = 0
        self.data.title = ""

    def closeWindow(self):
        self.parent.spec_win_open = 0
        self.resetInitialState()
        self.close()

    def closeEvent(self, event):
        # Called when the window is closed.
        self.parent.saveimg_win_open = 0
        self.resetInitialState()
        event.accept()

class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi,
                          constrained_layout=False)
        self.gs = self.fig.add_gridspec(1, 1, wspace=0.0, hspace=0.0)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
