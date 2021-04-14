import numpy as np
import pyedflib

def _check_label(label, label_list):
    """
    Checks if a label is in the label list
    """
    label_CAPS = {k.upper(): v for k, v in label_list.items()}
    ret = _check_label_helper(label, label_CAPS)
    if ret == -1:
        labels_noEEG = {}
        labels_noRef = {}
        for k,v in label_CAPS.items():
            loc = k.find("EEG ")
            if loc != -1:
                k2 = k[loc+4:]
                labels_noEEG[k2] = label_CAPS[k]
            else:
                labels_noEEG[k] = label_CAPS[k]
        ret = _check_label_helper(label, labels_noEEG)
        if ret == -1:
            for k,v in labels_noEEG.items():
                loc = k.find("-REF")
                if loc != -1:
                    k2 = k[0:loc]
                    labels_noRef[k2] = labels_noEEG[k]
                else:
                    labels_noRef[k] = labels_noEEG[k]
            ret = _check_label_helper(label, labels_noRef)
            if ret == -1:
                label2 = ""
                if label == "T7":
                    label2 = "T3"
                if label == "P7":
                    label2 = "T5"
                if label == "T8":
                    label2 = "T4"
                if label == "P8":
                    label2 = "T6"
                ret = _check_label_helper(label2, label_CAPS)
                if ret == -1:
                    ret = _check_label_helper(label2, labels_noEEG)
                    if ret == -1:
                        ret = _check_label_helper(label2, labels_noRef)
    return ret

def _check_label_helper(label, label_list):
    if label in label_list:
        return label_list[label]
    return -1

class ChannelInfo():
    """ Data structure for holding information relevant to selecting which signals to plot """

    def __init__(self):
        # Info from parent
        self.chns2labels = []
        self.labels2chns = []
        self.fs = 0
        self.max_time = 0
        self.edf_fn = ""

        self.total_nchns = 0
        self.list_of_chns = []
        self.convertedChnNames = []
        self.labelsBIP1020 = ["CZ-PZ","FZ-CZ","P4-O2","C4-P4","F4-C4","FP2-F4",
                              "P3-O1","C3-P3","F3-C3","FP1-F3","P8-O2","T8-P8",
                              "F8-T8","FP2-F8","P7-O1","T7-P7","F7-T7","FP1-F7"]
        self.labelsAR1020 = ["O2","O1","PZ","CZ","FZ","P8","P7","T8","T7","F8",
                             "F7","P4","P3","C4","C3","F4","F3","FP2","FP1"]
        """
        self.labelsBIP1010 = ["P10-O2","T10-P10","F10-T10","FP2-F10","P8-O2",
                               "T8-P8","F8-T8","FP2-F8","P4-O2","C4-P4","F4-C4",
                               "Fp2-F4","PZ-OZ","CZ-PZ","FZ-CZ","FPZ-FZ",
                               "P3-O1","C3-P3","F3-C3","FP1-F3","P7-O1","T7-P7",
                               "F7-T7","FP1-F7","P9-O1","T9-P9","F9-T9","FP1-F9"]
        self.labelsAR1010 = ["P10","T10","F10","O2","P8","T8","F8","FP2","P4",
                             "C4","F4","OZ","PZ","CZ","FZ","FPZ","P3","C3","F3",
                             "O1","P7","T7","F7","FP1","P9","T9","F9"]
        """
        self.labelsBIP1010 = ["F10-T10","FP2-F10","P8-O2","T8-P8","F8-T8",
                              "FP2-F8","P4-O2","C4-P4","F4-C4","Fp2-F4","CZ-PZ",
                              "FZ-CZ","P3-O1","C3-P3","F3-C3","FP1-F3","P7-O1",
                              "T7-P7","F7-T7","FP1-F7","F9-T9","FP1-F9"]
        self.labelsAR1010 = ["T10","F10","O2","P8","T8","F8","FP2","P4",
                             "C4","F4","PZ","CZ","FZ","P3","C3","F3",
                             "O1","P7","T7","F7","FP1","T9","F9"]
        self.otherLabels = ["T1","T2","A1","A2","FPZ","NZ","AF7","AF3","AF1",
                            "AFZ","AF2","AF4","AF8","F9","F5","F1","F2","F6",
                            "F10","FT9","FT7","FC5","FC3","FC1","FCZ","FC2",
                            "FC4","FC6","FT8","FT10","T9","C5","C1","C2","C6",
                            "T10","TP9","TP7","CP5","CP3","CP1","CPZ","CP2",
                            "CP4","CP6","TP8","TP10","P9","P5","P1","P2","P6",
                            "P10","PO7","PO3","POZ","PO4","PO8","OZ","LZ"]
        self.g = '#1f8c45'
        self.colorsBIP1020 = [self.g, self.g,'b','b','b','b','r','r','r','r','b',
                            'b','b','b','r','r','r','r','r']
        self.colorsAR1020 = ['b','r',self.g, self.g, self.g,'b','r','b','r','b',
                            'r','b','r','b','r','b','r','b','r','b']
        """
        self.colorsBIP1010 = ['b','b','b','b','b','b','b','b','b','b','b','b',
                                self.g,self.g,self.g,self.g,'r','r','r','r','r',
                                'r','r','r','r','r','r','r']
        self.colorsAR1010 = ['b','b','b','b','b','b','b','b','b','b','b',
                                self.g,self.g,self.g,self.g,self.g'r','r','r',
                                'r','r','r','r','r','r','r','r']
        """
        self.colorsBIP1010 = ['b','b','b','b','b','b','b','b','b','b',self.g,
                               self.g,'r','r','r','r','r','r','r','r','r','r']
        self.colorsAR1010 = ['b','b','b','b','b','b','b','b','b','b',self.g,
                              self.g,self.g,'r','r','r','r','r','r','r','r','r','r']
        self.otherColors = ['r','b','r','b',self.g, self.g,'r','r','r',self.g,
                                'b','b','b','r','r','r','b','b','b','r','r','r',
                                'r','r',self.g,'b','b','b','b','b','r','r','r',
                                'b','b','b','r','r','r','r','r',self.g,'b','b',
                                'b','b','b','r','r','r','b','b','b','r','r',
                                self.g,'b','b',self.g,self.g]
        self.pred_chn_data = []
        self.labelsFromTxtFile = {}
        self.use_loaded_txt_file = 0
        self.txtFile_fn = ""
        self.organize = 0

        self.labels_to_plot = []
        self.nchns_to_plot = 0
        self.mont_type = 5

    def write_data(self, ci2):
        """
        Writes data from ci2 into self
        """
        self.chns2labels = ci2.chns2labels
        self.labels2chns = ci2.labels2chns
        self.fs = ci2.fs
        self.max_time = ci2.max_time
        self.edf_fn = ci2.edf_fn

        self.labelsFromTxtFile = ci2.labelsFromTxtFile
        self.use_loaded_txt_file = ci2.use_loaded_txt_file
        self.txtFile_fn = ci2.txtFile_fn
        self.organize = ci2.organize

        self.total_nchns = ci2.total_nchns
        self.list_of_chns = ci2.list_of_chns
        self.convertedChnNames = ci2.convertedChnNames
        self.pred_chn_data = ci2.pred_chn_data

    def convertChnNames(self):
        """
        Converts given channel names to those in two montages.
        """

        for i in range(len(self.chns2labels)):
            self.convertedChnNames.append("")

        for k in range(len(self.labelsBIP1020)):
            ret = _check_label(self.labelsBIP1020[k],self.labels2chns)
            if ret != -1:
                self.convertedChnNames[ret] = self.labelsBIP1020[k]

        for k in range(len(self.labelsAR1020)):
            ret = _check_label(self.labelsAR1020[k],self.labels2chns)
            if ret != -1:
                if self.convertedChnNames[ret] == "":
                    self.convertedChnNames[ret] = self.labelsAR1020[k]

        for k in range(len(self.otherLabels)):
            ret = _check_label(self.otherLabels[k],self.labels2chns)
            if ret != -1:
                if self.convertedChnNames[ret] == "":
                    self.convertedChnNames[ret] = self.otherLabels[k]

        for k in range(len(self.convertedChnNames)):
            if self.convertedChnNames[k] == "":
                self.convertedChnNames[k] = self.chns2labels[k]

    def convertTxtChnNames(self,chn_txt):
        chn = chn_txt.upper()
        loc = chn.find("EEG ")
        if loc != -1:
            chn = chn[0:loc] + chn[loc+4:]
        loc = chn.find("-REF")
        if loc != -1:
            chn = chn[0:loc] + chn[loc+4:]
        if chn == "T7":
            chn = "T3"
        if chn == "P7":
            chn = "T5"
        if chn == "T8":
            chn = "T4"
        if chn == "P8":
            chn = "T6"
        return chn

    def canDoBIP_AR(self, bip_ar, mont1010_1020):
        """
        Whether or not the channels are present.
        inputs:
            bip_ar: 1 for average reference, 0 for bipolar
            mont1010_1020: 1 for 1010, 0 for 1020
        returns:
            1 for present, 0 for not present.
        """
        labels_to_check = []
        if bip_ar == 0 and mont1010_1020 == 0:
            labels_to_check = self.labelsBIP1020
        elif bip_ar == 1 and mont1010_1020 == 0:
            labels_to_check = self.labelsAR1020
        elif bip_ar == 0 and mont1010_1020 == 1:
            labels_to_check = self.labelsBIP1010
        else:
            labels_to_check = self.labelsAR1010

        ret = 1
        for i in range(len(labels_to_check)):
            if not (labels_to_check[i] in self.convertedChnNames):
                ret = 0
        return ret

    def canDoBIP_AR_idx(self, list_of_idxs, bip_ar, mont1010_1020):
        """
        Whether or not the channels for the montage are present.
        inputs:
            list_of_idxs: the list of indices
            bip_ar: 1 for average reference, 0 for bipolar
            mont1010_1020: 1 for 1010, 0 for 1020
        returns:
            1 for present, 0 for not present.
        """
        labels_to_check = []
        if bip_ar == 0 and mont1010_1020 == 0:
            labels_to_check = self.labelsBIP1020
        elif bip_ar == 1 and mont1010_1020 == 0:
            labels_to_check = self.labelsAR1020
        elif bip_ar == 0 and mont1010_1020 == 1:
            labels_to_check = self.labelsBIP1010
        else:
            labels_to_check = self.labelsAR1010

        for i in range(len(labels_to_check)):
            ret = 0
            for k in range(len(list_of_idxs)):
                if labels_to_check[i] == self.convertedChnNames[list_of_idxs[k]]:
                    ret = 1
            if ret == 0:
                return ret
        return ret

    def getChns(self, labels):
        """
        returns:
            A list of the indices of the channels given labels.
            The list is of length total_nchns and has 1 where it is
            a channel in the list and 0 otherwise.
        """
        ret = []
        for i in range(len(self.convertedChnNames)):
            if self.convertedChnNames[i] in labels:
                ret.append(1)
            else:
                ret.append(0)
        # Check for repeats
        for i in range(len(ret)):
            if ret[i]:
                for j in range(len(ret)):
                    if i != j and self.convertedChnNames[i] == self.convertedChnNames[j]:
                        if j > i:
                            ret[j] = 0
                        else:
                            ret[i] = 0
        return ret

    def prepareToPlot(self, idxs, parent, mont_type, plot_bip_from_ar = 0, txt_file_name = ""):
        """
        Prepares everything needed to plot the data.

        inputs:
            idxs - the list of the indices of the channels to be plotted,
                list is 1 where the chn is selected, otherwise 0
            parent - the main window, so that if needed self.predicted can
                be set to false
            mont_type - what montage is selected (0 = ar1020, 1 = bip1020, 2 = ar1010,
                                                    3 = bip1010, 4 = txtfile, 5 = none)
            plot_bip_from_ar - 1 if a bipolar montage should be generated
                from average reference data
            txt_file_name - name of text file if needed
        """
        print(self.edf_fn)
        f = pyedflib.EdfReader(self.edf_fn)

        # Things needed to plot - reset each time
        # see if channels are already loaded and ordered
        ret = 1
        self.nchns_to_plot = len(idxs)
        if plot_bip_from_ar:
            self.nchns_to_plot = len(idxs) - 1
        if (plot_bip_from_ar and len(self.labels_to_plot) != 0
                and len(self.labels_to_plot) == self.nchns_to_plot + 1):
            if self.canDoBIP_AR_idx(idxs,1,0) and len(idxs) == 18:
                for k in range(idxs):
                    if not self.labelsBIP1020[k] in self.labels_to_plot:
                        ret = 0
        elif plot_bip_from_ar:
            ret = 0

        if (not plot_bip_from_ar and len(self.labels_to_plot) != 0
                and len(self.labels_to_plot) == self.nchns_to_plot + 1):
            for k in range(len(idxs)):
                if not self.convertedChnNames[idxs[k]] in self.labels_to_plot:
                    ret = 0
        elif not plot_bip_from_ar:
            ret = 0
        if ret == 1 and not self.use_loaded_txt_file and self.organize: # already organized
            return
        if self.use_loaded_txt_file:
            self.organize = 0

        self.labels_to_plot = ["Notes"]
        self.colors = []
        self.data_to_plot = []
        self.nchns_to_plot = 0
        self.list_of_chns = []
        for k in range(len(idxs)):
            self.list_of_chns.append(idxs[k])

        ar = 0
        bip = 0
        ar1010 = 0
        bip1010 = 0
        self.nchns_to_plot = len(idxs)
        self.data_to_plot = np.zeros((self.nchns_to_plot, parent.edf_info_temp.nsamples[0])) # np.zeros((self.nchns_to_plot, data.shape[1]))
        if plot_bip_from_ar and self.canDoBIP_AR_idx(idxs,1,0):
            self.data_to_plot = np.zeros((self.nchns_to_plot - 1, parent.edf_info_temp.nsamples[0])) # np.zeros((self.nchns_to_plot - 1, data.shape[1]))
        c = 0

        if plot_bip_from_ar:
            if mont_type == 1:
                ar = self.canDoBIP_AR_idx(idxs,1,0) # must have all AR chns to convert to bipolar
                if ar:
                    bip_idx = np.zeros((18,2))
                    for k in range(18):
                        str0 = self.labelsBIP1020[k].split('-')[0]
                        str1 = self.labelsBIP1020[k].split('-')[1]
                        for i, strg in enumerate(self.convertedChnNames):
                            if strg == str0:
                                idx0 = i
                            if strg == str1:
                                idx1 = i
                        bip_idx[k,0] = idx0
                        bip_idx[k,1] = idx1
                    for k in range(18):
                        idx0 = bip_idx[k,0]
                        idx1 = bip_idx[k,1]
                        self.data_to_plot[k,:] = f.readSignal(int(idx0)) - f.readSignal(int(idx1)) #data[int(idx0),:] - data[int(idx1),:]
                        self.labels_to_plot.append(self.labelsBIP1020[k])
                        self.colors.append(self.colorsBIP1020[k])
                        c += 1

                    ar_idxs = self.getChns(self.labelsAR1020) # clear these from the list
                    k = 0
                    while k < len(idxs):
                        if ar_idxs[idxs[k]]:
                            idxs.pop(k)
                        else:
                            k += 1
                    self.nchns_to_plot = 18 + len(idxs)
                    if self.nchns_to_plot == 18:
                        self.mont_type = 1
            elif mont_type == 3:
                ar = self.canDoBIP_AR_idx(idxs,1,1) # must have all AR chns to convert to bipolar
                if ar:
                    bip_idx = np.zeros((len(self.labelsBIP1010),2))
                    for k in range(len(self.labelsBIP1010)):
                        str0 = self.labelsBIP1010[k].split('-')[0]
                        str1 = self.labelsBIP1010[k].split('-')[1]
                        for i, strg in enumerate(self.convertedChnNames):
                            if strg == str0:
                                idx0 = i
                            if strg == str1:
                                idx1 = i
                        bip_idx[k,0] = idx0
                        bip_idx[k,1] = idx1
                    for k in range(len(self.labelsBIP1010)):
                        idx0 = bip_idx[k,0]
                        idx1 = bip_idx[k,1]
                        self.data_to_plot[k,:] = f.readSignal(int(idx0)) - f.readSignal(int(idx1)) # data[int(idx0),:] - data[int(idx1),:]
                        self.labels_to_plot.append(self.labelsBIP1010[k])
                        self.colors.append(self.colorsBIP1010[k])
                        c += 1

                    ar_idxs = self.getChns(self.labelsAR1010) # clear these from the list
                    k = 0
                    while k < len(idxs):
                        if ar_idxs[idxs[k]]:
                            idxs.pop(k)
                        else:
                            k += 1
                    self.nchns_to_plot = len(self.labelsBIP1010) + len(idxs)
                    if self.nchns_to_plot == len(self.labelsBIP1010):
                        self.mont_type = 3

        # Check if any of the channels are average reference / bipolar
        #ar = self.canDoBIP_AR_idx(idxs,1,0)
        #bip = self.canDoBIP_AR_idx(idxs,0,0)
        ar = 0
        ar1010 = 0
        for i in range(len(idxs)):
            if self.convertedChnNames[idxs[i]] in self.labelsAR1020 and mont_type == 0:
                ar = 1
            elif self.convertedChnNames[idxs[i]] in self.labelsBIP1020 and mont_type == 1:
                bip = 1
            elif self.convertedChnNames[idxs[i]] in self.labelsAR1010 and mont_type == 2:
                ar1010 = 1
            elif self.convertedChnNames[idxs[i]] in self.labelsBIP1010 and mont_type == 3:
                bip1010 = 1
            elif mont_type == 4 or mont_type == 5:
                if self.convertedChnNames[idxs[i]] in self.labelsAR1020:
                    ar = 1
                elif self.convertedChnNames[idxs[i]] in self.labelsBIP1020:
                    bip = 1

        if self.use_loaded_txt_file and mont_type == 4:
            labels = self.labelsFromTxtFile[txt_file_name]
            colors = []
            for i in range(len(labels)):
                idx = -1
                if ar:
                    try:
                        idx = self.labelsAR1020.index(labels[i])
                        colors.append(self.colorsAR1020[idx])
                    except ValueError:
                        pass
                elif bip:
                    try:
                        idx = self.labelsBIP.index(labels[i])
                        colors.append(self.colorsBIP1020[idx])
                    except ValueError:
                        pass
                else:
                    try:
                        idx = self.otherLabels.index(labels[i])
                        colors.append(self.otherColors[idx])
                    except ValueError:
                        colors.append(self.g)
        elif bip:
            labels = self.labelsBIP1020
            colors = self.colorsBIP1020
        elif ar:
            labels = self.labelsAR1020
            colors = self.colorsAR1020
        elif bip1010:
            labels = self.labelsBIP1010
            colors = self.colorsBIP1010
        elif ar1010:
            labels = self.labelsAR1010
            colors = self.colorsAR1010


        # insert any data for the given montages
        if bip or ar or ar1010 or bip1010 or self.use_loaded_txt_file:
            for i in range(len(labels)):
                k = 0
                while k < len(idxs):
                    if self.convertedChnNames[idxs[k]] == labels[i]:
                        self.labels_to_plot.append(labels[i])
                        self.colors.append(colors[i])
                        self.data_to_plot[c,:] = f.readSignal(idxs[k]) # data[idxs[k],:]
                        c += 1
                        idxs.pop(k)
                        k = len(idxs)
                    else:
                        k += 1
        if len(idxs) > 0:
            # shift data back
            for k in range(c):
                self.data_to_plot[c - k + len(idxs) - 1,:] = self.data_to_plot[c - k - 1,:]
                self.data_to_plot[c - k - 1,:] = np.zeros((1,self.data_to_plot.shape[1]))
            c = len(idxs) - 1
            for k in range(len(idxs)):
                self.labels_to_plot.insert(1,self.convertedChnNames[idxs[k]])
                if self.convertedChnNames[idxs[k]] in self.otherLabels:
                    i = self.otherLabels.index(self.convertedChnNames[idxs[k]])
                    self.colors.insert(0,self.otherColors[i])
                else:
                    self.colors.insert(0,self.g)
                self.data_to_plot[c,:] = f.readSignal(idxs[k]) # data[idxs[k],:]
                c -= 1
