import numpy as np

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

        self.total_nchns = 0
        self.list_of_chns = []
        self.convertedChnNames = []
        self.labelsBIP = ["CZ-PZ","FZ-CZ","P4-O2","C4-P4","F4-C4","FP2-F4",
                       "P3-O1","C3-P3","F3-C3","FP1-F3","P8-O2","T8-P8",
                       "F8-T8","FP2-F8","P7-O1","T7-P7","F7-T7","FP1-F7"]
        self.labelsAR = ["O2","O1","PZ","CZ","FZ","P8","P7","T8","T7","F8",
                        "F7","P4","P3","C4","C3","F4","F3","FP2","FP1"]
        self.colorsBIP = ['g','g','b','b','b','b','r','r','r','r','b','b','b',
                            'b','r','r','r','r','r']
        self.colorsAR = ['b','r','g','g','g','b','r','b','r','b','r','b','r','b',
                            'r','b','r','b','r','b']
        self.pred_chn_data = []

    def write_data(self, ci2):
        """
        Writes data from ci2 into self
        """
        self.chns2labels = ci2.chns2labels
        self.labels2chns = ci2.labels2chns
        self.fs = ci2.fs
        self.max_time = ci2.max_time

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

        for k in range(len(self.labelsBIP)):
            ret = _check_label(self.labelsBIP[k],self.labels2chns)
            if ret != -1:
                self.convertedChnNames[ret] = self.labelsBIP[k]

        for k in range(len(self.labelsAR)):
            ret = _check_label(self.labelsAR[k],self.labels2chns)
            if ret != -1:
                if self.convertedChnNames[ret] == "":
                    self.convertedChnNames[ret] = self.labelsAR[k]

        for k in range(len(self.convertedChnNames)):
            if self.convertedChnNames[k] == "":
                self.convertedChnNames[k] = self.chns2labels[k]

    def canDoBIP(self):
        """
        Whether or not the channels for bipolar are present.
        returns:
            1 for present, 0 for not present.
        """
        ret = 1
        for i in range(len(self.labelsBIP)):
            if not (self.labelsBIP[i] in self.convertedChnNames):
                ret = 0
        return ret

    def canDoAR(self):
        """
        Whether or not the channels for average reference are present.
        returns:
            1 for present, 0 for not present.
        """
        ret = 1
        for i in range(len(self.labelsAR)):
            if not (self.labelsAR[i] in self.convertedChnNames):
                ret = 0
        return ret

    def canDoAR_idx(self, list_of_idxs):
        """
        Whether or not the channels for average reference are present.
        returns:
            1 for present, 0 for not present.
        """
        for i in range(len(self.labelsAR)):
            ret = 0
            for k in range(len(list_of_idxs)):
                if self.labelsAR[i] == self.convertedChnNames[list_of_idxs[k]]:
                    ret = 1
            if ret == 0:
                return ret
        return ret

    def getARchns(self):
        """
        returns:
            A list of the indices of the average reference channels.
            The list is of length total_nchns and has 1 where it is
            a AR channel and 0 otherwise.
        """
        ret = []
        for i in range(len(self.convertedChnNames)):
            if self.convertedChnNames[i] in self.labelsAR:
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

    def getBIPchns(self):
        """
        returns:
            A list of the indices of the bipolar channels.
            The list is of length total_nchns and has 1 where it is
            a bipolar channel and 0 otherwise.
        """
        ret = []
        for i in range(len(self.convertedChnNames)):
            if self.convertedChnNames[i] in self.labelsBIP:
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

    def prepareToPlot(self, idxs, data, parent, plot_bip_from_ar = 0):
        """
        Prepares everything needed to plot the data.

        inputs:
            idxs - the list of the indices of the channels to be plotted,
                list is 1 where the chn is selected, otherwise 0
            data - the data for each channel
            parent - the main window, so that if needed self.predicted can
                be set to false
            plot_bip_from_ar - 1 if a bipolar montage should be generated
                from average reference data
        """
        # Things needed to plot - reset each time
        self.labels_to_plot = ["Notes"]
        self.colors = []
        self.data_to_plot = []
        self.nchns_to_plot = 0
        self.list_of_chns = []
        for k in range(len(idxs)):
            self.list_of_chns.append(idxs[k])

        ar = 0
        bip = 0
        self.nchns_to_plot = len(idxs)
        self.data_to_plot = np.zeros((self.nchns_to_plot, data.shape[1]))
        if plot_bip_from_ar:
            self.data_to_plot = np.zeros((self.nchns_to_plot - 1, data.shape[1]))
        c = 0

        if plot_bip_from_ar:
            ar = self.canDoAR_idx(idxs) # must have all AR chns to convert to bipolar
            if ar:
                bip_idx = np.zeros((18,2))
                for k in range(18):
                    str0 = self.labelsBIP[k].split('-')[0]
                    str1 = self.labelsBIP[k].split('-')[1]
                    idx0 = np.argmax(np.char.find(self.convertedChnNames,str0))
                    idx1 = np.argmax(np.char.find(self.convertedChnNames,str1))
                    bip_idx[k,0] = idx0
                    bip_idx[k,1] = idx1
                for k in range(18):
                    idx0 = bip_idx[k,0]
                    idx1 = bip_idx[k,1]
                    self.data_to_plot[k,:] = data[int(idx0),:] - data[int(idx1),:]
                    self.labels_to_plot.append(self.labelsBIP[k])
                    self.colors.append(self.colorsBIP[k])
                    c += 1

                ar_idxs = self.getARchns() # clear these from the list
                k = 0
                while k < len(idxs):
                    if ar_idxs[idxs[k]]:
                        idxs.pop(k)
                    else:
                        k += 1
                self.nchns_to_plot = 18 + len(idxs)

        # Check for average reference / bipolar
        ar = 0
        for i in range(len(idxs)):
            if self.convertedChnNames[idxs[i]] in self.labelsAR:
                ar = 1
            elif self.convertedChnNames[idxs[i]] in self.labelsBIP:
                bip = 1
        if bip:
            labels = self.labelsBIP
            colors = self.colorsBIP
        elif ar:
            labels = self.labelsAR
            colors = self.colorsAR
        # insert any data for the given montages
        if bip or ar:
            for i in range(len(labels)):
                k = 0
                while k < len(idxs):
                    if self.convertedChnNames[idxs[k]] == labels[i]:
                        self.labels_to_plot.append(labels[i])
                        self.colors.append(colors[i])
                        self.data_to_plot[c,:] = data[idxs[k],:]
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
                self.colors.insert(0,'g')
                self.data_to_plot[c,:] = data[idxs[k],:]
                c -= 1

        # If nchns != len(predictions) do not plot predictions
        set_predicted = parent.pi.updatePredicted(self.data_to_plot, parent.max_time, parent.predicted)
        parent.predicted = set_predicted
