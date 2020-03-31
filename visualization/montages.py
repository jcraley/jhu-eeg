import numpy as np
import pyedflib
from preprocessing.eeg_info import EegInfo


def _check_label(label, label_list):
    """
    Checks if a label is in the label list
    """
    # If the label is not present, try splitting it
    if label not in label_list:
        label_CAPS = {k.upper(): v for k, v in label_list.items()}
        if label in label_CAPS:
            return label_CAPS[label]

    labels_noEEG = {}
    labels_noRef = {}
    if label not in label_list:
        for k,v in label_list.items():
            loc = k.find("EEG ")
            if loc != -1:
                k2 = k[loc+4:]
                labels_noEEG[k2] = label_list[k]
        if label in labels_noEEG:
            return labels_noEEG[label]

        for k,v in labels_noEEG.items():
            loc = k.find("-REF")
            if loc != -1:
                k2 = k[0:loc]
                labels_noRef[k2] = labels_noEEG[k]

        if label in labels_noRef:
            return labels_noRef[label]

        if label not in labels_noRef:
            label_CAPS_noRef = {k.upper(): v for k, v in labels_noRef.items()}
            if label in label_CAPS_noRef:
                return label_CAPS_noRef[label]

    if label not in label_list:
        label = label[4:].split('-')[0].upper()

    # return label
    if label in label_list:
        return label_list[label]
    else:
        return -1

def _check_montage(label):
    """
    Checks if we are in ref montage or normal montage
    returns:
        0 for normal, 1 for ref
    """
    if label[len(label)-3:].upper() == "REF":
        return 1
    return 0


class EdfMontage():
    """ A class for loading info and buffers from EDF files """

    def __init__(self, eeg_info):
        self.montage_data = []
        self.labels = ["CZ-PZ","FZ-CZ","P4-O2","C4-P4","F4-C4","FP2-F4",
                       "P3-O1","C3-P3","F3-C3","FP1-F3","P8-O2","T8-P8",
                       "F8-T8","FP2-F8","P7-O1","T7-P7","F7-T7","FP1-F7"]
        self.labelsAR = ["O2","O1","PZ","CZ","FZ","P8","P7","T8","T7","F8",
                        "F7","P4","P3","C4","C3","F4","F3","FP2","FP1"]
        self.eeg_info = eeg_info
        self.nchns = self.eeg_info.nchns


    def reorder_data(self, data):
        """
        Reorder data so that it is in the proper montage format

        inputs:
            data - buffer of data loaded from .edf file

        returns:
            montage_data - a EdfMontage object with the correctly ordered channels
        """
        mont = _check_montage(list(self.eeg_info.labels2chns.keys())[0])
        if mont == 1:
            self.nchns = 19
            self.montage_data = self.ar(data)
        else:
            self.nchns = 18
            self.montage_data = np.zeros((self.nchns, data.shape[1]))
            for chn in range(len(self.labels)):
                edf_chn = _check_label(self.labels[chn],self.eeg_info.labels2chns)
                if edf_chn != -1:
                    self.montage_data[chn,:] = data[edf_chn,:]
        return self.montage_data

    def ar(self, data):
        """
        Make average reference montage

        inputs:
            data - buffer of signals
        outputs:
            the montage
        """

        montage_ar= np.zeros((self.nchns, data.shape[1]))
        for chn in range(len(self.labelsAR)):
            edf_chn = _check_label(self.labelsAR[chn],self.eeg_info.labels2chns)
            if edf_chn != -1:
                montage_ar[chn,:] = data[edf_chn,:]
        return montage_ar
