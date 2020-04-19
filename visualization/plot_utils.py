import numpy as np
import preprocessing.dsp as dsp
import torch
import scipy.signal
from copy import deepcopy
from models.basemodel import BaseModel

def checkAnnotations(t_start,window_size,edf_info):
    """
    Checks to see if there are any anotations in the range t_start to t_end sec

    inputs:
        t_start - start time for a graph
        window_size - the number of seconds being plotted at a time
        edf_info - edf_info object containing annotations

    returns:
        ret - an array that is filled with any annotations[t_start:t_end]
        idx_w_ann - an array of size window_size that tells whether or not there is
                    an adjacent annotation
    """

    fs = edf_info.fs
    ann = edf_info.annotations
    t_end = t_start + window_size - 1
    t_startPts = t_start * fs
    t_endPts = t_end * fs
    i = 0
    ret = []
    idx_w_ann = np.zeros((window_size))

    if len(ann[0]) != 0:
        ann_i = int(float(ann[0,i]))
        while ann_i <= t_end and i < ann.shape[1]:
            if int(float(ann[0,i])) >= t_start and int(float(ann[0,i])) <= t_end:
                ret.append(ann[:,i])
                idx_w_ann[int(float(ann[0,i]))-t_start] = 1
                i += 1
            elif int(float(ann[0,i])) <= t_end:
                i += 1
            else:
                i = ann.shape[1]
    else:
        return ret, idx_w_ann

    if not(idx_w_ann[0] == 1 and idx_w_ann[1] == 1):
        idx_w_ann[0] = 0
    i = 1
    while i < window_size - 1:
        if not((idx_w_ann[i-1] == 1 or idx_w_ann[i+1] == 1) and idx_w_ann[i] == 1):
            idx_w_ann[i] = 0
        i += 1
    if idx_w_ann[window_size - 2] == 0 and idx_w_ann[window_size - 1] == 1:
        idx_w_ann[window_size - 1] = 0

    return ret, idx_w_ann

def filterData(data, fs, fi):
    """
    Calls dsp.prefilter to filter the data

    inputs:
        data - the data to filter
        lp - lowpass frequency
        hp - highpass frequency
        standardize - whether or not to standardize the data
    returns:
        filtered data
    """
    if fi.do_lp == 0:
        lp = 0
    else:
        lp = fi.lp
    if fi.do_hp == 0:
        hp = 0
    else:
        hp = fi.hp

    nchns = len(data)
    filt_bufs = deepcopy(data)
    for chn in range(nchns):
        # Notch
        if fi.notch > 0:
            filt_bufs[chn] = applyNotch(filt_bufs[chn], fs,fi.notch)
        # LPF
        if lp > 0:
            filt_bufs[chn] = dsp.applyLowPass(filt_bufs[chn], fs, lp)
        # HPF
        if hp > 0:
            filt_bufs[chn] = dsp.applyHighPass(filt_bufs[chn], fs, hp)
        # filt_bufs[chn] = dsp.scale(filt_bufs[chn])

    return filt_bufs

def predict(data,model,parent):
    """
    Loads model, passes data through the model to get binary seizure predictions

    inputs:
        data - the pytorch tensor, fully preprocessed
        model_fn - filename of the model to load

    returns:
        preds - a numpy array of binary predictions
    """
    try:
        preds = model.predict(data)
        preds = np.array(preds)
    except:
        parent.throwAlert("An error occured when trying to call the predict() " +
                    "function using your model. Please check your model and data.")
        return

    parent.predicted = 1
    return preds

def getTime(count):
    t_str = ""
    t = count
    hrs = 0
    min = 0
    sec = 0
    if int(t / 3600) > 0:
        hrs = int(t / 3600)
        t = t % 3600
    if int(t / 60) > 0:
        min = int(t / 60)
        t = t % 60
    sec = t
    if sec >= 10:
        str_sec = str(sec)
    else:
        str_sec = "0" + str(sec)
    if min >= 10:
        str_min = str(min)
    else:
        str_min = "0" + str(min)
    str_hr = str(hrs)
    t_str = str_hr + ":" + str_min + ":" + str_sec
    return t_str


def applyNotch(x, fs, fc=60, Q=20.0):
    """Apply a notch filter at 60 Hz
    """
    w60Hz = fc / (fs / 2)
    b, a = scipy.signal.iirnotch(w60Hz, Q)
    return scipy.signal.filtfilt(b, a, x, method='gust')
