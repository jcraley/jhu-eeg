import numpy as np
import preprocessing.dsp as dsp
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
    if ann != []:
        ann_i = int(float(ann[0,i]))
        while ann_i <= t_start and i < ann.shape[1]:
            if int(float(ann[0,i])) >= t_start and int(float(ann[0,i])) <= t_end:
                ret.append(ann[:,i])
                idx_w_ann[int(float(ann[0,i]))-t_start] = 1
                i += 1
            elif int(float(ann[0,i])) <= t_end:
                i += 1
            else:
                i = ann.shape[1]
    else:
        return [], idx_w_ann

    if not(idx_w_ann[0] == 1 and idx_w_ann[1] == 1):
        idx_w_ann[0] = 0
    i = 1
    while i < window_size - 1:
        if not((idx_w_ann[i-1] == 1 or idx_w_ann[i+1] == 1) and idx_w_ann[i] == 1):
            idx_w_ann[i] = 0
        i += 1
    if idx_w_ann[window_size - 2] == 0 and idx_w_ann[window_size - 1] == 1:
        idx_w_ann[window_size - 1] = 0

    return np.array(ret).T, idx_w_ann

def filterData(data, fs, lp=30, hp=1.6, standardize=True):
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
    return dsp.prefilter(data, fs, notch=True, lpf_fc=lp,
                            hpf_fc=hp, standardize=standardize)

def predict(data,model_fn):
    """
    Loads model, passes data through the model to get binary seizure predictions

    inputs:
        data - the pytorch tensor, fully preprocessed
        model_fn - filename of the model to load

    returns:
        preds - a numpy array of binary predictions
    """
    # TODO - implement this function
    model = torch.load(model_fn)
    preds = model.predict(data)

    return None
