import numpy as np
import preprocessing.dsp as dsp
import torch
import scipy.signal
from copy import deepcopy
from models.basemodel import BaseModel
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QProgressDialog
import mne

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

    if window_size > 1:
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
    Progress bar is created if the process is estimated to take > 4s

    inputs:
        data - the data to filter
        lp - lowpass frequency
        hp - highpass frequency
        standardize - whether or not to standardize the data
    returns:
        filtered data
    """
    lp = fi.lp
    hp = fi.hp
    bp1 = fi.bp1
    bp2 = fi.bp2
    if fi.do_lp == 0 or lp < 0 or lp > fs / 2:
        lp = 0
    if fi.do_hp == 0 or hp < 0 or hp > fs / 2:
        hp = 0
    if fi.do_bp == 0 or bp1 < 0 or bp1 > fs / 2 or bp2 < 0 or bp1 > fs / 2 or bp2 - bp1 <= 0:
        bp1 = 0
        bp2 = 0


    nchns = len(data)
    filt_bufs = deepcopy(data)
    progress = QProgressDialog("Filtering...", "Cancel", 0, nchns * 4)
    progress.setWindowModality(Qt.WindowModal)

    i = 0
    for chn in range(nchns):
        # Notch
        if fi.notch > 0 and fi.notch < fs / 2:
            filt_bufs[chn] = applyNotch(filt_bufs[chn], fs,fi.notch)
        i += 1
        progress.setValue(i)
        if progress.wasCanceled():
            fi.filter_canceled = 1
            break
        # LPF
        if lp > 0:
            filt_bufs[chn] = dsp.applyLowPass(filt_bufs[chn], fs, lp)
        i += 1
        progress.setValue(i)
        if progress.wasCanceled():
            fi.filter_canceled = 1
            break
        # HPF
        if hp > 0:
            filt_bufs[chn] = dsp.applyHighPass(filt_bufs[chn], fs, hp)
        i += 1
        progress.setValue(i)
        if progress.wasCanceled():
            fi.filter_canceled = 1
            break
        # BPF
        if bp1 > 0:
            filt_bufs[chn] = applyBandPass(filt_bufs[chn], fs, [bp1, bp2])
        i += 1
        progress.setValue(i)
        if progress.wasCanceled():
            fi.filter_canceled = 1
            break
        # filt_bufs[chn] = dsp.scale(filt_bufs[chn])

    return filt_bufs

def convertFromCount(count):
    """
    Converts time from count (int in seconds) to the time format
    hh:mm:ss.

    input:
        count - the value of count
    returns:
        hrs, min, sec - the time
    """
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
    return hrs, min, sec

def getTime(count):
    """
    Creates a string for the time in seconds.

    inputs:
        count - the current value of the plot in seconds
    returns:
        t_str - a string of the seconds in the form hrs:min:sec
    """
    t_str = ""
    hrs, min, sec = convertFromCount(count)
    """t = count
    hrs = 0
    min = 0
    sec = 0
    if int(t / 3600) > 0:
        hrs = int(t / 3600)
        t = t % 3600
    if int(t / 60) > 0:
        min = int(t / 60)
        t = t % 60
    sec = t"""
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

def loadSignals(data, fsArray):
    """
    Loads signals into a buffer based on the array of freqeuncies.
    Signals with frequencies != max_fs will be interpolated.

    inputs:
        data - the raw EEG data
        fsArray - array of fs for each signal
        nsamples - array of the number of samples in each signal
    returns:
        fs - the frequency of the signals
        buf - the loaded buffer of signals
    """
    nchns = len(data)
    if nchns == 1:
        return fsArray, np.array(data)
    same_fs = 1
    try:
        if len(fsArray) > 1:
            fs = np.max(fsArray)
            fs_idx = np.argmax(fsArray)
            same_fs = 0
        elif len(fsArray) == 1:
            fsArray = fsArray[0]
    except:
        fs = fsArray
        same_fs = 1

    buf = np.array(data, dtype=object)
    if buf.ndim == 1:
        if same_fs:
            data_temp = np.zeros((buf.shape[0],buf[0].shape[0]))
            for i in range(buf.shape[0]):
                data_temp[i,:] = buf[i]
            buf = np.array(data_temp)
        else:
            nsamples = data[fs_idx].shape[0]
            data_temp = np.zeros((buf.shape[0],nsamples))
            for i in range(buf.shape[0]):
                fs_i = fsArray[i]
                if fs == fs_i:
                    data_temp[i,:] = data[i]
                else:
                    xp = np.arange(0,nsamples,int(fs / fs_i))
                    yp = data[i]
                    x = np.arange(0,nsamples,1)
                    data_temp[i,:] = np.interp(x,xp,yp)
            buf = np.array(data_temp)

    return fs, buf

def applyNotch(x, fs, fc=60, Q=20.0):
    """Apply a notch filter at fc Hz
    """
    w60Hz = fc / (fs / 2)
    b, a = scipy.signal.iirnotch(w60Hz, Q)
    return scipy.signal.filtfilt(b, a, x, method='gust')

def applyBandPass(x, fs, fc=[1.6,30], N=4):
    """Apply a low-pass filter to the signal
    """
    wc = fc / (fs / 2)
    b, a = scipy.signal.butter(N, wc, btype='bandpass')
    return scipy.signal.filtfilt(b, a, x, method='gust')


def topoplot(scores, label_list, title=None, fn='',
             plot_hemisphere=False, plot_lobe=False, zone=None,
             lobe_correct=None, lat_correct=None):
    # Create the layout
    layout = mne.channels.read_layout('EEG1005')
    # positions = []
    pos2d = []
    layout_names = [name.upper() for name in layout.names]
    for ch in label_list:
        if '-' in ch:
            anode, cathode = ch.split('-')
            anode_idx = layout_names.index(anode)
            cathode_idx = layout_names.index(cathode)
            anode_pos = layout.pos[anode_idx, 0:2]
            cathode_pos = layout.pos[cathode_idx, 0:2]
            pos2d.append([(a + c) / 2 for a, c in zip(anode_pos, cathode_pos)])
        else:
            idx = layout_names.index(ch)
            # positions.append(layout.pos[idx, :])
            pos2d.append(layout.pos[idx, 0:2])
    # positions = np.asarray(positions)
    pos2d = np.asarray(pos2d)
    # Scale locations from [-1, 1]
    pos2d = 2 * (pos2d - 0.5)

    # fig = plt.figure()
    ax = plt.gca()
    im, cn = mne.viz.plot_topomap(scores, pos2d, sphere=1,
                                  axes=ax, vmin=0, vmax=1, show=False,
                                  outlines='head')
    if plot_hemisphere:
        ax.plot([0, 0], [-1, 1], linestyle='--', color='k', linewidth=6)
    if plot_lobe:
        ax.plot([-0.96, 0.96], [0.2, 0.2], linestyle='--', color='k',
                linewidth=6)

    # Thicken outline
    circle = plt.Circle((0, 0), .98, edgecolor='k', facecolor='none',
                        linewidth=6)
    ax.add_artist(circle)
    ax.plot([-0.167, 0, 0.167], [.988, 1.15, .988], color='k', linewidth=6)
    ax.plot([.98, 1.02, 1.05, 1.08, 1.09, 1.06, 1.02, 0.98],
            [.12, .16, .15, .11, -.18, -.26, -.27, -.24], color='k', linewidth=6)
    ax.plot([-.98, -1.02, -1.05, -1.08, -1.09, -1.06, -1.02, -0.98],
            [.12, .16, .15, .11, -.18, -.26, -.27, -.24], color='k', linewidth=6)

    if lobe_correct and lat_correct:
        color = 'green'
    elif lobe_correct or lat_correct:
        color = 'orange'
    elif not lobe_correct and not lat_correct:
        color = 'red'
    else:
        color = 'blue'

    if zone == 1:
        circle = plt.Circle((-0.95, 0.8), 0.16, color=color)
        ax.add_artist(circle)
    elif zone == 2:
        circle = plt.Circle((0.95, 0.8), 0.16, color=color)
        ax.add_artist(circle)
    elif zone == 3:
        circle = plt.Circle((-0.95, -0.8), 0.16, color=color)
        ax.add_artist(circle)
    elif zone == 4:
        circle = plt.Circle((0.95, -0.8), 0.16, color=color)
        ax.add_artist(circle)

    if title:
        ax.set_title(title, fontsize=16)
    if fn:
        plt.savefig(fn, bbox_inches='tight')
    plt.close()
