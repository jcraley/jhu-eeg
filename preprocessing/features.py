import nolds
import numpy as np
import scipy.signal
import torch


def sampen(windowed_buffers, **kwargs):
    """Find the sample entropy of the buffers"""
    T, C, _ = windowed_buffers.size()
    sampen_feature = torch.zeros((T, C, 1), dtype=torch.float32)
    for tt in range(T):
        for cc in range(C):
            sampen_feature[tt, cc, 0] = nolds.sampen(
                windowed_buffers[tt, cc, :])
    return sampen_feature


def lle(windowed_buffers, **kwargs):
    """Find the largest Lyapunov exponent of the buffers"""
    T, C, _ = windowed_buffers.size()
    lyap_e_feature = torch.zeros((T, C, 1), dtype=torch.float32)
    for tt in range(T):
        for cc in range(C):
            lyap = torch.tensor(nolds.lyap_e(windowed_buffers[tt, cc, :]))
            lyap_e_feature[tt, cc, 0] = lyap[0]
    return lyap_e_feature


def power(windowed_buffers, fs=200):
    """Take the power in a given channel"""
    T, C, _ = windowed_buffers.size()
    power_feature = torch.zeros((T, C, 1), dtype=torch.float32)
    power_feature[:, :, 0] = torch.mean(windowed_buffers.pow(2), dim=2)
    return power_feature


def fft(windowed_buffers, fs=200, fs_max=30):
    """Take the fft"""
    t = windowed_buffers.size(2)
    window = scipy.signal.tukey(t)
    freq = np.fft.fftfreq(t, d=1/fs)
    idx = np.where((0 <= freq) * (freq <= fs_max))
    f = np.fft.fft(windowed_buffers.numpy()
                   * window[np.newaxis, np.newaxis, :])[:, :, idx]
    return torch.tensor(np.absolute(f), dtype=torch.float32)


def linelength(windowed_buffers, fs=200):
    """Compute the linelength"""
    diffs = torch.abs(windowed_buffers[:, :, 1:] - windowed_buffers[:, :, :-1])
    return torch.mean(diffs, dim=2).unsqueeze(2)


def bandpass(windowed_buffers, M=10, high=30, fs=200, order=4):
    """Pass the signal through a bandpass"""

    # Create the filters
    nyq = 0.5 * fs
    critical_freqs = high / M * np.arange(M + 1) / nyq
    critical_freqs[0] = 0.5 / nyq
    T, C, _ = windowed_buffers.size()
    data = torch.zeros((T, C, M), dtype=torch.float32)

    for mm in range(M):
        low, high = [critical_freqs[mm], critical_freqs[mm + 1]]
        b, a = scipy.signal.butter(order, [low, high], btype='bandpass')
        print('{} Hz to {} Hz band'.format(low, high))
        # Apply the filters
        for tt in range(T):
            filtered = scipy.signal.lfilter(
                b, a, windowed_buffers[tt, :, :].numpy())
            for cc in range(C):
                data[tt, cc, mm] = np.mean(np.power(filtered[cc, :], 2))
    return data
