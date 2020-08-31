import nolds
import numpy as np
import scipy.signal
import torch
import pywt
from scipy.signal import resample_poly


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
    return torch.tensor(np.absolute(f[:, :, 0, :]), dtype=torch.float32)


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


def kaleem_features(windowed_buffers, fs=200):
    """
    Compute features from "Patient-specific seizure detection in long-term
    EEG using wavelet decomposition"

    Args:
        windowed_buffers (torch tensor): (T, C, L)
        fs (sample rate): Defaults to 200.
    """

    assert (fs == 200 or fs == 256), "Sample frequency must be 200 or 256"

    # Initialize feature tensor
    T, C, L = windowed_buffers.shape
    kaleem_feats = torch.zeros((T, C, 12))

    # Loop and extract features
    for tt in range(T):
        for cc in range(C):
            signal = windowed_buffers[tt, cc, :]
            if fs == 200:  # resample to 256
                signal = resample_poly(signal, up=32, down=25, axis=0)

            # Take the wavelet decomposition
            cA5, cD5, cD4, cD3, cD2, cD1 = pywt.wavedec(signal, 'db6', level=5)

            # Compute the energy features
            kaleem_feats[tt, cc, 0] = np.sum(np.power(cA5, 2)) / len(cA5)
            kaleem_feats[tt, cc, 1] = np.sum(np.power(cD1, 2)) / len(cD1)
            kaleem_feats[tt, cc, 2] = np.sum(np.power(cD2, 2)) / len(cD2)
            kaleem_feats[tt, cc, 3] = np.sum(np.power(cD3, 2)) / len(cD3)

            # Compute the spectrum of each set of coefficients
            cA5_fft = np.abs(np.fft.fft(cA5))
            cD1_fft = np.abs(np.fft.fft(cD1))
            cD2_fft = np.abs(np.fft.fft(cD2))
            cD3_fft = np.abs(np.fft.fft(cD3))

            # Compute sparsity features
            kaleem_feats[tt, cc, 4] = sparsity(cA5_fft)
            kaleem_feats[tt, cc, 5] = sparsity(cD1_fft)
            kaleem_feats[tt, cc, 6] = sparsity(cD2_fft)
            kaleem_feats[tt, cc, 7] = sparsity(cD3_fft)

            # Compute derivative features
            kaleem_feats[tt, cc, 8] = derivative2(cA5_fft)
            kaleem_feats[tt, cc, 9] = derivative2(cD1_fft)
            kaleem_feats[tt, cc, 10] = derivative2(cD2_fft)
            kaleem_feats[tt, cc, 11] = derivative2(cD3_fft)

    return kaleem_feats


def sparsity(signal):
    N = signal.size
    s = np.sum(signal)
    s2 = np.sum(np.power(signal, 2))
    return (np.sqrt(N) - s / np.sqrt(s2)) / (np.sqrt(N) - 1)


def derivative2(signal):
    return np.sum(np.power(np.diff(signal), 2)) / len(signal)
