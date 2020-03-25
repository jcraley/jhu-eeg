import os
import sys
import torch
import numpy as np
import utils.read_files as read
import utils.pathmanager as pm
import utils.testconfiguration as tc
import scipy.signal
import time


def power(windowed_buffers, fs=200):
    """Take the power in a given channel"""
    T, C, _ = windowed_buffers.size()
    power_feature = torch.zeros((T, C, 1), dtype=torch.float32)
    power_feature[:, :, 0] = torch.mean(windowed_buffers.pow(2), dim=2)
    return power_feature


def fft(windowed_buffers, fs=200):
    """Take the fft"""
    t = windowed_buffers.size(2)
    window = scipy.signal.tukey(t)
    t = windowed_buffers.size(2) // 2
    f = np.fft.fft(windowed_buffers.numpy()
                   * window[np.newaxis, np.newaxis, :])[:, :, :t]
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
        low, high =[critical_freqs[mm], critical_freqs[mm + 1]]
        b, a = scipy.signal.butter(order, [low, high], btype='bandpass')
        print('{} Hz to {} Hz band'.format(low, high))
        # Apply the filters
        for tt in range(T):
            filtered = scipy.signal.lfilter(
                    b, a, windowed_buffers[tt, :, :].numpy())
            for cc in range(C):
                data[tt, cc, mm] = np.mean(np.power(filtered[cc, :], 2))
    return data


def main():
    """Set up the experiment, initialize folders, and write config"""
    # Load the configuration files
    argv = sys.argv[1:]
    params = tc.TestConfiguration('default.ini', argv)
    paths = pm.PathManager(params)
    paths.initialize_folder('data')

    features = ['power']
    for feat_name in features:
        print("Extracting {}".format(feat_name))
        paths.add_feature_folder(feat_name)

        # Load the manifest files
        manifest_files = read.read_manifest(params['train manifest'])
        fs = int(manifest_files[0]['fs'])

        # Loop over files and create windowed versions
        for file in manifest_files:
            fn = file['fn'].split('/')[-1].split('.')[0] + '.pt'
            fn_buf = os.path.join(paths['data'], fn)
            windowed_buffers = torch.load(fn_buf)
            feat = eval(feat_name + '(windowed_buffers, fs=fs)')
            feat_fn = os.path.join(paths[feat_name], fn)
            torch.save(feat, feat_fn)


if __name__ == '__main__':
    main()
