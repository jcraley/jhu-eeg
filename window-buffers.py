import os
import sys
import torch
import numpy as np
import utils.read_files as read
import utils.pathmanager as pm
import utils.testconfiguration as tc


def apply_window(buffers, fs, window_length, overlap):
    """Create windows from a buffered recording
    """
    window_length_samples = int(np.floor(fs * window_length))
    overlap_samples = int(np.floor(fs * overlap))
    advance_samples = int(window_length_samples - overlap_samples)
    start_sample = 0
    end_sample = window_length_samples
    L = buffers.size(0)
    data = []
    while end_sample <= L:
        data.append(buffers[start_sample:end_sample,
                            :].transpose(0, 1).unsqueeze(0))
        start_sample += advance_samples
        end_sample += advance_samples
    data = torch.cat(data, dim=0)
    return data


def main():
    """Set up the experiment, initialize folders, and write config"""
    # Load the configuration files
    argv = sys.argv[1:]
    params = tc.TestConfiguration('default.ini', argv)
    paths = pm.PathManager(params)
    paths.initialize_folder('data')

    # Load the manifest files
    manifest_files = read.read_manifest(params['train manifest'])
    fs = int(manifest_files[0]['fs'])

    # Loop over files and create windowed versions
    for file in manifest_files:
        fn = file['fn'].split('/')[-1].split('.')[0] + '.pt'
        fn_buf = os.path.join(paths['buffers'], fn)
        buffers = torch.load(fn_buf)
        data = apply_window(buffers, fs=fs,
                            window_length=params['window length'],
                            overlap=params['overlap'])
        fn_win_buf = os.path.join(paths['data'], fn)
        torch.save(data, fn_win_buf)


if __name__ == '__main__':
    main()
