import os
import sys
import json
import torch
import numpy as np
import utils.read_files as read
import utils.testconfiguration as tc
import utils.pathmanager as pm


def make_labels(nsamples, fs, starts, ends, window_length, overlap):
    """Make a set of labels for each file"""
    # Get window length constants
    window_length_samples = int(np.floor(fs * window_length))
    overlap_samples = int(np.floor(fs * overlap))
    advance_samples = int(window_length_samples - overlap_samples)
    start_sample = 0
    end_sample = window_length_samples

    # Loop and create labels
    labels = []
    while end_sample <= nsamples:
        labels.append(0)
        for start, end in zip(starts, ends):
            if (start_sample / float(fs) >= start
                    and end_sample / float(fs) <= end):
                labels[-1] = 1
        start_sample += advance_samples
        end_sample += advance_samples
    return labels


def main():
    """Set up the experiment, initialize folders, and write config"""
    # Load the configuration files
    argv = sys.argv[1:]
    params = tc.TestConfiguration('default.ini', argv)
    paths = pm.PathManager(params)
    paths.initialize_folder('labels')

    # Loop over edf files and convert
    manifest_list = read.read_manifest(params['train manifest'])
    for file in manifest_list:
        fn = file['fn'].split('/')[-1].split('.')[0] + '.pt'
        fn_labels = os.path.join(paths['labels'], fn)
        labels = make_labels(int(file['nsamples']), int(file['fs']),
                             json.loads(file['sz_starts']),
                             json.loads(file['sz_ends']),
                             params['window length'], params['overlap'])
        labels = torch.tensor(labels, dtype=torch.long)
        torch.save(labels, fn_labels)


if __name__ == '__main__':
    main()
