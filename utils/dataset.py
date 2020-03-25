import os
import time

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.read_files import *


def compute_nwindows(duration, window_length, overlap):
    """Compute the number of windows in a file"""
    advance = window_length * (1 - overlap)
    return int(np.floor((duration - window_length) / advance)) + 1


def load_labels(total_windows, manifest_files, labels_dir):
    # Load the labels
    labels = torch.zeros(total_windows, dtype=torch.long)
    idx = 0
    for file in manifest_files:
        fn = file['fn'].split('.')[0] + '.pt'
        next_labels = torch.load(os.path.join(labels_dir, fn))
        nsamples = next_labels.size(0)
        labels[idx:idx + nsamples] = next_labels
        idx += nsamples
    return labels


class EpilepsyDataset(Dataset):
    """Load pre-windowed and pre-processed EDFs into a dataset
    """

    def __init__(self, manifest_fn, data_dir, labels_dir,
                 window_length, overlap, device='cpu', features_dir='',
                 features=[]):
        self.as_sequences = False
        self.data_dir = data_dir
        self.labels_dir = labels_dir
        self.window_length = window_length
        self.overlap = overlap
        self.device = device
        self.features_dir = features_dir
        self.features = features

        # Read manifest, get number of channels and sample frequency
        self.manifest_files = read_manifest(manifest_fn)
        self.nchns = int(self.manifest_files[0]['nchns'])
        self.fs = int(self.manifest_files[0]['fs'])
        self.nfiles = len(self.manifest_files)

        # Get the number of windows for the entire dataset
        self.nwindows = 0
        for file in self.manifest_files:
            duration = float(file['duration'])
            self.nwindows += compute_nwindows(duration, window_length, overlap)
        self.labels = load_labels(self.nwindows, self.manifest_files,
                                  self.labels_dir)

        # Load features
        if features:
            self.load_features()
        else:
            self.load_windowed_buffers()

    def load_windowed_buffers(self):
        # Initialize the data tensors
        window_length_samples = int(self.fs * self.window_length)
        self.data = torch.zeros(
            (self.nwindows, self.nchns, window_length_samples),
            dtype=torch.float32, device=self.device)
        self.filenames = []
        self.sequence_indices = []

        # Load files into the data tensors
        print('Loading files', flush=True)
        start = time.time()
        idx = 0
        for file in self.manifest_files:
            fn = file['fn'].split('.')[0] + '.pt'
            self.filenames.append(fn)
            curr_file = torch.load(os.path.join(self.data_dir, fn),
                                   map_location=self.device)
            nsamples = curr_file.size(0)
            self.data[idx:idx + nsamples, :, :] = curr_file
            # store the indices for each recording
            self.sequence_indices.append([idx, idx + nsamples])
            idx += nsamples
        end = time.time()

        self.d_out = [self.data.size(1), self.data.size(2)]
        print('Loaded in {} seconds'.format(end - start), flush=True)

    def load_features(self):
        # Load the features

        start = time.time()

        # Load the first files
        first_files = []
        feat_dims = []
        fn = self.manifest_files[0]['fn'].split('.')[0] + '.pt'
        self.filenames = [fn]
        for feat in self.features:
            first_files.append(torch.load(os.path.join(
                self.features_dir, feat, fn), map_location=self.device))
            feat_dims.append(first_files[-1].size(2))

        # Initialize the data tensors
        self.data = torch.zeros((self.nwindows, self.nchns, sum(feat_dims)),
                                dtype=torch.float32, device=self.device)
        feat_idx = 0
        for feat in first_files:
            nsamples, _, feat_dim = feat.size()
            self.data[:nsamples, :, feat_idx:feat_idx + feat_dim] = feat
            feat_idx += feat_dim
        self.sequence_indices = [[0, nsamples]]

        window_idx = nsamples
        for file in self.manifest_files[1:]:
            fn = file['fn'].split('.')[0] + '.pt'
            self.filenames.append(fn)

            # Load the features
            feat_idx = 0
            for feat in self.features:
                curr_file = (
                    torch.load(os.path.join(self.features_dir, feat, fn),
                               map_location=self.device))
                nsamples, _, feat_dim = curr_file.size()
                self.data[window_idx:window_idx + nsamples,
                          :, feat_idx:feat_idx + feat_dim] = curr_file
                feat_idx += feat_dim

            # Track sequence indices
            self.sequence_indices.append(
                [window_idx, window_idx + nsamples])
            window_idx += nsamples

        end = time.time()
        print('Loaded in {} seconds'.format(end - start), flush=True)

        self.d_out = [self.data.size(1), self.data.size(2)]

    def set_as_sequences(self, as_sequences):
        self.as_sequences = as_sequences

    def __len__(self):
        if self.as_sequences:
            return self.nfiles
        else:
            return self.nwindows

    def __getitem__(self, idx):
        if self.as_sequences:
            start_idx, end_idx = self.sequence_indices[idx]
            return {'buffers': self.data[start_idx:end_idx],
                    'labels': self.labels[start_idx:end_idx],
                    'filename': self.filenames[idx]}
        else:
            return {'buffers': self.data[idx], 'labels': self.labels[idx]}

    def class_counts(self):
        return torch.bincount(self.labels)
