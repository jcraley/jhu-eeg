import os
import time
import json

import numpy as np
import torch
from torch.utils.data import Dataset

import utils.read_files as read


def normalize(buffers):
    """Given a set of buffers in (T, C, L) perform normalization
    """
    ndim = buffers.ndim
    mean = buffers.mean(dim=ndim-1).unsqueeze(ndim-1)
    std = buffers.std(dim=ndim-1).unsqueeze(ndim-1)
    std[torch.where(std == 0)] = 1.0
    return torch.div(buffers - mean, std)


def compute_nwindows(duration, window_length, overlap):
    """Compute the number of windows for a recording

    Args:
        duration (float): Length of recording in seconds
        window_length (float): Length of the window in seconds
        overlap (float): Window overlap in seconds

    Returns:
        int: Length of the recording in windows
    """
    advance = window_length - overlap
    return int(np.floor((duration - window_length) / advance)) + 1


def create_label(duration, sz_starts, sz_ends, window_length, overlap,
                 post_sz_state=False):
    """Create a tensor of labels

    Arguments:
        duration {float} -- Length of file in seconds
        sz_starts {list} -- Seizure starts in seconds
        sz_ends {list} -- Seizure ends in seconds
        window_length {float} -- Length of a window in seconds
        overlap {float} -- Window overlap in seconds

    Returns:
        torch.tensor -- Long tensor of labels
    """
    nwindows = compute_nwindows(duration, window_length, overlap)
    # Initialize all labels to 0
    label = torch.zeros(nwindows, dtype=torch.long)
    window_time = (window_length - overlap) * np.arange(nwindows)
    if post_sz_state:
        label[np.where(window_time >= sz_ends[0])] = 2
    # Set any labels between starts to 1
    for start, end in zip(sz_starts, sz_ends):
        label[np.where((window_time >= start) * (window_time <= end))] = 1
    return label


def onset_zone_to_lateralization(onset_zone):
    """Returns the lateralization target from the onset zone

    Args:
        onset_zone (int): onset zone

    Returns:
        torch tensor long: binary lateralization label
    """

    # Set target to 0, 1
    if onset_zone == 1 or onset_zone == 3:
        return 0
    if onset_zone == 2 or onset_zone == 4:
        return 1


def onset_zone_to_lobe(onset_zone):
    """Returns the lateralization target from the onset zone

    Args:
        onset_zone (int): onset zone

    Returns:
        torch tensor long: binary lateralization label
    """

    # Set target to 0, 1
    if onset_zone == 1 or onset_zone == 2:
        return 0
    if onset_zone == 3 or onset_zone == 4:
        return 1


class EpilepsyDataset(Dataset):
    """Load pre-windowed and pre-processed EDFs into a dataset
    """

    def __init__(self, manifest_fn, data_dir,
                 window_length, overlap, device='cpu', features_dir='',
                 features=[], no_load=False, normalize_windows=False,
                 post_sz=False, transform=None):
        self.as_sequences = False
        self.data_dir = data_dir

        self.device = device
        self.features_dir = features_dir
        self.features = features
        self.normalize_windows = normalize_windows
        self.post_sz = post_sz
        self.transform = transform

        # Read manifest, get number of channels and sample frequency
        self.manifest_files = read.read_manifest(manifest_fn)
        self.nchns = int(self.manifest_files[0]['nchns'])
        self.fs = int(self.manifest_files[0]['fs'])
        self.nfiles = len(self.manifest_files)

        # Set window parameters and compute window sample lengths
        assert overlap < window_length, "Overlap is longer than window"
        self.window_length = window_length
        self.overlap = overlap
        self.window_advance_seconds = window_length - overlap
        self.window_samples = int(window_length * self.fs)
        self.window_advance_samples = int(
            self.window_advance_seconds * self.fs)

        # Create the labels for the dataset
        self.labels = []
        self.start_windows = []
        self.onset_zones = []
        self.lateralizations = []
        self.lobes = []
        self.patient_numbers = []
        window_idx = 0
        for recording in self.manifest_files:
            label = create_label(float(recording['duration']),
                                 json.loads(recording['sz_starts']),
                                 json.loads(recording['sz_ends']
                                            ), self.window_length,
                                 self.overlap, post_sz_state=self.post_sz)
            label = label.to(self.device)
            self.labels.append(label)
            self.start_windows.append(window_idx)
            window_idx += len(self.labels[-1])

            oz = int(recording['onset_zone'])
            self.onset_zones.append(oz)
            self.lateralizations.append(onset_zone_to_lateralization(oz))
            self.lobes.append(onset_zone_to_lobe(oz))
            self.patient_numbers.append(recording['pt_num'])
        self.nwindows = window_idx

        # Get the total duration and number of seizures
        self.total_duration = 0
        self.total_sz = 0
        for recording in self.manifest_files:
            self.total_duration += float(recording['duration'])
            self.total_sz += len(json.loads(recording['sz_starts']))

        # Load features
        if not no_load:
            if self.features:
                self._load_features()
            else:
                self._load_buffers()

    def load_data(self):
        if self.features:
            self._load_features()
        else:
            self._load_buffers()

    def _load_buffers(self):
        """Load buffers into a list of buffers
        """
        # Load all files into a list
        print('Loading files', flush=True)
        start = time.time()
        self.buffer_list = []
        self.filenames = []
        self.buffer_windows = []

        for eeg in self.manifest_files:
            # Load the relevant file
            fn = eeg['fn'].split('.')[0] + '.pt'
            curr_file = torch.load(os.path.join(self.data_dir, fn),
                                   map_location=self.device)

            # Store the current buffer in the list of buffers
            self.filenames.append(fn)
            self.buffer_list.append(curr_file)

            # Keep track of what window index starts with each file
            nwindows = compute_nwindows(int(eeg['duration']),
                                        self.window_length,
                                        self.overlap)
            self.buffer_windows.append(nwindows)

        end = time.time()
        print('Loaded in {} seconds'.format(end - start), flush=True)

        # Set the length of the output
        self.d_out = [self.nchns, self.window_samples]

    def _load_features(self):
        """Load features

        Features are loaded into a single tensor self.data. This tensor is
        of dimensions (nsamples, nchns, feature lengths).
        """
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
        # Get the label
        sample = {}
        if self.as_sequences:
            sample['labels'] = self.labels[idx]
            sample['filename'] = self.filenames[idx]
            sample['onset zone'] = self.onset_zones[idx]
            sample['lateralization'] = self.lateralizations[idx]
            sample['lobe'] = self.lobes[idx]
            sample['patient number'] = self.patient_numbers[idx]
            if self.features:
                start_idx, end_idx = self.sequence_indices[idx]
                sample['buffers'] = self.data[start_idx:end_idx]
            else:
                windowed_buffer = self.buffer_list[0].new_zeros(
                    (self.buffer_windows[idx],
                     self.nchns, self.window_samples),
                    device=self.device)
                for ii in range(self.buffer_windows[idx]):
                    start = ii * self.window_advance_samples
                    end = ii * self.window_advance_samples + self.window_samples
                    windowed_buffer[ii, :, :] = torch.transpose(
                        self.buffer_list[idx][start:end, :], 0, 1)
                sample['buffers'] = windowed_buffer
        else:
            # Find the buffer containing the window
            for buffer_idx, start_window in enumerate(self.start_windows):
                if start_window > idx:
                    buffer_idx -= 1
                    break
            # Calculate the indexes of the start and end of the window
            window_number = idx - self.start_windows[buffer_idx]
            sample['labels'] = self.labels[buffer_idx][window_number]
            sample['filename'] = self.filenames[buffer_idx]
            sample['onset zones'] = self.onset_zones[buffer_idx]
            sample['lateralization'] = self.lateralizations[idx]
            sample['lobe'] = self.lobes[idx]
            sample['patient numbers'] = self.patient_numbers[buffer_idx]
            if self.features:
                sample['buffers'] = self.data[idx]
            else:
                sample_start = window_number * self.window_advance_samples
                sample['buffers'] = torch.transpose(
                    self.buffer_list[buffer_idx][sample_start:sample_start
                                                 + self.window_samples, :],
                    0, 1)
        if self.normalize_windows:
            sample['buffers'] = normalize(sample['buffers'])
        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_all_data(self):
        if self.features:
            return self.data
        else:
            all_data = torch.zeros(
                (self.start_windows[-1] + self.buffer_windows[-1],
                 self.nchns, self.window_samples))
            idx = 0
            for buffer, nwindows in zip(self.buffer_list, self.buffer_windows):
                for ii in range(nwindows):
                    start = ii * self.advance_samples
                    end = ii * self.advance_samples + self.window_samples
                    all_data[idx, :, :] = torch.transpose(
                        buffer[start:end, :], 0, 1)
                    idx += 1
            return all_data

    def get_all_labels(self):
        return torch.cat(self.labels)

    def class_counts(self):
        if self.post_sz:
            counts = torch.zeros(3)
            for label in self.labels:
                new_counts = torch.bincount(label)
                counts[:len(new_counts)] += new_counts
            return counts
        return sum([torch.bincount(label) for label in self.labels])

    def get_total_sz(self):
        return self.total_sz

    def get_total_duration(self):
        return self.total_duration

    def get_window_advance_seconds(self):
        return self.window_advance_seconds

    def get_pt_onset_zone(self, pt):
        return self.onset_zones[self.patient_numbers.index(pt)]
