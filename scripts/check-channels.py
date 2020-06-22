import os
import sys

import torch

import preprocessing.dsp as dsp
import utils.pathmanager as pm
import utils.read_files as read
import utils.testconfiguration as tc
from preprocessing.edf_loader import EdfLoader
from preprocessing.eeg_info import EegInfo


def main():
    """Load the command line args and parse"""
    # Load the configuration files
    argv = sys.argv[1:]
    params = tc.TestConfiguration('default.ini', argv)
    paths = pm.PathManager(params)
    paths.initialize_folder('buffers')

    # Initialize the loader
    loader = EdfLoader()

    # Load the manifest files
    manifest_files = read.read_manifest(params['train manifest'])

    # Load the edf file
    edf_fn = os.path.join(paths['raw data'], manifest_files[0]['fn'])
    print(edf_fn)
    eeg_info = loader.load_metadata(edf_fn)
    label_lists = [eeg_info.label_list]
    label_set = set(eeg_info.label_list)

    for eeg in manifest_files[1:]:
        # Load the edf file
        edf_fn = os.path.join(paths['raw data'], eeg['fn'])
        print(edf_fn)
        eeg_info = loader.load_metadata(edf_fn)
        label_lists.append(eeg_info.label_list)
        label_set = label_set & set(eeg_info.label_list)

    print(len(label_set))
    print(label_set)


if __name__ == '__main__':
    main()
