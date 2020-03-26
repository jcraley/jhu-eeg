import os
import sys

import torch

import preprocessing.dsp as dsp
import utils.pathmanager as pm
import utils.read_files as read
import utils.testconfiguration as tc
from preprocessing.edf_loader import EdfLoader
from preprocessing.eeg_info import EegInfo

DONT_WRITE = ["Segment", "A1+A2 OFF", "+", "Schedule"]


def main():
    """Load the command line args and parse"""
    # Load the configuration files
    argv = sys.argv[1:]
    params = tc.TestConfiguration('default.ini', argv)
    paths = pm.PathManager(params)

    # Read in the connections and make a label list
    label_list = read.read_channel_list(params['channel list'])
    loader = EdfLoader(label_list)

    # Load the manifest files
    manifest_files = read.read_manifest(params['train manifest'])

    for file in manifest_files:
        # Load the edf file
        edf_fn = os.path.join(paths['raw data'], file['fn'])
        print(edf_fn)
        eeg_info = loader.load_metadata(edf_fn)
        for time, annotation in zip(eeg_info.annotations[0],
                                    eeg_info.annotations[2]):
            printable = True
            for word in DONT_WRITE:
                if annotation.startswith(word):
                    printable = False
            if printable:
                print("{0:<7}{1:<50}".format(time, annotation))
        print('')


if __name__ == '__main__':
    main()
