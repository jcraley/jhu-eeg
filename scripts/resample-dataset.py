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
    paths['buffers'] = paths['buffers'] + '_200'
    paths.initialize_folder('buffers')

    # Read in the channel list
    label_list = read.read_channel_list(params['channel list'])
    loader = EdfLoader(label_list)

    # Load the manifest files
    manifest_files = read.read_manifest(params['train manifest'])

    for file in manifest_files:
        # Load the edf file
        edf_fn = os.path.join(paths['raw data'], file['fn'])
        print(edf_fn)
        eeg_info = loader.load_metadata(edf_fn)
        buffers = loader.load_buffers(eeg_info)
        print(eeg_info.fs)
        buffers = dsp.resample_256to200(buffers)
        eeg_info.fs = 200
        print(eeg_info.fs)
        buffers = dsp.prefilter(buffers, eeg_info.fs,
                                params['notch'], params['lpf fc'],
                                params['hpf fc'], params['clip level'],
                                params['normalize'])
        buffers = torch.tensor(buffers, dtype=torch.float32).transpose(0, 1)

        fn_out = edf_fn.split('/')[-1].split('.')[0] + '.pt'
        fn_out = os.path.join(paths['buffers'], fn_out)
        torch.save(buffers, fn_out)


if __name__ == '__main__':
    main()
