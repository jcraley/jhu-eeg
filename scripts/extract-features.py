import os
import sys

import torch

import preprocessing.features as features
import utils.pathmanager as pm
import utils.read_files as read
import utils.testconfiguration as tc
from utils.dataset import EpilepsyDataset


def main():
    """Set up the experiment, initialize folders, and write config"""
    # Load the configuration files
    argv = sys.argv[1:]
    params = tc.TestConfiguration('default.ini', argv)
    paths = pm.PathManager(params)
    paths.initialize_folder('data')

    # Load the dataset
    train_dataset = EpilepsyDataset(
        params['train manifest'],
        paths['buffers'],
        params['window length'],
        params['overlap'],
        device='cpu',
    )
    train_dataset.set_as_sequences(True)

    for feat_name in params['features']:
        print("Extracting {}".format(feat_name))
        paths.add_feature_folder(feat_name)

        # Load the manifest files
        manifest_files = read.read_manifest(params['train manifest'])
        fs = int(manifest_files[0]['fs'])

        # Loop over files and create windowed versions
        for sample in train_dataset:
            fn = sample['filename'].split('/')[-1].split('.')[0] + '.pt'
            windowed_buffers = sample['buffers']
            feat = eval('features.'
                        + feat_name + '(windowed_buffers, fs=fs)')
            feat_fn = os.path.join(paths[feat_name], fn)
            torch.save(feat, feat_fn)


if __name__ == '__main__':
    main()
