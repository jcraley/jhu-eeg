import os
import sys

import torch

import preprocessing.features as features
import utils.pathmanager as pm
import utils.read_files as read
import utils.testconfiguration as tc


def main():
    """Set up the experiment, initialize folders, and write config"""
    # Load the configuration files
    argv = sys.argv[1:]
    params = tc.TestConfiguration('default.ini', argv)
    paths = pm.PathManager(params)
    paths.initialize_folder('data')

    for feat_name in params['features']:
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
            feat = eval('features.'
                        + feat_name + '(windowed_buffers, fs=fs)')
            feat_fn = os.path.join(paths[feat_name], fn)
            torch.save(feat, feat_fn)


if __name__ == '__main__':
    main()
