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
        paths.add_feature_folder(feat_name + '_normalized')

        feature_list = []

        # Load the manifest files
        manifest_files = read.read_manifest(params['train manifest'])
        fs = int(manifest_files[0]['fs'])

        # Loop over files and load the features
        nwindows = 0
        for file in manifest_files:
            fn = file['fn'].split('/')[-1].split('.')[0] + '.pt'
            fn_feat = os.path.join(paths[feat_name], fn)
            feature_list.append(torch.load(fn_feat))
            nwindows += feature_list[-1].size(0)
            if torch.sum(torch.isnan(feature_list[-1])) > 0:
                print('{} contains nans'.format(fn))
                feature_list[-1][torch.where(torch.isnan(feature_list[-1]))] = 0
            if torch.sum(torch.isinf(feature_list[-1])) > 0:
                print('{} contains infs'.format(fn))
                feature_list[-1][torch.where(torch.isinf(feature_list[-1]))] = 0

        # Add everything to a tensor
        nchns = feature_list[-1].size(1)
        feat_dim = feature_list[-1].size(2)
        all_data = torch.zeros((nwindows, nchns, feat_dim))
        idx = 0
        for feat in feature_list:
            all_data[idx:idx+feat.size(0), :, :] = feat
            idx += feat.size(0)

        # Take mean and std
        feat_means = torch.mean(all_data, 0)
        feat_stds = torch.std(all_data, 0)

        # Write out
        print('Writing normalized files')
        for file, feature_tensor in zip(manifest_files, feature_list):
            fn = file['fn'].split('/')[-1].split('.')[0] + '.pt'
            fn_feat = os.path.join(paths[feat_name + '_normalized'], fn)
            torch.save((feature_tensor - feat_means) / feat_stds, fn_feat)


if __name__ == '__main__':
    main()
