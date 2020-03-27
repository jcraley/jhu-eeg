import csv
import json
import os
import pickle
import time

import numpy as np
import torch

import utils.evaluation as evaluation
import utils.output_tools as out
import utils.pathmanager as pm
import utils.read_files as read
import utils.testconfiguration as tc
import utils.visualization as viz
from models.sklearnmodels import LogisticRegression
from utils.dataset import EpilepsyDataset


def make_images(fns, preds, labels, viz_folder, prefix, suffix):
    for fn, pred, label in zip(fns, preds, labels):
        pic_fn = os.path.join(viz_folder,
                              '{}{}{}.png'.format(prefix, fn, suffix))
        viz.plot_yhat(pred, label, fn=pic_fn)


def iid_window_report(all_preds, all_labels, report_folder, prefix, suffix):
    # Compute window based statistics and write out
    stats = evaluation.compute_metrics(np.concatenate(all_labels),
                                       np.concatenate(all_preds))
    stats_fn = os.path.join(report_folder,
                            '{}stats{}.pkl'.format(prefix, suffix))
    pr_fn = os.path.join(report_folder,
                         '{}pr{}.pkl'.format(prefix, suffix))
    roc_fn = os.path.join(report_folder,
                          '{}roc{}.pkl'.format(prefix, suffix))
    results_fn = os.path.join(report_folder,
                              '{}iid_results{}.csv'.format(prefix, suffix))
    with open(stats_fn, 'wb') as f:
        pickle.dump(stats, f)
    with open(pr_fn, 'wb') as f:
        pickle.dump(stats['pr curve'], f)
    with open(roc_fn, 'wb') as f:
        pickle.dump(stats['roc curve'], f)
    with open(results_fn, 'w', newline='') as csvfile:
        fieldnames = ['acc', 'sens', 'spec', 'prec', 'f1', 'auc-roc',
                      'auc-pr']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({k: stats[k] for k in fieldnames})

    frmt = "{:<8}"*len(fieldnames)
    print(frmt.format(*fieldnames))
    frmt = "{:<8.3f}"*len(fieldnames)
    print(frmt.format(*[stats[k] for k in fieldnames]))


def sequence_report(all_fns, all_preds, all_labels, report_folder, prefix,
                    suffix):
    # Score based on sequences
    total_fps = 0
    total_latency_samples = 0
    total_correct = 0
    all_results = []
    for fn, pred, label in zip(all_fns, all_preds, all_labels):
        stats = evaluation.score_recording(label, pred)
        all_results.append({
            'fn': fn,
            'nfps': stats['nfps'],
            'latency_samples': stats['latency_samples'],
            'ncorrect': stats['ncorrect'],
        })
        total_fps += stats['nfps']
        total_latency_samples += stats['latency_samples']
        total_correct += stats['ncorrect']

    # Write sequence stats
    results_fn = os.path.join(report_folder,
                              '{}seizure_results{}.csv'.format(prefix, suffix))
    with open(results_fn, 'w', newline='') as csvfile:
        fieldnames = ['fn', 'nfps', 'latency_samples', 'ncorrect']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)


def smooth(all_preds, smoothing):
    smoothed_preds = []
    for pred in all_preds:
        new_pred = np.zeros_like(pred)
        for ii in range(len(pred)):
            start = max(ii - smoothing // 2, 0)
            end = ii + smoothing // 2
            new_pred[ii, :] = np.mean(pred[start:end, :], axis=0)
        smoothed_preds.append(new_pred)
    return smoothed_preds


class Pipeline():
    """Make an experiment pipeline for doing all things experiment related"""

    def __init__(self, params, paths, device='cpu'):
        """Initialize the experiment pipeline"""
        self.params = params
        self.paths = paths
        self.device = device

        # Initilize things that don't very long
        manifest = read.read_manifest(self.params['train manifest'])
        self.fs = int(manifest[0]['fs'])
        self.channel_list = read.read_channel_list(self.params['channel list'])
        self.nchns = len(self.channel_list)
        print(self.channel_list)

    def write_config_file(self):
        """Write the config file"""
        new_config_fn = os.path.join(self.paths['trial'], 'config.ini')
        self.params.write(new_config_fn)

    def initialize_train_dataset(self):
        self.train_dataset = EpilepsyDataset(
            self.params['train manifest'],
            self.paths['data'],
            self.paths['labels'],
            self.params['window length'],
            self.params['overlap'],
            device='cpu',
            features_dir=self.paths['features'],
            features=self.params['features']
        )

    def initialize_val_dataset(self):
        self.val_dataset = EpilepsyDataset(
            self.params['val manifest'],
            self.paths['data'],
            self.paths['labels'],
            self.params['window length'],
            self.params['overlap'],
            device='cpu',
            features_dir=self.paths['features'],
            features=self.params['features']
        )

    def initialize_model(self):
        """Initialize the experiment's model"""
        model_kwargs = json.loads(self.params['model kwargs'])
        self.model = eval(self.params['model type'] + '(**model_kwargs)')

    def train(self):
        # Train the model
        self.model.fit(self.train_dataset)
        model_fn = os.path.join(self.paths['models'], 'model.pt')
        torch.save(self.model, model_fn)

    def score_train_dataset(self):
        print("Scoring the training dataset")
        self.score_dataset(self.train_dataset, 'train_',
                           self.params['visualize train'],
                           self.paths['figures'], self.paths['results'])
        print("")

    def score_val_dataset(self):
        print("Scoring the validation dataset")
        self.score_dataset(self.val_dataset, 'val_',
                           self.params['visualize val'],
                           self.paths['figures'], self.paths['results'])
        print("")

    def score_dataset(self, dataset, prefix, visualize,
                      figures_folder, results_folder):
        dataset.set_as_sequences(True)

        # Loop over dataset and score all
        all_preds = []
        all_labels = []
        all_fns = []
        for file in dataset:
            # Run and save predictions
            fn = file['filename'].split('.')[0]
            X = file['buffers']
            pred = self.model.predict_proba(X)
            pred_fn = os.path.join(self.paths['predictions'], fn + '.pt')
            torch.save(pred, pred_fn)

            # Save prediction information
            all_fns.append(fn)
            all_preds.append(pred)
            all_labels.append(file['labels'].detach().cpu().numpy())

        # If visualize, output pngs for each
        if visualize:
            make_images(all_fns, all_preds, all_labels,
                        self.paths['figures'], prefix, '')

        print("Unsmoothed results")
        # Compute windowise statistics and write out
        iid_window_report(all_preds, all_labels, self.paths['results'],
                          prefix, '')

        # Score based on sequences
        sequence_report(all_fns, all_preds, all_labels, self.paths['results'],
                        prefix, '')

        # Check for smoothing and run if so
        if self.params['smoothing'] > 0:
            print("Smoothed results")
            smoothed_preds = smooth(all_preds, self.params['smoothing'])

            if visualize:
                make_images(all_fns, smoothed_preds, all_labels,
                            self.paths['figures'], prefix, '_smoothed')

            # Compute windowise statistics and write out
            iid_window_report(smoothed_preds, all_labels, self.paths['results'],
                              prefix, '_smoothed')

            # Score based on sequences
            sequence_report(all_fns, smoothed_preds, all_labels,
                            self.paths['results'], prefix, '_smoothed')
