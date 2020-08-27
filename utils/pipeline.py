import json
import os
import time

import torch
import numpy as np

import utils.evaluation as evaluation
import utils.read_files as read
import utils.visualization as viz
from utils.dataset import EpilepsyDataset

from models.sklearnmodels import *
from models.sklearnchannelmodels import *


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

    def write_config_file(self):
        """Write the config file"""
        new_config_fn = os.path.join(self.paths['trial'], 'config.ini')
        self.params.write(new_config_fn)

    def initialize_train_dataset(self):
        if self.params['load to device']:
            device = self.device
        else:
            device = 'cpu'
        self.train_dataset = EpilepsyDataset(
            self.params['train manifest'],
            self.paths['buffers'],
            self.params['window length'],
            self.params['overlap'],
            device=device,
            features_dir=self.paths['features'],
            features=self.params['features']
        )
        if self.params['load as'] == 'sequences':
            self.train_dataset.set_as_sequences(True)

    def initialize_val_dataset(self):
        if self.params['load to device']:
            device = self.device
        else:
            device = 'cpu'
        self.val_dataset = EpilepsyDataset(
            self.params['val manifest'],
            self.paths['buffers'],
            self.params['window length'],
            self.params['overlap'],
            device=device,
            features_dir=self.paths['features'],
            features=self.params['features']
        )
        if self.params['load as'] == 'sequences':
            self.val_dataset.set_as_sequences(True)

    def initialize_model(self):
        """Initialize the experiment's model"""
        model_kwargs = json.loads(self.params['model kwargs'])
        self.model = eval(self.params['model type'] + '(**model_kwargs)')

    def train(self):
        # Train the model
        print('Training')
        start = time.time()
        self.model.fit(self.train_dataset)
        self.save_model()
        end = time.time()
        print('Training complete in {} seconds'.format(end-start))

    def save_model(self):
        model_fn = os.path.join(self.paths['models'], 'model.pt')
        torch.save(self.model, model_fn)

    def load_model(self):
        self.model = torch.load(self.params['load model fn'])

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
        if hasattr(self.model, "eval"):
            self.model.eval()

        # Check if model can score by channel
        has_predict_by_channel = False
        channel_wise_method = getattr(
            self.model, "predict_channel_proba", None)
        if callable(channel_wise_method):
            has_predict_by_channel = True

        # Loop over dataset and score all
        all_preds = []
        all_labels = []
        all_fns = []
        for seq in dataset:
            # Run and save predictions
            fn = seq['filename'].split('.')[0]
            X = seq['buffers'].to(self.device)
            if ('classifier' in self.params
                    and self.params['classifier'] == 'LstmClassifier'):
                T, c, l = X.size()
                pred = self.model.predict_proba(
                    X.view((1, T, c, l))).view(T, 2)
            else:
                pred = self.model.predict_proba(X)
            if type(pred) == torch.Tensor:
                pred = pred.cpu().detach().numpy()
            pred_fn = os.path.join(self.paths['predictions'], fn + '.pt')
            torch.save(pred, pred_fn)

            # Save prediction information
            all_fns.append(fn)
            all_preds.append(pred)
            label = seq['labels'].detach().cpu().numpy()
            label[np.where(label == 2)] = 0
            all_labels.append(label)
            del pred

            # Save channel predictions
            if has_predict_by_channel:
                channel_pred = self.model.predict_channel_proba(X)
                pred_fn = os.path.join(
                    self.paths['predictions'], fn + '_channel.pt')
                torch.save(channel_pred, pred_fn)

        # If visualize, output pngs for each
        if visualize:
            viz.make_images(all_fns, all_preds, all_labels,
                            self.paths['figures'], prefix, '')

        print("Unsmoothed results")
        # Compute windowise statistics and write out
        evaluation.iid_window_report(all_preds, all_labels,
                                     self.paths['results'], prefix, '')

        # Score based on sequences
        evaluation.sequence_report(all_fns, all_preds, all_labels,
                                   self.paths['results'], prefix, '')

        # Perform the threshold sweep
        evaluation.threshold_sweep(all_preds, all_labels,
                                   self.paths['results'], prefix, '',
                                   dataset.get_total_sz(),
                                   dataset.get_total_duration(),
                                   dataset.get_window_advance_seconds())

        # Check for smoothing and run if so
        if self.params['smoothing'] > 0:
            print("Smoothed results")
            smoothed_preds = evaluation.smooth(
                all_preds, self.params['smoothing'])

            if visualize:
                viz.make_images(all_fns, smoothed_preds, all_labels,
                                self.paths['figures'], prefix, '_smoothed')

            # Compute windowise statistics and write out
            evaluation.iid_window_report(smoothed_preds, all_labels,
                                         self.paths['results'],
                                         prefix, '_smoothed')

            # Score based on sequences
            evaluation.sequence_report(all_fns, smoothed_preds, all_labels,
                                       self.paths['results'], prefix,
                                       '_smoothed')

            # Perform the threshold sweep on the smoothed predictions.
            evaluation.threshold_sweep(smoothed_preds, all_labels,
                                       self.paths['results'], prefix,
                                       '_smoothed', dataset.get_total_sz(),
                                       dataset.get_total_duration(),
                                       dataset.get_window_advance_seconds())
