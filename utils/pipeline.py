import json
import os
import time

import torch
import numpy as np

import utils.evaluation as evaluation
import utils.read_files as read
import utils.visualization as viz
from utils.dataset import EpilepsyDataset
from utils.dataset import create_label
from utils.read_files import read_manifest

from models.sklearnmodels import *
from models.sklearnchannelmodels import *


def load_predictions(preds_folder, file_list):
    """Load all predictions

    Args:
        preds_folder ([type]): [description]
        file_list ([type]): [description]
    """

    all_preds = []
    for pred_fn in file_list:
        fn = os.path.join(preds_folder, pred_fn)
        all_preds.append(torch.load(fn))

    return all_preds


class Pipeline():
    """Make an experiment pipeline for doing all things experiment related"""

    def __init__(self, params, paths, device='cpu'):
        """Initialize the experiment pipeline"""
        self.params = params
        self.paths = paths
        self.device = device

        # Initilize things that don't take very long
        manifest = read.read_manifest(self.params['train manifest'])
        self.fs = int(manifest[0]['fs'])
        self.channel_list = read.read_channel_list(self.params['channel list'])
        self.nchns = len(self.channel_list)

        # Initialize the train threshold to None
        self.train_threshold = None

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

    def score(self, dataset):
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
        return all_preds, all_labels, all_fns

    def score_train_dataset(self):
        """Score the training dataset

        If an "fps per hour" value has been set, the corresponding threshold
        will be calculated
        """
        print("Scoring the training dataset")
        print("Forward pass through model")
        all_preds, all_labels, all_fns = self.score(self.train_dataset)

        # If no "fps per hour" is set, self.train_threshold = None
        self.train_threshold = self.score_preds(
            all_preds, all_labels, all_fns,
            'train_', self.params['visualize train'],
            self.paths['figures'], self.paths['results'],
            self.train_dataset.get_total_sz(),
            self.train_dataset.get_total_duration(),
            self.train_dataset.get_window_advance_seconds(),
            fps_per_hr=self.params['fps per hour'],
            fp_time_per_hr=self.params['fp time per hour']
        )
        print("")

    def score_val_dataset(self):
        print("Forward pass through model")
        all_preds, all_labels, all_fns = self.score(self.val_dataset)

        print("Scoring the validation dataset")
        self.score_preds(
            all_preds, all_labels, all_fns,
            'val_', self.params['visualize val'],
            self.paths['figures'], self.paths['results'],
            self.val_dataset.get_total_sz(),
            self.val_dataset.get_total_duration(),
            self.val_dataset.get_window_advance_seconds(),
            threshold=self.train_threshold
        )
        print("")

    def score_saved_val_predictions(self):
        # Read manifest
        manifest = read_manifest(self.params['val manifest'])

        # Load predictions
        all_fns = [eeg['fn'].split('.')[0] + '.pt' for eeg in manifest]
        all_preds = load_predictions(self.paths['predictions'], all_fns)

        all_labels = []
        total_sz = 0
        total_duration = 0
        for eeg in manifest:
            total_sz += len(json.loads(eeg['sz_starts']))
            total_duration += float(eeg['duration'])
            all_labels.append(
                create_label(float(eeg['duration']),
                             json.loads(eeg['sz_starts']),
                             json.loads(eeg['sz_ends']),
                             float(self.params['window length']),
                             float(self.params['overlap'])).numpy())

        window_advance_seconds = (
            self.params['window length'] - self.params['overlap'])
        all_fns = [fn.split('.')[0] for fn in all_fns]
        self.score_preds(all_preds, all_labels, all_fns, 'val_rescore_',
                         self.params['visualize val'], self.paths['figures'],
                         self.paths['results'], total_sz,
                         total_duration, window_advance_seconds,
                         threshold=self.train_threshold
                         )

    def score_saved_train_predictions(self):
        # Read manifest
        manifest = read_manifest(self.params['train manifest'])

        # Load predictions
        all_fns = [eeg['fn'].split('.')[0] + '.pt' for eeg in manifest]
        all_preds = load_predictions(self.paths['predictions'], all_fns)

        all_labels = []
        total_sz = 0
        total_duration = 0
        for eeg in manifest:
            total_sz += len(json.loads(eeg['sz_starts']))
            total_duration += float(eeg['duration'])
            all_labels.append(
                create_label(float(eeg['duration']),
                             json.loads(eeg['sz_starts']),
                             json.loads(eeg['sz_ends']),
                             float(self.params['window length']),
                             float(self.params['overlap'])).numpy())

        window_advance_seconds = (
            self.params['window length'] - self.params['overlap'])
        self.train_threshold = self.score_preds(
            all_preds, all_labels, all_fns, 'train_rescore_',
            self.params['visualize train'], self.paths['figures'],
            self.paths['results'], total_sz,
            total_duration, window_advance_seconds,
            fps_per_hr=self.params['fps per hour'],
            fp_time_per_hr=self.params['fp time per hour']
        )

    def score_preds(self, all_preds, all_labels, all_fns, prefix, visualize,
                    figures_folder, results_folder, total_sz=0,
                    total_duration=0, window_advance_seconds=0, threshold=None,
                    fps_per_hr=0, fp_time_per_hr=0):

        # If visualize, output pngs for each
        if visualize:
            if threshold is None:
                viz.make_images(all_fns, all_preds, all_labels,
                                self.paths['figures'], prefix, '',)
            else:
                viz.make_images(all_fns, all_preds, all_labels,
                                self.paths['figures'], prefix, '',
                                threshold=threshold)

        print("Unsmoothed results")

        # Perform the threshold sweep
        sweep_results = evaluation.threshold_sweep(
            all_preds, all_labels, self.paths['results'], prefix,
            '', self.params['max samples before sz'],
            total_sz, total_duration, window_advance_seconds,
            self.params['count post sz']
        )

        # If a threshold is not specified and an allowable fps_per_hour is,
        # compute the corresponding threshold
        if (threshold is None and self.params['smoothing'] == 0
                and (fps_per_hr > 0 or fp_time_per_hr > 0)):
            # Decrease the threshold until the fp criteria is met
            threshold_idx = len(sweep_results['thresholds']) - 1
            stop = False
            while not stop:
                # Check if fps per hour is set and exceeds limit
                if (fps_per_hr > 0 and
                    sweep_results['fps_per_hour'][threshold_idx]
                        > fps_per_hr):
                    stop = True
                # Check if fp time per hour is set and exceeds limit
                elif (fp_time_per_hr > 0 and
                        sweep_results['fp_time_per_hour'][threshold_idx]
                        > fp_time_per_hr):
                    stop = True
                elif threshold_idx == 10:
                    stop = True
                else:
                    threshold_idx -= 1
            # Get the threshold
            threshold = sweep_results['thresholds'][threshold_idx]
            fn = '{}threshold{}.txt'.format(prefix, '')
            with open(os.path.join(self.paths['results'], fn), 'w') as f:
                f.write(str(threshold) + '\n')

        # Compute windowise statistics and write out
        evaluation.iid_window_report(all_preds, all_labels,
                                     self.paths['results'], prefix, '',
                                     threshold=threshold)

        # Score based on sequences
        evaluation.sequence_report(
            all_fns, all_preds, all_labels, self.paths['results'], prefix, '',
            threshold=threshold,
            max_samples_before_sz=self.params['max samples before sz'],
            nsz=total_sz, total_duration=total_duration,
            window_advance_seconds=window_advance_seconds,
            count_post_sz=self.params['count post sz']
        )

        # Check for smoothing and run if so
        if self.params['smoothing'] > 0:
            print("Smoothed results")
            smoothed_preds = evaluation.smooth(
                all_preds, self.params['smoothing'])

            # Perform the threshold sweep on the smoothed predictions.
            sweep_results = evaluation.threshold_sweep(
                smoothed_preds, all_labels, self.paths['results'], prefix,
                '_smoothed', self.params['max samples before sz'], total_sz,
                total_duration, window_advance_seconds,
                self.params['count post sz']
            )

            # If a threshold is not specified and an allowable fps_per_hour is,
            # compute the corresponding threshold
            if threshold is None and (fps_per_hr > 0 or fp_time_per_hr > 0):
                # Decrease the threshold until the fp criteria is met
                threshold_idx = len(sweep_results['thresholds']) - 1
                stop = False
                while not stop:
                    # Check if fps per hour is set and exceeds limit
                    if (fps_per_hr > 0 and
                        sweep_results['fps_per_hour'][threshold_idx]
                            > fps_per_hr):
                        stop = True
                    # Check if fp time per hour is set and exceeds limit
                    elif (fp_time_per_hr > 0 and
                          sweep_results['fp_time_per_hour'][threshold_idx]
                            > fp_time_per_hr):
                        stop = True
                    elif threshold_idx == 10:
                        stop = True
                    else:
                        threshold_idx -= 1
                # Get the threshold
                threshold = sweep_results['thresholds'][threshold_idx]
                fn = '{}threshold{}.txt'.format(prefix, '_smoothed')
                with open(os.path.join(self.paths['results'], fn), 'w') as f:
                    f.write(str(threshold) + '\n')

            # Compute windowise statistics and write out
            evaluation.iid_window_report(smoothed_preds, all_labels,
                                         self.paths['results'],
                                         prefix, '_smoothed',
                                         threshold=threshold)

            # Score based on sequences
            evaluation.sequence_report(all_fns, smoothed_preds, all_labels,
                                       self.paths['results'], prefix,
                                       '_smoothed', threshold,
                                       self.params['max samples before sz'],
                                       total_sz,
                                       total_duration,
                                       window_advance_seconds,
                                       self.params['count post sz'])

            # Visualize
            if visualize:
                viz.make_images(all_fns, smoothed_preds, all_labels,
                                self.paths['figures'], prefix, '_smoothed',
                                threshold=threshold)

        return threshold
