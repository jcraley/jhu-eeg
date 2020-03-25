import csv
import json
import os
import pickle
import time
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression

import utils.evaluation as evaluation
import utils.output_tools as out
import utils.pathmanager as pm
import utils.read_files as read
import utils.testconfiguration as tc
import utils.visualization as viz
from utils.dataset import EpilepsyDataset


# def collate_sequences(batch):
#     data = [item['buffers'] for item in batch]
#     target = [item['labels'] for item in batch]
#     return {'buffers': data, 'labels': target}


# def create_adjacency_matrix(connection_fn):
#     """Create an adjacency matrix for a given connections file
#     """
#     # Load labels and edges
#     label_list = []
#     edge_lists = []
#     with open(connection_fn, 'r') as adj_file:
#         for line in adj_file:
#             # Get adjacencies
#             node, edges = line.strip().split(':')
#             node = node.upper()
#             edges = edges.upper().split(',')
#             label_list.append(node)
#             edge_lists.append(edges)

#     # Create an adjacency create matrix
#     nchns = len(label_list)
#     A = torch.zeros((nchns, nchns))
#     for ii, edge_list in enumerate(edge_lists):
#         for edge in edge_list:
#             A[ii, label_list.index(edge)] = 1.0
#     return A


def make_images(fns, preds, labels, viz_folder, prefix, suffix):
    for fn, pred, label in zip(fns, preds, labels):
        pic_fn = os.path.join(viz_folder,
                              '{}{}{}.png'.format(prefix, fn, suffix))
        viz.plot_yhat(pred, label, fn=pic_fn)


def window_report(all_preds, all_labels, report_folder, prefix, suffix):
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
        # Read label list
        with open(self.params['channel list']) as f:
            self.channel_list = f.readlines()
        self.channel_list = [label.strip() for label in self.channel_list]
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

    # def initialize_data_loaders(self):
    #     """Create the dataloaders"""
    #     loader_kwargs = {
    #         'batch_size': self.params['batch size'],
    #         'num_workers': 0,
    #         'shuffle': True,
    #     }
    #     if self.params['load as'] == 'sequences':
    #         loader_kwargs['collate_fn'] = collate_sequences
    #     train_dataloader = DataLoader(self.train_dataset, **loader_kwargs)
    #     val_dataloader = DataLoader(self.val_dataset, **loader_kwargs)
    #     self.dataloaders = {
    #         'train': train_dataloader,
    #         'val': val_dataloader
    #     }

    def initialize_model(self):
        """Initialize the experiment's model"""

        # Initialize the encoder
        # nchns, T = self.train_dataset.d_out
        # encoder_kwargs = json.loads(self.params['encoder kwargs'])
        # encoder_kwargs.update({'nchns': nchns, 'T': T})
        # encoder_str = self.params['encoder'] + '(**encoder_kwargs)'
        # encoder = eval(encoder_str)

        # # If specified load a pretrained encoder
        # if self.params['load encoder cfg'] is not '':
        #     encoder_exp_params = tc.TestConfiguration(
        #         self.params['load encoder cfg'])
        #     encoder_exp_paths = pm.PathManager(encoder_exp_params)
        #     classifier_kwargs = json.loads(encoder_exp_params['classifier kwargs'])
        #     classifier_str = (encoder_exp_params['classifier']
        #                       + '(encoder.d_out, **classifier_kwargs)')
        #     classifier = eval(classifier_str)
        #     load_model = EncoderClassifier(encoder, classifier)
        #     model_fn = os.path.join(encoder_exp_paths['models'],
        #                             'model.pt')
        #     load_model.load_state_dict(torch.load(model_fn))
        #     encoder = load_model.encoder

        # print(encoder.d_out)
        # # Initialize the classifier
        # classifier_kwargs = json.loads(self.params['classifier kwargs'])
        # classifier_str = (self.params['classifier']
        #                   + '(encoder.d_out, **classifier_kwargs)')
        # classifier = eval(classifier_str)

        # # Combine encoder and classifier
        # self.model = EncoderClassifier(encoder, classifier)
        # self.model.float()
        # self.model = self.model.to(self.device)
        model_kwargs = json.loads(self.params['model kwargs'])
        self.model = eval(self.params['model type'] + '(**model_kwargs)')

    # def load_model(self):
    #     """Load the model"""
    #     self.initialize_model()
    #     model_fn = os.path.join(self.paths['models'], 'model.pt')
    #     self.model.load_state_dict(torch.load(model_fn))
    #     self.model.eval()

    # def initialize_loss(self):
    #     """Set up the loss function, optimizer, and lr scheduler"""
    #     class_counts = self.train_dataset.class_counts()
    #     class_counts[class_counts == 0] = 1
    #     alpha = None
    #     if self.params['weight by class']:
    #         weights = torch.sum(class_counts).float() / class_counts.float()
    #         alpha = weights
    #         alpha = alpha.to(self.device)
    #     self.criterion = FocalLoss(gamma=self.params['gamma'], alpha=alpha)

    # def initialize_combined_losses(self):
    #     """Set up the sequence loss"""
    #     # Focal Loss
    #     class_counts = self.train_dataset.class_counts()
    #     class_counts[class_counts == 0] = 1
    #     alpha = None
    #     if self.params['weight by class']:
    #         weights = torch.sum(class_counts).float() / class_counts.float()
    #         alpha = weights
    #         alpha = alpha.to(self.device)
    #     self.focalloss = FocalLoss(gamma=self.params['gamma'], alpha=alpha)
    #     self.focalloss = self.focalloss.to(self.device)

    #     # Seizure Loss
    #     transition_counts = self.train_dataset.transition_counts()
    #     A = transition_counts / torch.sum(transition_counts, dim=1).view(-1, 1)
    #     A = A.to(self.device)
    #     print(A)
    #     self.seizureloss = SeizureLoss(A, self.device)
    #     self.seizureloss = self.seizureloss.to(self.device)

    #     # Combine losses
    #     lamba_fl = torch.tensor(self.params['lambda fl']).to(self.device)
    #     lambda_seizure = torch.tensor(
    #         self.params['lambda seizure']).to(self.device)
    #     self.criterion = CombinedLoss([self.focalloss, self.seizureloss],
    #                                   [lamba_fl, lambda_seizure],
    #                                   self.device)
    #     self.criterion = self.criterion.to(self.device)

    # def initialize_training(self):

    #     self.optimizer = optim.Adam(
    #         self.model.parameters(),
    #         lr=self.params['lr'],
    #         weight_decay=self.params['weight decay']
    #     )
    #     self.exp_lr_scheduler = lr_scheduler.StepLR(
    #         self.optimizer,
    #         step_size=self.params['step size'],
    #         gamma=self.params['schedule gamma']
    #     )

    # def train(self, save_model=True, save_history=True,
    #           visualize_history=True):
    #     """Train the model"""
    #     since = time.time()
    #     self.model, self.history = train_model(
    #         self.model,
    #         self.dataloaders,
    #         self.criterion, self.optimizer,
    #         self.exp_lr_scheduler, self.device,
    #         num_epochs=self.params['epochs'],
    #         save_folder=None
    #     )
    #     time_elapsed = time.time() - since
    #     print('Training complete in {:.0f}m {:.0f}s'.format(
    #         time_elapsed // 60, time_elapsed % 60), flush=True)

    #     if save_model:
    #         # self.model.to('cpu')
    #         torch.save(self.model.state_dict(),
    #                    os.path.join(self.paths['models'], 'model.pt'))
    #     if save_history:
    #         fn = os.path.join(self.paths['trial'], 'history.pkl')
    #         out.save_obj(self.history, fn)
    #     if visualize_history:
    #         fn = os.path.join(self.paths['figures'], 'history.png')
    #         viz.visualize_history(self.history, fn)
    
    def train(self):
        # Train the model
        X = self.train_dataset.data.numpy()
        X = X.reshape(X.shape[0], -1)
        y = self.train_dataset.labels.numpy()
        self.model.fit(X, y)

    def score_train_manifest(self):
        self.score_dataset(self.train_dataset, 'train_',
                           self.params['visualize train'],
                           self.paths['figures'], self.paths['results'])

    def score_val_manifest(self):
        self.score_dataset(self.val_dataset, 'val_',
                           self.params['visualize val'],
                           self.paths['figures'], self.paths['results'])

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
            X = file['buffers'].to(self.device).numpy()
            X = X.reshape(X.shape[0], -1)
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

        # Compute windowise statistics and write out
        window_report(all_preds, all_labels, self.paths['results'],
                      prefix, '')

        # Score based on sequences
        sequence_report(all_fns, all_preds, all_labels, self.paths['results'],
                        prefix, '')

        # Check for smoothing and run if so
        if self.params['smoothing'] > 0:
            smoothed_preds = smooth(all_preds, self.params['smoothing'])

            if visualize:
                make_images(all_fns, smoothed_preds, all_labels,
                            self.paths['figures'], prefix, '_smoothed')

            # Compute windowise statistics and write out
            window_report(smoothed_preds, all_labels, self.paths['results'],
                          prefix, '_smoothed')

            # Score based on sequences
            sequence_report(all_fns, smoothed_preds, all_labels,
                            self.paths['results'], prefix, '_smoothed')
