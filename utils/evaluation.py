import os
import csv
import torch
import numpy as np
from sklearn import metrics
from collections import OrderedDict


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


def compute_metrics(labels, preds, threshold=0.5):
    """Compute the statistics for a set of labels and predictions"""
    # Compute stats
    y_hat = np.zeros(preds.shape[0])
    y_hat[np.where(preds[:, 1] >= threshold)] = 1
    tp = float(np.sum((y_hat == 1) * (labels == 1)))
    fp = float(np.sum((y_hat == 1) * (labels == 0)))
    tn = float(np.sum((y_hat == 0) * (labels == 0)))
    fn = float(np.sum((y_hat == 0) * (labels == 1)))

    # Compute metrics
    stats = OrderedDict()
    stats['acc'] = float((tp + tn) / (tp + fp + tn + fn))
    if tp + fn > 0:
        stats['sens'] = float(tp / (tp + fn))
    else:
        stats['sens'] = 0.0
    if tp + fp > 0:
        stats['prec'] = float(tp / (tp + fp))
    else:
        stats['prec'] = 0.0
    if tp > 0:
        stats['f1'] = 2 * (stats['prec'] * stats['sens']
                           / (stats['prec'] + stats['sens']))
    else:
        stats['f1'] = 0.0
    stats['spec'] = float(tn / (tn + fp))
    if np.sum(labels) > 0:
        # Compute the AUC-ROC
        fpr, tpr, thresholds = metrics.roc_curve(labels, preds[:, 1])
        stats['roc curve'] = {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }
        stats['auc-roc'] = metrics.auc(fpr, tpr)

        # Compute AUC-PR
        precision, recall, thresholds = metrics.precision_recall_curve(
            labels, preds[:, 1])
        stats['pr curve'] = {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds
        }
        stats['auc-pr'] = metrics.auc(recall, precision)
    else:
        stats['auc-roc'] = 0.0
        stats['auc-pr'] = 0.0

    return stats


def score_recording(labels, preds, threshold=0.5):
    """Score a single recording"""
    # Find the true onsets and offsets
    true_onsets = np.where(np.diff(labels) == 1)[0] + 1
    true_offsets = np.where(np.diff(labels) == -1)[0] + 1

    # Find the true positives
    y_hat = np.zeros(preds.shape[0])
    y_hat[np.where(preds[:, 1] >= threshold)] = 1
    tp_samples = y_hat * labels
    fp_samples = y_hat * (1 - labels)

    # Find beginning of seizure detections
    detections = np.where(np.diff(y_hat) == 1)[0] + 1

    # Loop over the seizures
    ncorrect = 0
    latency_samples = 0
    for onset, offset in zip(true_onsets, true_offsets):

        # Check for an accurate sample
        if np.sum(tp_samples[onset:offset]) > 0:
            ncorrect += 1

            # Find the onset
            if tp_samples[onset] == 1:
                # Detection occurs before onset annotation
                idx = onset
                while y_hat[idx] == 1 and idx >= 0:
                    tp_samples[idx] = 1
                    fp_samples[idx] = 0
                    idx = idx - 1
                first_tp = idx + 1
                latency_samples += first_tp - onset
            else:
                # Detection is after onset
                latency_samples += np.where(
                    tp_samples[onset:offset] == 1)[0][0]
    nfps = np.sum(fp_samples[detections])

    return {
        'nfps': nfps,
        'latency_samples': latency_samples,
        'ncorrect': ncorrect
    }


def iid_window_report(all_preds, all_labels, report_folder, prefix, suffix):
    # Compute window based statistics and write out
    stats = compute_metrics(np.concatenate(all_labels),
                            np.concatenate(all_preds))
    stats_fn = os.path.join(report_folder,
                            '{}stats{}.pkl'.format(prefix, suffix))
    pr_fn = os.path.join(report_folder,
                         '{}pr{}.pkl'.format(prefix, suffix))
    roc_fn = os.path.join(report_folder,
                          '{}roc{}.pkl'.format(prefix, suffix))
    results_fn = os.path.join(report_folder,
                              '{}iid_results{}.csv'.format(prefix, suffix))

    # Save results
    torch.save(stats, stats_fn)
    torch.save(stats['pr curve'], pr_fn)
    torch.save(stats['roc curve'], roc_fn)

    # Write out report
    with open(results_fn, 'w', newline='') as csvfile:
        fieldnames = ['acc', 'sens', 'spec', 'prec', 'f1', 'auc-roc',
                      'auc-pr']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({k: stats[k] for k in fieldnames})

    frmt = "{:<8}"*len(fieldnames)
    print(frmt.format(*fieldnames))
    frmt = "{:<8.4f}"*len(fieldnames)
    print(frmt.format(*[stats[ff] for ff in fieldnames]))


def sequence_report(all_fns, all_preds, all_labels, report_folder, prefix,
                    suffix):
    # Score based on sequences
    total_fps = 0
    total_latency_samples = 0
    total_correct = 0
    all_results = []
    for fn, pred, label in zip(all_fns, all_preds, all_labels):
        stats = score_recording(label, pred)
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


def threshold_sweep(all_preds, all_labels, report_folder,
                    prefix="", suffix="", nsz=0, total_duration=0,
                    window_advance_seconds=0):
    """For a set of predictions, sweep the threshold compute results

    Args:
        all_preds (list): list of predictions
        all_labels (list): list of labels
    """

    # Initialize values
    thresholds = np.arange(0, 1, 0.01)
    results = {
        'thresholds': thresholds,
        'nfps': np.zeros_like(thresholds),
        'latency_samples': np.zeros_like(thresholds),
        'ncorrect': np.zeros_like(thresholds)
    }

    # Loop over lists of preds and labels
    for pred, labels in zip(all_preds, all_labels):
        # Score the recording
        for ii, thresh in enumerate(thresholds):
            stats = score_recording(labels, pred, threshold=thresh)
            results['nfps'][ii] += stats['nfps']
            results['latency_samples'][ii] += stats['latency_samples']
            results['ncorrect'][ii] += stats['ncorrect']

    # If needed stats are provided, compute sensitivity, fp/hr, latency_seconds
    if nsz > 0:
        results['sensitivity'] = results['ncorrect'] / nsz
        results['average_latency_samples'] = results['latency_samples'] / nsz
    if total_duration > 0:
        results['fps_per_hour'] = results['nfps'] * 3600 / total_duration
    if window_advance_seconds > 0:
        results['latency_seconds'] = (
            results['latency_samples'] * window_advance_seconds
        )
        if nsz > 0:
            results['average_latency_seconds'] = results['latency_seconds'] / nsz

    # Save the results
    fn = '{}threshold_sweep{}.pkl'.format(prefix, suffix)
    torch.save(results, os.path.join(report_folder, fn))
