import numpy as np
from sklearn import metrics
from collections import OrderedDict


def compute_metrics(labels, preds):
    """Compute the statistics for a set of labels and predictions"""
    # Compute stats
    y_hat = np.argmax(preds, axis=1)
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
        stats['sens'] = 'NA'
    if tp + fp > 0:
        stats['prec'] = float(tp / (tp + fp))
    else:
        stats['prec'] = 'NA'
    if tp > 0:
        stats['f1'] = 2 * ( stats['prec'] * stats['sens']
                         / (stats['prec'] + stats['sens']) )
    else:
        stats['f1'] = 'NA'
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
        stats['auc-roc'] = 'NA'
        stats['auc-pr'] = 'NA'

    return stats


def score_recording(labels, preds):
    """Score a single recording"""
    # Find the true onsets and offsets
    true_onsets = np.where(np.diff(labels) == 1)[0] + 1
    true_offsets = np.where(np.diff(labels) == -1)[0] + 1
    
    # Find the true positives
    y_hat = np.argmax(preds, axis=1)
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
                latency_samples += np.where(tp_samples[onset:offset] == 1)[0][0]
    nfps = np.sum(fp_samples[detections])
    
    return {
        'nfps': nfps,
        'latency_samples': latency_samples,
        'ncorrect': ncorrect
    }
