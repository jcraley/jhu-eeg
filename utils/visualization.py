import os
import utils.output_tools as out
import matplotlib
import numpy as np
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def make_images(fns, preds, labels, viz_folder, prefix, suffix, threshold=None):
    for fn, pred, label in zip(fns, preds, labels):
        pic_fn = os.path.join(viz_folder,
                              '{}{}{}.png'.format(prefix, fn, suffix))
        plot_yhat(pred, label, fn=pic_fn, threshold=threshold)


def plot_yhat(y_hat, label, fn=None, threshold=None):
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(y_hat[:, 1], label='Prediction')
    ax.plot(label, label='Ground Truth')
    if threshold:
        plt.axhline(threshold, 0, 1, linestyle='--',
                    color='k', label='Threshold')
        classifications = np.asarray(y_hat[:, 1] >= threshold, dtype=int)
        plt.fill_between(np.arange(len(classifications)),
                         y_hat[:, 1] * classifications, 0,
                         color='blue',       # The outline color
                         alpha=0.2)

    ax.set_xlabel('Time (s)', fontsize=16)
    ax.set_ylabel('Model Output', fontsize=16)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    # ax.legend(ncol=3, loc='upper right')
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim([0, len(y_hat[:, 1])])
    if fn:
        plt.tight_layout()
        plt.savefig(fn)
        plt.close()

    else:
        plt.show()


def visualize_history(history, fn):
    # nepochs = len()

    # Four axes, returned as a 2-d array
    f, axarr = plt.subplots(2, 2)
    axarr[0, 0].set_title('Loss')
    axarr[0, 0].plot(history['train']['loss'], color='b')
    axarr[0, 0].plot(history['val']['loss'], color='m')

    axarr[0, 1].set_title('Accuracy')
    axarr[0, 1].plot(history['train']['acc'], color='b')
    axarr[0, 1].plot(history['val']['acc'], color='m')

    axarr[1, 0].set_title('Sensitivity')
    axarr[1, 0].plot(history['train']['sens'], color='b')
    axarr[1, 0].plot(history['val']['sens'], color='m')

    axarr[1, 1].set_title('Specificity')
    axarr[1, 1].plot(history['train']['spec'], color='b', label='Train')
    axarr[1, 1].plot(history['val']['spec'], color='m', label='Test')
    axarr[1, 1].legend()

    plt.savefig(fn)


def plot_curves(xs, ys, labels, xlabel, ylabel, fn=None,
                labelsize=16, legendsize=16, legend=True):
    """Plot curves"""
    for x, y, label in zip(xs, ys, labels):
        plt.plot(x, y, label=label)
    plt.xlabel(xlabel, fontsize=labelsize)
    plt.ylabel(ylabel, fontsize=labelsize)
    if legend:
        plt.legend(fontsize=legendsize)
    if fn:
        plt.savefig(fn)
        plt.close()
    else:
        plt.show()
