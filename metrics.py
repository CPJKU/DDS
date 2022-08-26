import numpy as np
import sys


def thresh(values, threshold, retain):
    if retain:
        values = np.copy(values)
    values[values<threshold] = 0
    values[values>=threshold] = 1
    return values


def calc_stats(predictions, labels, thr):
    p = thresh(predictions, thr, True).astype(bool)
    l = thresh(labels, thr, True).astype(bool)

    P = np.sum(l)
    N = np.size(l) - P
    TP = np.sum(p * l)
    TN = np.sum((~p) * (~l))
    FP = np.sum((p ^ l) * p)
    FN = np.sum((p ^ l) * l)

    return P, N, TP, TN, FP, FN


def calc_metrics(predictions, labels, thr=0.5, stats=None):
    '''Calculates and returns dicitonary of most relevant
    MIR metrics.
    '''

    if stats:
        P, N, TP, TN, FP, FN = stats
    else:
        P, N, TP, TN, FP, FN = calc_stats(predictions,
                                          labels,
                                          thr)

    c = sys.float_info.min # prevent zero division

    metrics = {}
    # Positive Predictive Value - Precision
    metrics['P'] = (TP) / (TP + FP + c)
    # True Positive Rate - Recall
    metrics['R'] = (TP) / (P + c)

    # F1-Score
    metrics['F'] = ((metrics['P'] * metrics['R']) / (metrics['P'] + metrics['R'] + c)) * 2
    # Frame-level Accuracy as proposed by Dixon [2000]
    metrics['A'] = (TP) / (FP + FN + TP + c)

    return metrics


def get_optimal_threshold(activations, labels, n_vals=100, metric='F', incl_metric=False, verbose=False):
    import operator

    thres_metric = {}

    for k in range(1, n_vals):
        thr = k / n_vals
        metrics = calc_metrics(activations, labels, thr)
        thres_metric[thr] = metrics[metric]

    optimal = max(thres_metric.items(), key=operator.itemgetter(1))

    if verbose:
        print(f'Optimal threshold is {optimal[0]} with metric {metric} = {optimal[1]:.5f}')

    return optimal if incl_metric else optimal[0]

def calc_aucs(activations, labels):
    '''Calculates AUC (Area Under the Curve) metrics (AUCs) for both PR and ROC curves.'''

    from sklearn.metrics import precision_recall_curve, roc_curve, auc

    precision, recall, _ = precision_recall_curve(labels.flatten(), activations.flatten())
    auc_pr = auc(recall, precision)

    fpr, tpr, _ = roc_curve(labels.flatten(), activations.flatten())
    auc_roc = auc(fpr, tpr)

    return auc_pr, auc_roc
