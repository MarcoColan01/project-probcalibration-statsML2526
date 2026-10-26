import numpy as np
EPS = 1e-15

def log_loss(y,p,eps=EPS):
    """
    Computes the log-loss (binary cross-entropy) of the probabilities p for the labels y, after clipping p to [eps, 1 - eps] so that predictions of exactly 0 or 1 give a finite value.
    
    Returns: the mean log-loss over the examples.
    """
    y = np.asarray(y)
    p = np.clip(np.asarray(p, dtype=float),eps, 1-eps)
    return -np.mean(y*np.log(p) + (1-y)*np.log(1-p))

def accuracy(y,p,threshold=0.5):
    """
    Predicts the positive class when p >= threshold and compares the predicted labels with y.
    
    Returns: the fraction of correctly classified examples.
    """
    return np.mean((np.asarray(p) >= threshold) == np.asarray(y))

def brier_score(y,p):
    """
    Computes the squared difference between each probability p and its label y.
    
    Returns: the mean squared difference, which is the the Brier score.
    """
    return np.mean((np.asarray(p, dtype=float) - np.asarray(y)) ** 2)

def evaluate(y,p):
    """
    Computes all the metrics of the report for the probabilities p and the labels y.
    
    Returns: a dictionary with the accuracy, log-loss, Brier score and ECE (keys accuracy, log_loss, brier, ece).
    """
    p = np.asarray(p,dtype=float)
    return{
        "accuracy": accuracy(y,p),
        "log_loss": log_loss(y,p),
        "brier": brier_score(y,p),
        "ece": expected_calibration_error(y,p),
    }

def bin_index(p, n_bins=10):
    """
    Assigns each probability p to one of n_bins equal-width bins of [0, 1], with index floor(p * n_bins); p = 1 is put in the last bin.
    
    Returns: the integer bin index of each probability.
    """
    return np.minimum((np.asarray(p, dtype=float) * n_bins).astype(int), n_bins - 1)

def reliability_bins(y,p, n_bins=10):
    """
    Divides the predictions into n_bins equal-width bins and computes, for each non-empty bin, the mean predicted probability and the fraction of positive examples.
    
    Returns: three arrays with the mean prediction, the fraction of positives and the number of predictions of each non-empty bin.
    """
    y,p = np.asarray(y), np.asarray(p, dtype=float)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    counts =np.bincount(idx, minlength=n_bins)
    keep = counts > 0
    mean_pred = np.bincount(idx, weights=p, minlength=n_bins)[keep] / counts[keep]
    frac_pos = np.bincount(idx, weights=y, minlength=n_bins)[keep]/counts[keep]

    return mean_pred, frac_pos, counts[keep]


def expected_calibration_error(y, p, n_bins=10):
    """
    Computes the ECE, the absolute difference between the fraction of positives and the mean prediction of each bin, averaged over the bins with weights proportional to their number of predictions.
    
    Returns: the ECE as a float.
    """
    mean_pred, frac_pos, counts = reliability_bins(y, p, n_bins)
    return np.sum(counts * np.abs(frac_pos - mean_pred)) / np.sum(counts)