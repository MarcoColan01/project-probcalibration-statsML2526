import numpy as np
EPS = 1e-15

def log_loss(y,p,eps=EPS):
    y = np.asarray(y)
    p = np.clip(np.asarray(p, dtype=float),eps, 1-eps)
    return -np.mean(y*np.log(p) + (1-y)*np.log(1-p))

def accuracy(y,p,threshold=0.5):
    return np.mean((np.asarray(p) >= threshold) == np.asarray(y))

def brier_score(y,p):
    return np.mean((np.asarray(p, dtype=float) - np.asarray(y)) ** 2)

def evaluate(y,p):
    p = np.asarray(p,dtype=float)
    return{
        "accuracy": accuracy(y,p),
        "log_loss": log_loss(y,p),
        "brier": brier_score(y,p),
        "ece": expected_calibration_error(y,p),
    }

def bin_index(p, n_bins=10):
    return np.minimum((np.asarray(p, dtype=float) * n_bins).astype(int), n_bins - 1)

def reliability_bins(y,p, n_bins=10):
    y,p = np.asarray(y), np.asarray(p, dtype=float)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    counts =np.bincount(idx, minlength=n_bins)
    keep = counts > 0
    mean_pred = np.bincount(idx, weights=p, minlength=n_bins)[keep] / counts[keep]
    frac_pos = np.bincount(idx, weights=y, minlength=n_bins)[keep]/counts[keep]

    return mean_pred, frac_pos, counts[keep]


def expected_calibration_error(y, p, n_bins=10):
    mean_pred, frac_pos, counts = reliability_bins(y, p, n_bins)
    return np.sum(counts * np.abs(frac_pos - mean_pred)) / np.sum(counts)