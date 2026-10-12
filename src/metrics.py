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

def evaluate(y,p,eps=EPS):
    p = np.asarray(p,dtype=float)
    return{
        "accuracy": accuracy(y,p),
        "log_loss": log_loss(y,p,eps),
        "brier": brier_score(y,p),
        "clipped": int(np.sum((p < eps) | (p > 1 - eps))),
    }

