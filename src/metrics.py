import numpy as np

EPS = 1e-15

def log_loss(y,p,eps=EPS):
    y = np.asarray(y)
    p = np.clip(np.asarray(p, dtype=float),eps, 1-eps)
    return -np.mean(y*np.log(p) + (1-y)*np.log(1-p))

