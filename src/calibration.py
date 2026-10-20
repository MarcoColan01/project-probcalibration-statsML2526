from abc import ABC, abstractmethod
import numpy as np
import warnings

def _sigmoid(z):
    return np.exp(-np.logaddexp(0.0,-z))

class Calibrator(ABC):

    @abstractmethod
    def fit(self,scores,y):
        pass

    @abstractmethod
    def predict_proba(self,scores):
        pass

class PlattScaling(Calibrator):
    def __init__(self, lr=1.0, tol=1e-6, max_iter=10_000):
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, scores, y):
        f, y = np.asarray(scores, dtype=float), np.asarray(y)
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        t = np.where(y == 1, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))

        mu, sigma = f.mean(), f.std()
        z = (f-mu)/sigma
        A,B = 0.0, np.log((n_neg+1) / (n_pos+1))
        self.loss_history_ = []
        for iteration in range(1,self.max_iter+1):
            u = A*z +B
            self.loss_history_.append(np.mean(t*np.logaddexp(0,u) + (1-t)*np.logaddexp(0,-u)))
            residual = t - _sigmoid(-u)
            gradient = np.array([np.mean(residual*z), np.mean(residual)])
            if np.linalg.norm(gradient) < self.tol: break
            A,B = A-self.lr * gradient[0], B-self.lr * gradient[1]     

        self.n_iter_=iteration
        self.converged_ = bool(np.linalg.norm(gradient) < self.tol)
        if not self.converged_:
            warnings.warn(f"Platt scaling did not converge in {self.max_iter} iterations")
        self.A_ = A/sigma
        self.B_ = B-A*mu/sigma
        return self

    def predict_proba(self,scores):         
        return _sigmoid(-(self.A_*np.asarray(scores,dtype=float)+self.B_))


class IsotonicCalibrator(Calibrator):
    def fit(self, scores, y):
        unique, inverse, counts = np.unique(np.asarray(scores, dtype=float), return_inverse=True, return_counts=True)
        means = np.bincount(inverse, weights=np.asarray(y,dtype=float))/counts
        highs, values, weights= [], [], []
        for high, value, weight in zip(unique, means, counts):
            while values and values[-1] >= value:
                prev_value, prev_weight = values.pop(), weights.pop()
                value = (prev_value*prev_weight + value*weight)/(prev_weight+weight)
                weight = prev_weight+weight
                highs.pop()
            highs.append(high)
            values.append(value)
            weights.append(weight)

        self.thresholds_ = np.array(highs)  
        self.values_ = np.array(values)
        return self
    

    def predict_proba(self, scores):
        idx = np.searchsorted(self.thresholds_, np.asarray(scores, dtype=float), side="left")
        return self.values_[np.minimum(idx, len(self.values_) - 1)]