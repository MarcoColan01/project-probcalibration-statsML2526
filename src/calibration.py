from abc import ABC, abstractmethod
import numpy as np
import warnings

def _sigmoid(z):
    """
    Computes the logistic function 1 / (1 + exp(-z)) element-wise, written as exp(-log(1 + exp(-z))) with np.logaddexp so that it does not overflow for large values of |z|.
    
    Returns: an array with the same shape as z and values in (0, 1).
    """
    return np.exp(-np.logaddexp(0.0,-z))

class Calibrator(ABC):
    """
    Abstract interface of a post-hoc calibrator. It only declares the two methods that every calibrator must implement: 
    - fit, which learns the calibration map from the uncalibrated predictions of the calibration set,
    - predict_proba, which applies the map to new predictions.
    """
    @abstractmethod
    def fit(self,scores,y):
        """
        Fits the calibration map on the uncalibrated predictions scores of the calibration set and on their binary labels `y` (0 or 1).
     
        Returns: the fitted calibrator.
        """
        pass

    @abstractmethod
    def predict_proba(self,scores):
        """
        Applies the fitted calibration map to new uncalibrated predictions scores.
        
        Returns: an array with the calibrated probability of the positive class of each prediction.
        """
        pass

class PlattScaling(Calibrator):
    """
    Platt scaling. It implements the Calibrator interface with a parametric map, the sigmoid 1 / (1 + exp(A * f + B)) of the uncalibrated prediction f: fit estimates only the two parameters A and B, by minimizing the cross-entropy with Platt's smoothed targets, and predict_proba applies the sigmoid.
    """
    def __init__(self, lr=1.0, tol=1e-6, max_iter=10_000):
        self.lr = lr    # step size of gradient descent
        self.tol = tol  # gradient descent stops when the norm of the gradient falls below tol
        self.max_iter = max_iter    #max number of gradient descent iterations

    def fit(self, scores, y):
        """
        Estimates A and B by minimizing the mean cross-entropy between the sigmoid and the smoothed targets with full-batch gradient descent.
        
        Returns: the fitted calibrator, with A_ and B_ set.
        """
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
        """
        Applies the fitted sigmoid to the uncalibrated predictions scores.
        
        Returns: the calibrated probabilities 1 / (1 + exp(A_ * f + B_)), one for each prediction f.
        """   
        return _sigmoid(-(self.A_*np.asarray(scores,dtype=float)+self.B_))


class IsotonicCalibrator(Calibrator):
    """
    Isotonic regression. It implements the Calibrator interface with a non-parametric map: fit computes the one that minimizes the squared error on the calibration set with the Pair-Adjacent Violators (PAV) algorithm, and predict_proba returns the value of the step that contains each prediction.
    """
    def fit(self, scores, y):
        """
        Runs PAV on the calibration set.
        
        Returns: the fitted calibrator, with thresholds_ and values_ set.
        """
        # distinct predictions in increasing order, with the number of examples and the mean label of each
        unique, inverse, counts = np.unique(np.asarray(scores, dtype=float), return_inverse=True, return_counts=True)
        means = np.bincount(inverse, weights=np.asarray(y,dtype=float))/counts
        #highest prediction, value (mean label) and weight (number of examples) of each block
        highs, values, weights= [], [], []
        for high, value, weight in zip(unique, means, counts):
            #merge the new point with the last blocks as long as their values are not strictly increasing
            while values and values[-1] >= value:
                prev_value, prev_weight = values.pop(), weights.pop()
                value = (prev_value*prev_weight + value*weight)/(prev_weight+weight)
                weight = prev_weight+weight
                highs.pop()
            highs.append(high)
            values.append(value)
            weights.append(weight)

        self.thresholds_ = np.array(highs)  #highest calibration prediction of each block in increasing order
        self.values_ = np.array(values) # calibrated probability of each block
        return self
    

    def predict_proba(self, scores):
        """
        Assigns each prediction f to the block (f_i, f_j] that contains it.
        
        Returns: the calibrated probabilities, one for each prediction.
        """
        idx = np.searchsorted(self.thresholds_, np.asarray(scores, dtype=float), side="left")
        return self.values_[np.minimum(idx, len(self.values_) - 1)]