import collections
import numpy as np
from river import optim


class FTRL(optim.base.Optimizer):
    """
    Follow the Regularized Leader (FTRL) optimizer.

    This optimizer is based on the idea of combining the advantages of online learning algorithms with regularization.

    Parameters
    ----------
    alpha : float
        Learning rate.
    beta : float
        Smoothing parameter to avoid division by zero.
    l1 : float
        L1 regularization term.
    l2 : float
        L2 regularization term.

    Attributes
    ----------
    z : collections.defaultdict
        Intermediate variable storing the weighted sum of past gradients.
    n : collections.defaultdict
        Intermediate variable storing the sum of squares of past gradients.

    """

    def __init__(self, alpha=0.05, beta=1.0, l1=0.0, l2=1.0):
        self.alpha = alpha
        self.beta = beta
        self.l1 = l1
        self.l2 = l2
        self.z = collections.defaultdict(float)
        self.n = collections.defaultdict(float)
        self.n_iterations = 0


    def _step_with_dict(self, w, g):
        alpha = self.alpha
        beta = self.beta
        l1 = self.l1
        l2 = self.l2
        z = self.z
        n = self.n

        for i, gi in g.items():
            sigma = (np.sqrt(n[i] + gi ** 2) - np.sqrt(n[i])) / alpha
            z[i] += gi - sigma * w.get(i, 0)
            n[i] += gi ** 2

            if abs(z[i]) <= l1:
                w[i] = 0
            else:
                w[i] = -1 / ((beta + np.sqrt(n[i])) / alpha + l2) * (z[i] - np.sign(z[i]) * l1)

        return w
    
