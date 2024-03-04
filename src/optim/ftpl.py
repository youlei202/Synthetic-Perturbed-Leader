from __future__ import annotations

import collections
import numpy as np
from river import optim

class FTPL(optim.base.Optimizer):
    """
    Follow the Perturbed Leader (FTPL) optimizer.

    This optimizer is based on the idea of making decisions based on the perturbation of the cumulative losses.

    Parameters
    ----------
    eta : float
        The learning rate.
    gamma : float
        The perturbation parameter.

    Attributes
    ----------
    cumulative_loss : collections.defaultdict
        Cumulative loss for each feature.
    perturbation : collections.defaultdict
        Random perturbation for each feature.

    References
    ----------
    [Reference to the theoretical background of FTPL]
    """

    def __init__(self, eta=1e3, gamma=0.1):
        self.eta = eta
        self.gamma = gamma
        self.cumulative_loss = collections.defaultdict(float)
        self.perturbation = collections.defaultdict(lambda: np.random.normal())
        self.n_iterations = 0

    def _step_with_dict(self, w, g):
        eta = self.eta
        gamma = self.gamma

        t = self.n_iterations + 1
        # gt_est = self._f_gradient(w['w'], t)
        gt_est = 0

        # Update the cumulative loss and compute the decision based on perturbation
        for i, gi in g.items():
            self.cumulative_loss[i] += (gi+gt_est)
            self.perturbation[i] = np.random.normal(scale=gamma)
            w[i] = -self.cumulative_loss[i] / eta + self.perturbation[i]

        return w

    # def _f_gradient(self, x, t):
    #     # The gradient is only non-zero for the unpredictable component when x > 0
    #     a = 1
    #     d = 1
    #     if x > 0:
    #         return a * t * np.exp(d * x) * np.cos(np.exp(d * x) * t) * d
    #     else:
    #         # For x <= 0, the function is a simple sinusoid in t, and the gradient wrt x is 0
    #         return 2 * x
        
