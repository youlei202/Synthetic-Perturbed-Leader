from river import base, optim
import numpy as np

class CustomFunc(base.Regressor):

    def __init__(
        self,
        optimizer,
        a=1, b=1, c=0.5, d=1,
        w=1,
    ):
        self.optimizer = optimizer
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self._weights = {'x': w}

    def learn_one(self, x: dict, y=None):
        """Fits to a set of features `x` and a real-valued target `y`.

        Parameters
        ----------
        x
            A dictionary of features.
        y
            A numeric target.

        """
        g = dict(zip(
            self._weights.keys(), 
            [self._f_gradient(w, x['t']) for _,w in self._weights.items()]
            ))
        
        self.optimizer.step(w=self._weights, g=g)


    def predict_one(self, x: dict):
        x_value = x['x']
        t_value = x['t']
        return self._f(x=x_value, t=t_value)

    def _f(self, x, t):
        # Main sinusoidal component, predictable for all x
        main_component = self.b * np.sin(self.c * t)
        
        # Additional component that makes the function hard to predict for some x values
        # The frequency of this component is an exponential function of x
        unpredictable_component = self.a * np.sin(np.exp(self.d * x) * t) if x > 0 else 0
        
        return main_component + unpredictable_component

    def _f_gradient(self, x, t):
        # The gradient is only non-zero for the unpredictable component when x > 0
        if x > 0:
            return self.a * t * np.exp(self.d * x) * np.cos(np.exp(self.d * x) * t) * self.d
        else:
            # For x <= 0, the function is a simple sinusoid in t, and the gradient wrt x is 0
            return 0
