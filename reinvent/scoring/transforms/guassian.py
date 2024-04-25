"""Gaussian normalization transform"""

__all__ = ["GaussianNormalization"]

from dataclasses import dataclass
from scipy.stats import norm
import numpy as np

from .transform import Transform


@dataclass
class Parameters:
    type: str
    center: float  # the center value for the Gaussian distribution
    std_dev: float  # the standard deviation of the Gaussian distribution


class GaussianNormalization(Transform, param_cls=Parameters):
    def __init__(self, params: Parameters):
        super().__init__(params)
        self.center = params.center
        self.std_dev = params.std_dev
        
        # Precompute the normal distribution
        self.distribution = norm(loc=self.center, scale=self.std_dev)

    def __call__(self, values) -> np.ndarray:
        transformed = self.distribution.pdf(values)  # Computes the probability density
        max_pdf = self.distribution.pdf(self.center)  # The maximum probability density at the center value
        
        # Normalize the values such that the center value becomes 1.0
        normalized = transformed / max_pdf
        return np.clip(normalized, 0, 1)  # Ensure that the values are clipped between 0 and 1