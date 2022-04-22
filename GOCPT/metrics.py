import numpy as np
from numpy import linalg as la
from .utils import rec_from_factors

def PoF(X, factors, mask=None):
    """
    The percentage of fitness (POF) metric
    INPUT:
        - <tensor> X: the original tensor
        - <tensor> rec: the reconstructed tensor
    OUTPUT:
        - <scalar> PoF_metric: the metric, higher is better 
    """
    if mask is None:
        Rec = rec_from_factors(factors)
        PoF_metric = 1 - la.norm(Rec - X) / la.norm(X)
    else:
        Rec = rec_from_factors(factors)
        PoF_metric = 1 - la.norm(mask * Rec - X) / la.norm(X)
    return PoF_metric