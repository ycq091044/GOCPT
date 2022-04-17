import numpy as np
from scipy import linalg as la
from .utils import rec

def PoF(X, factors):
    """
    The percentage of fitness (POF) metric
    INPUT:
        - <tensor> X: the original tensor
        - <tensor> rec: the reconstructed tensor
    OUTPUT:
        - <scalar> PoF_metric: the metric, higher is better 
    """
    Rec = rec(factors)
    PoF_metric = 1 - la.norm(Rec - X) / la.norm(X)
    return PoF_metric