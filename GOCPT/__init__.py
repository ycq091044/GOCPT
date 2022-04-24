import sys

# README
README = """
---------------------------- Help ---------------------------
GOCPT is a python open-source package for handing generalized 
online tensor decomposition and completion.

We provide the following functions:

    - for online tensor factorization of arbitraty tensors
        - GOCPT.otf.OnlineCPD
        - GOCPT.otf.MAST
        - GOCPT.otf.CPStream
        - GOCPT.otf.GOCPTE

    - for online tensor factorization of 3-order tensors
        - GOCPT.otf.SDT
        - GOCPT.otf.RLST

    - for online tensor completion of arbitraty tensors
        - GOCPT.otc.OnlineSGD
        - GOCPT.otc.OLSTEC
        - GOCPT.otc.GOCPTE

EXAMPLE:
    >> from GOCPT import otf, datasets
    >> X = datasets.GCSS()
    >> model = otf.GOCPTE(base_X=X, R=5, iters=50)
    >> new_slice = X[...,-2:-1]
    >> model.update(new_slice, verbose=True) 
"""

def GOCPT():
    print (README)