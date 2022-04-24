import sys

# README
README = """
---------------------------- Help ---------------------------
Chaoqi Yang @ UIUC, chaoqiy2@illinois.edu
-------------------------------------------------------------
GOCPT is a python open-source package for handing generalized 
online tensor decomposition and completion.

We provide the following tensor functions:

    - for generalized online tensor factorization (GOTF)
        - GOCPT.gotf.GOCPT
        - GOCPT.gotf.GOCPTE
    
    - for generalized online tensor completion (GOTC)
        - GOCPT.gotc.GOCPT
        - GOCPT.gotc.GOCPTE
    
    - for common online tensor factorization (OTF)
        - arbitrary order tensor
            - GOCPT.otf.OnlineCPD
            - GOCPT.otf.MAST
            - GOCPT.otf.CPStream
        - 3-order tensors and 1 slices at each step only
            - GOCPT.otf.SDT
            - GOCPT.otf.RLST

    - for common online tensor completion (OTC)
        - GOCPT.otc.OnlineSGD
        - GOCPT.otc.OLSTEC

EXAMPLE:
    >> from GOCPT import gotf, datasets, simulation
    >> X = datasets.GCSS()
    >> [X_0, X_inc_ls] = simulation.growth_1d(X, prep=0.3, inc=3)
    >> model = gotf.GOCPTE(X_0, R=5)
    >> for increments in X_inc_ls:
    ...     model.update(increments) 
"""

def GOCPT():
    print (README)