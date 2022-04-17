import sys
sys.path.append('/home/chaoqiy2/github/GOCPT')
from GOCPT import datasets, otf, metrics
import numpy as np

# load datasets
def test_load_datasets():
    FACE_3D = datasets.face3d()
    GCSS = datasets.GCSS()
    Indian_Pines = datasets.indian_pines()

    print ()
    print ('----- data tensor shape -----')
    print ('face 3d dataset: {}'.format(FACE_3D.shape))
    print ('GCSS dataset: {}'.format(GCSS.shape))
    print ('Indian Pines dataset: {}'.format(Indian_Pines.shape))

# test online tensor factorization
def test_online_tensor_factorization():
    # synthesize a tensor
    X = datasets.face3d()
    # obtain the factors
    factors = otf.CP_ALS(X, 5, 100, verbose=True)
    # calculate the metric
    print (metrics.PoF(X, factors))

if __name__ == '__main__':    
    # test load datasets
    # test_load_datasets()

    # test online tensor decomposition
    test_online_tensor_factorization()    