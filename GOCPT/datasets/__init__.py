import pickle
import os
import numpy as np
from ..utils import rec_from_factors
root = '../GOCPT/datasets/' 

def face3d():
    face_3d_dataset = pickle.load(open(root + 'FACE-3D.pkl', 'rb'))
    return face_3d_dataset

def GCSS():
    gcss_dataset = pickle.load(open(root + 'GCSS.pkl', 'rb'))
    return gcss_dataset

def syn_data(R, size, dist='unif', with_mask=None):
    In = size
    if 'unif' in dist:
        factors = [np.random.random((Ii, R)) for Ii in In]
    elif 'normal' in dist:
        factors = [np.random.randn(Ii, R) for Ii in In]
    
    syn_tensor = rec_from_factors(factors)

    if with_mask is None:
        return syn_tensor
    else:
        mask = np.random.random(syn_tensor.shape) >= with_mask
        return [syn_tensor * mask, mask]