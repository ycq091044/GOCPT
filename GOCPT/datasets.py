import pickle
import numpy as np
from .utils import rec_from_factors
from urllib import request

def FACE_3D():
    face_3d = pickle.load(request.urlopen('https://drive.google.com/uc?id=1GGXsG7Pqr0Bt6PDOUDE_gkv81ou-hJHT'))
    return face_3d

def GCSS():
    gcss = pickle.load(request.urlopen('https://drive.google.com/uc?id=1X9-vbZUMm_VhsUgs8uKgsCQ5MbB7QYuX'))
    return gcss

def JHU_COVID():
    jhu_covid = pickle.load(request.urlopen('https://drive.google.com/uc?id=1noy2iDLGB1xF8NNp4XcsMn6x4KXbjM_9'))
    return jhu_covid

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