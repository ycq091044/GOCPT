import pickle
import os
import scipy.io as IO
        
root = './GOCPT/datasets/' 

def face3d():
    face_3d_dataset = pickle.load(open(root + 'FACE-3D.pkl', 'rb'))
    return face_3d_dataset

def GCSS():
    gcss_dataset = pickle.load(open(root + 'GCSS.pkl', 'rb'))
    return gcss_dataset

def indian_pines():
    indian_pines_dataset = IO.loadmat(root + 'Indian_pines_corrected.mat')
    return indian_pines_dataset['indian_pines_corrected']