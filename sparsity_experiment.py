import numpy as np
from scipy import linalg as la
import time
import pickle

"""
For Sparsity Experiments
"""
# configuration
I, J, K, R, Iter = 50, 50, 50, 5, 50


def optimize(A, B):
    """
    Least Squares Solver: AX = B
    """
    L = la.cholesky(A + np.eye(A.shape[1]) * 1e-8)
    y = la.solve_triangular(L.T, B, lower=True)
    u = la.solve_triangular(L, y, lower=False)
    return u

def iterationImputed(T, mask, A1, A2, A3, reg=1e-5):
    T = mask * T + (1-mask) * np.einsum('ir,jr,kr->ijk',A1,A2,A3,optimize=True)
    eye = np.eye(A1.shape[1])

    A1 = optimize(np.einsum('ir,im,jr,jm->rm',A2,A2,A3,A3,optimize=True) + reg * eye, np.einsum('ijk,jr,kr->ri',T,A2,A3,optimize=True)).T
    A2 = optimize(np.einsum('ir,im,jr,jm->rm',A1,A1,A3,A3,optimize=True) + reg * eye, np.einsum('ijk,ir,kr->rj',T,A1,A3,optimize=True)).T
    A3 = optimize(np.einsum('ir,im,jr,jm->rm',A1,A1,A2,A2,optimize=True) + reg * eye, np.einsum('ijk,ir,jr->rk',T,A1,A2,optimize=True)).T

    return A1, A2, A3

def imputed(X, mask, Iter):
    # initialization

    A = np.random.random((I, R))
    B = np.random.random((J, R))
    C = np.random.random((K, R))

    tic_start = time.time()
    tic = time.time()
    result_imputed, time_imputed = [], []
    for _ in range(Iter):
        A, B, C = iterationImputed(X, mask, A, B, C)
        toc = time.time()
        _, loss, PoF = metric(A, B, C, X, mask); result_imputed.append(PoF); time_imputed.append(time.time() - tic_start)
        # print ('loss:{}, PoF:{}, time:{}'.format(loss, PoF, toc-tic))
        tic = time.time()
    print ('finish imputed')
    return result_imputed, time_imputed

def Optimizer(Omega, A, RHS, num, reg):
    """
    masked least square optimizer:
        A @ u.T = Omega * RHS
        number: which factor
        reg: 2-norm regulizer
    """
    N = len(A)
    R = A[0].shape[1]
    lst_mat = []
    T_inds = "".join([chr(ord('a')+i) for i in range(Omega.ndim)])
    einstr=""
    for i in range(N):
        if i != num:
            einstr+=chr(ord('a')+i) + 'r' + ','
            lst_mat.append(A[i])
            einstr+=chr(ord('a')+i) + 'z' + ','
            lst_mat.append(A[i])
    einstr+= T_inds + "->"+chr(ord('a')+num)+'rz'
    lst_mat.append(Omega)
    P = np.einsum(einstr,*lst_mat,optimize=True)
    o = np.zeros(RHS.shape)
    for j in range(A[num].shape[0]):
        o[j,:] = np.linalg.inv(P[j]+reg*np.eye(R)) @ RHS[j,:]
    return o

def iterationMasked(A1, A2, A3, mask, T, reg=1e-5):
    """
    The preparation routine (before streaming and CPC setting)
    """
    T = T * mask

    A1 = Optimizer(mask, [A1, A2, A3], np.einsum('ijk,jr,kr->ir',T,A2,A3,optimize=True), 0, reg)
    A2 = Optimizer(mask, [A1, A2, A3], np.einsum('ijk,ir,kr->jr',T,A1,A3,optimize=True), 1, reg)
    A3 = Optimizer(mask, [A1, A2, A3], np.einsum('ijk,ir,jr->kr',T,A1,A2,optimize=True), 2, reg)
    return A1, A2, A3

def masked(X, mask, Iter):
    # initialization

    A = np.random.random((I, R))
    B = np.random.random((J, R))
    C = np.random.random((K, R))

    tic_start = time.time()
    tic = time.time()
    result_masked, time_masked = [], []
    for _ in range(Iter):
        A, B, C = iterationMasked(A, B, C, mask, X)
        toc = time.time()
        _, loss, PoF = metric(A, B, C, X, mask); result_masked.append(PoF); time_masked.append(time.time() - tic_start)
        # print ('loss:{}, PoF:{}, time:{}'.format(loss, PoF, toc-tic))
        tic = time.time()
    print ('finish masked')
    return result_masked, time_masked


def metric(A, B, C, X, mask=None):
    rec = np.einsum('ir,jr,kr->ijk',A,B,C)
    if mask is not None:
        loss = la.norm(mask * (rec - X)) ** 2
    else:
        loss = la.norm(rec - X) ** 2
    PoF = 1 - la.norm(rec - X) / la.norm(X) 
    return rec, loss, PoF


if __name__ == '__main__':
    
    """
    Data Generation
    """

    A0 = np.random.random((I, R))
    B0 = np.random.random((J, R))
    C0 = np.random.random((K, R))

    X = np.einsum('ir,jr,kr->ijk',A0,B0,C0)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(20,7))
    plt.rcParams.update({"font.size":12})

    sparse_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    for index, sparsity in enumerate(sparse_list):
        plt.subplot(2,len(sparse_list)//2,index+1)
        mask = np.random.random(X.shape) >= sparsity
        result_masked, time_masked = masked(X, mask, Iter)
        result_imputed, time_imputed = imputed(X, mask, Iter)
        plt.plot(time_masked, result_masked, label="Masked CPD")
        plt.plot(time_imputed, result_imputed, label="Imputed CPD")
        plt.legend()
        plt.ylabel('PoF')
        plt.xlabel('Runing Time (s)')
        plt.title('Sparsity: {}%'.format(sparsity*100))
    plt.tight_layout()
    plt.show()
