from typing import Awaitable
import numpy as np
from scipy import linalg as la
import time


"""
For Streaming Tensor Completion
    - tensor size changes along the time mode
    - tensor is always incomplete
    - the previous slides will not be updated
    - the new slides are also incomplete
Methods
    - Oracle CPD on the overall complete tensor
    - step by step EM imputation + CP decomposition
        - |mask * (x - Adiag(c)B.T)| + alpha * |mask * (X - [A, B, C_o])| + beta * reg
    - row-wise least squares method
        - |mask * (x - Adiag(c)B.T)| + alpha * |[A, B, C] - [A_o, B_o, C_o])| + beta * reg
Note:
    If the mask is sparse, then use our method
    If the mask is dense, then use the EM method
"""

# configuration, K is the temporal mode
I, J, K, R = 50, 50, 100, 10

def optimize(A, B):
    """
    Least Squares Solver: AX = B
    """
    L = la.cholesky(A + np.eye(A.shape[1]) * 1e-8)
    y = la.solve_triangular(L.T, B, lower=True)
    u = la.solve_triangular(L, y, lower=False)
    return u

def iterationCPD(T, A1, A2, A3, reg=1e-5):
    """
    The CPD routine
    """
    eye = np.eye(A1.shape[1])

    A1 = optimize(np.einsum('ir,im,jr,jm->rm',A2,A2,A3,A3) + reg * eye, np.einsum('ijk,jr,kr->ri',T,A2,A3)).T
    A2 = optimize(np.einsum('ir,im,jr,jm->rm',A1,A1,A3,A3) + reg * eye, np.einsum('ijk,ir,kr->rj',T,A1,A3)).T
    A3 = optimize(np.einsum('ir,im,jr,jm->rm',A1,A1,A2,A2) + reg * eye, np.einsum('ijk,ir,jr->rk',T,A1,A2)).T

    rec = np.einsum('ir,jr,kr->ijk',A1,A2,A3)
    return A1, A2, A3, rec

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

def iterationPre(A1, A2, A3, mask, T_, reg=1e-5):
    """
    The preparation routine (before streaming and CPC setting)
    """
    T_ = T_ * mask

    A1 = Optimizer(mask, [A1, A2, A3], np.einsum('ijk,jr,kr->ir',T_,A2,A3,optimize=True), 0, reg)
    A2 = Optimizer(mask, [A1, A2, A3], np.einsum('ijk,ir,kr->jr',T_,A1,A3,optimize=True), 1, reg)
    A3 = Optimizer(mask, [A1, A2, A3], np.einsum('ijk,ir,jr->kr',T_,A1,A2,optimize=True), 2, reg)
    return A1, A2, A3

def iterationStream(mask, T, A1, A2, A3, beta, reg=1e-5):
    """
    The streaming setting routine
    """
    rec = np.einsum('ir,jr,kr->ijk',A1,A2,A3,optimize=True)

    T_ = mask * T + (1 - mask) * rec
    eye = np.eye(A1.shape[1])

    A1 = optimize(np.einsum('ir,im,jr,jm->rm',A2,A2,A3,A3) + (reg + beta) * eye, np.einsum('ijk,jr,kr->ri',T_,A2,A3,optimize=True) + beta * A.T).T
    A2 = optimize(np.einsum('ir,im,jr,jm->rm',A1,A1,A3,A3) + (reg + beta) * eye, np.einsum('ijk,ir,kr->rj',T_,A1,A3,optimize=True) + beta * B.T).T
    A3 = optimize(np.einsum('ir,im,jr,jm->rm',A1,A1,A2,A2) + (reg + beta) * eye, np.einsum('ijk,ir,jr->rk',T_,A1,A2,optimize=True) + beta * C.T).T
    
    return A1, A2, A3

def maskOptimizer(Omega, A, A_, RHS, num, alpha, reg):
    """
    masked least square optimizer:
        A @ u.T = Omega * RHS
        number: which factor
        reg: 2-norm regulizer
    """
    N = len(A)
    R = A[0].shape[1]
    lst_mat, lst_mat3 = [], [A_[num]]
    T_inds = "".join([chr(ord('a')+i) for i in range(Omega.ndim)])
    einstr = ""
    for i in range(N):
        if i != num:
            einstr+=chr(ord('a')+i) + 'r' + ','
            lst_mat.append(A[i]); lst_mat3.append(A_[i])
            einstr+=chr(ord('a')+i) + 'z' + ','
            lst_mat.append(A[i]); lst_mat3.append(A[i])
    einstr2 = einstr[:-1] + "->rz"
    einstr3 = "tr," + einstr[:-1] + "->tz"
    einstr += T_inds + "->"+chr(ord('a')+num)+'rz'
    P2 = np.einsum(einstr2,*lst_mat,optimize=True)
    lst_mat.append(Omega)
    P = np.einsum(einstr,*lst_mat,optimize=True)
    P3 = np.einsum(einstr3,*lst_mat3,optimize=True)
    o = np.zeros(RHS.shape)

    for j in range(A[num].shape[0]):
        o[j,:] = np.linalg.inv(P[j] + alpha*P2 + reg*np.eye(R)) @ (RHS[j,:]  + alpha*P3[j,:])
    return o

def iterationCPC(A1, A2, A3, A, B, C, mask, T_, alpha=5e-3, reg=1e-5):
    """
    The CPC setting routine
    """
    T_ = mask * T_

    A1 = maskOptimizer(mask, [A1, A2, A3], [A, B, C], np.einsum('ijk,jr,kr->ir',T_,A2,A3,optimize=True), 0, alpha, reg)
    A2 = maskOptimizer(mask, [A1, A2, A3], [A, B, C], np.einsum('ijk,ir,kr->jr',T_,A1,A3,optimize=True), 1, alpha, reg)
    A3 = maskOptimizer(mask, [A1, A2, A3], [A, B, C], np.einsum('ijk,ir,jr->kr',T_,A1,A2,optimize=True), 2, alpha, reg)
    return A1, A2, A3

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

    # A0 = np.random.random((I, R))
    # B0 = np.random.random((J, R))
    # C0 = np.random.random((K, R))

    # X = np.einsum('ir,jr,kr->ijk',A0,B0,C0)

    import scipy.io as IO
    path = './exp-data/Indian_pines_corrected.mat'
    data = IO.loadmat(path)
    X = data['indian_pines_corrected']
    I, J, K, R = *X.shape, 5

    # X = np.random.random((I,J,K))
    # X += np.random.random(X.shape) * 0.1

    # the initial tensor and the mask
    amplitude, timestamp, sparsity, preIter = 1e3, 100, 0.95, 10
    mask_list = []
    mask_tensor = []
    for i in range(timestamp):
        tmp = np.random.random(X.shape)
        mask_list.append(tmp >= sparsity)
        mask_tensor.append(np.random.random(X.shape) * amplitude)

    print ('finish data loading')
    print ()

    """
    Preparation for Everyone
    """

    A = np.random.random((I,R))
    B = np.random.random((J,R))
    C = np.random.random((K,R))

    tic = time.time()

    for i in range(preIter):
        A, B, C, rec = iterationCPD(X, A, B, C, reg=1e-5)
        toc = time.time()
        rec, loss, PoF = metric(A, B, C, X)
        print ('loss:{}, PoF: {}, time: {}'.format(loss, PoF, toc - tic))
        tic = time.time()
    
    print ('finished preparation')
    print ()

    """
    Oracle CP Decomposition
    """

    A_, B_, C_ = A.copy(), B.copy(), C.copy()
    X_ = X.copy()

    tic_method1 = time.time()
    result_CPD = []
    time_CPD = []
    for mask, tensor in zip(mask_list, mask_tensor):
        X_ = X_ + mask * tensor
        A_, B_, C_, rec = iterationCPD(X_, A_, B_, C_, reg=1e-5)
        toc = time.time()
        rec, loss, PoF = metric(A_, B_, C_, X_); result_CPD.append(PoF); time_CPD.append(time.time() - tic_method1)
        print ('loss:{}, PoF: {}, time: {}'.format(loss, PoF, toc - tic))
        tic = time.time()
    print ('finish CPD')
    print ()



    """
    row-wise streaming CPC
    """

    A_, B_, C_ = A.copy(), B.copy(), C.copy()
    X_ = X.copy()

    tic_cpc = time.time()
    tic = time.time()
    result_cpc = []
    time_cpc = []
    for mask, tensor in zip(mask_list, mask_tensor):
        for i in range(1):
            X_ = X_ + mask * tensor
            A_, B_, C_ = iterationCPC(A_, B_, C_, A, B, C, mask, X_, alpha = 1)
            A, B, C = A_.copy(), B_.copy(), C_.copy()
        toc = time.time()
        rec, loss, PoF = metric(A, B, C, X_); result_cpc.append(PoF); time_cpc.append(time.time() - tic_cpc)
        print ('loss:{}, PoF:{}, time:{}'.format(loss, PoF, toc - tic))
        tic = time.time()

    print ('finish CPC-ALS')

    """
    Plot
    """

    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.plot(time_CPD, result_CPD, label="Initialization + step-by-step CPD")
    plt.plot(time_cpc, result_cpc, label="Initialization + GO-CPC")
    plt.legend()
    plt.ylabel('PoF')
    plt.xlabel('Running Time (s)')
    # plt.title('Synthetic Data')
    plt.title('Indian Pins')
    plt.show()
