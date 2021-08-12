from typing import Awaitable
import numpy as np
from scipy import linalg as la
import time


"""
For Delayed Update Problem
    - tensor size changes along the time mode
    - tensor is always incomplete
    - the previous slides will also be updated
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

def optimize(A, B):
    L = la.cholesky(A + np.eye(A.shape[1]) * 1e-8)
    y = la.solve_triangular(L.T, B, lower=True)
    u = la.solve_triangular(L, y, lower=False)
    return u

def iterationCPD(T, A1, A2, A3, reg=1e-5):
    eye = np.eye(A1.shape[1])

    A1 = optimize(np.einsum('ir,im,jr,jm->rm',A2,A2,A3,A3) + reg * eye, np.einsum('ijk,jr,kr->ri',T,A2,A3)).T
    A2 = optimize(np.einsum('ir,im,jr,jm->rm',A1,A1,A3,A3) + reg * eye, np.einsum('ijk,ir,kr->rj',T,A1,A3)).T
    A3 = optimize(np.einsum('ir,im,jr,jm->rm',A1,A1,A2,A2) + reg * eye, np.einsum('ijk,ir,jr->rk',T,A1,A2)).T

    rec = np.einsum('ir,jr,kr->ijk',A1,A2,A3)
    return A1, A2, A3, rec

def Optimizer(Omega, A, RHS, num, reg=1e-8):
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
    T_ = T_ * mask

    A1 = Optimizer(mask, [A1, A2, A3], np.einsum('ijk,jr,kr->ir',T_,A2,A3,optimize=True), 0, reg)
    A2 = Optimizer(mask, [A1, A2, A3], np.einsum('ijk,ir,kr->jr',T_,A1,A3,optimize=True), 1, reg)
    A3 = Optimizer(mask, [A1, A2, A3], np.einsum('ijk,ir,jr->kr',T_,A1,A2,optimize=True), 2, reg)
    return A1, A2, A3

def iterationPre2(A1, A2, A3, mask, T_, reg=1e-5):
    T_ = T_ * mask + (1-mask) * np.einsum('ir,jr,kr->ijk',A1,A2,A3,optimize=True)
    eye = np.eye(A1.shape[1])
    A1 = optimize(np.einsum('ir,im,jr,jm->rm',A2,A2,A3,A3) + reg * eye, np.einsum('ijk,jr,kr->ri',T_,A2,A3)).T
    A2 = optimize(np.einsum('ir,im,jr,jm->rm',A1,A1,A3,A3) + reg * eye, np.einsum('ijk,ir,kr->rj',T_,A1,A3)).T
    A3 = optimize(np.einsum('ir,im,jr,jm->rm',A1,A1,A2,A2) + reg * eye, np.einsum('ijk,ir,jr->rk',T_,A1,A2)).T
    return A1, A2, A3

def iterationStream(mask, T, A, B, C, alpha, reg=1e-5):
    """
    input:
        - mask: I x J x (K+1)
        - T:    I x J x (K+1)
        - |omega * (x - Adiag(c)B.T)| + alpha * |mask * (X - [A, B, C_o])| + beta * reg
    """

    # get c
    c = Optimizer(mask[:,:,-1:], [A, B, np.random.random((1,R))], \
        np.einsum('ijk,ir,jr->kr',(T*mask)[:,:,-1:],A,B,optimize=True), 2, reg)

    eye = np.eye(R)
    # update A1, A2, A3
    rec_X = mask * T + (1-mask) * np.einsum('ir,jr,kr->ijk',A,B,np.concatenate([C, c], axis=0),optimize=True)
    A = optimize(alpha * np.einsum('ir,im,jr,jm->rm',B,B,C,C,optimize=True) + np.einsum('ir,im,jr,jm->rm',B,B,c,c,optimize=True) + reg * eye, \
                alpha * np.einsum('ijk,jr,kr->ri',rec_X[:,:,:-1],B,C,optimize=True) + np.einsum('ijk,jr,kr->ri',rec_X[:,:,-1:],B,c,optimize=True)).T
    B = optimize(alpha * np.einsum('ir,im,jr,jm->rm',A,A,C,C,optimize=True) + np.einsum('ir,im,jr,jm->rm',A,A,c,c,optimize=True) + reg * eye, \
                alpha * np.einsum('ijk,ir,kr->rj',rec_X[:,:,:-1],A,C,optimize=True) + np.einsum('ijk,ir,kr->rj',rec_X[:,:,-1:],A,c,optimize=True)).T
    C = optimize(np.einsum('ir,im,jr,jm->rm',A,A,B,B) + reg * eye, np.einsum('ijk,ir,jr->rk',rec_X[:,:,:-1],A,B,optimize=True)).T
    C = np.concatenate([C, c], axis=0)

    return A, B, C


def maskOptimizer(Omega, Omega_, A, A_, RHS, RHS_, num, alpha, reg=1e-8):
    """
    masked least square optimizer:
        A @ u.T = Omega * RHS
        number: which factor
        reg: 2-norm regulizer

    P is for normal left hand
    P2 is for historical left hand
    P3 is for historical right hand
    """
    
    N = len(A)
    R = A[0].shape[1]
    lst_mat, lst_mat_, lst_mat2, lst_mat3 = [], [], [], [A_[num]]
    T_inds = "".join([chr(ord('a')+i) for i in range(Omega.ndim)])
    einstr = ""
    for i in range(N):
        if i != num:
            if i == N-1:
                lst_mat.append(A_[-1]); lst_mat.append(A_[-1])
            else:
                lst_mat.append(A[i]); lst_mat.append(A[i])
            lst_mat_.append(A[i]); lst_mat_.append(A[i])
            einstr+=chr(ord('a')+i) + 'r' + ','
            lst_mat2.append(A[i]); lst_mat3.append(A_[i])
            einstr+=chr(ord('a')+i) + 'z' + ','
            lst_mat2.append(A[i]); lst_mat3.append(A[i])
    einstr2 = einstr[:-1] + "->rz"
    einstr3 = "tr," + einstr[:-1] + "->tz"
    einstr += T_inds + "->"+chr(ord('a')+num)+'rz'
    P2 = np.einsum(einstr2,*lst_mat2,optimize=True)
    lst_mat.append(Omega)
    P = np.einsum(einstr,*lst_mat,optimize=True)
    P3 = np.einsum(einstr3,*lst_mat3,optimize=True)
    o = np.zeros(RHS.shape)

    lst_mat_.append(Omega_)
    P_ = np.einsum(einstr,*lst_mat_,optimize=True)

    I = np.eye(R)
    for j in range(A[num].shape[0]):
        o[j,:] = np.linalg.inv(P[j] + P_[j] + alpha*P2 + reg*I) @ (RHS[j,:] + RHS_[j,:]  + alpha*P3[j,:])
    return o

def maskOptimizer2(Omega, A, A_, RHS, num, alpha, reg=1e-8):
    """
    masked least square optimizer:
        A @ u.T = Omega * RHS
        number: which factor
        reg: 2-norm regulizer

    P is for normal left hand
    P2 is for historical left hand
    P3 is for historical right hand
    """
    
    N = len(A)
    R = A[0].shape[1]
    lst_mat, lst_mat2, lst_mat3 = [], [], [A_[num]]
    T_inds = "".join([chr(ord('a')+i) for i in range(Omega.ndim)])
    einstr = ""
    for i in range(N):
        if i != num:
            lst_mat.append(A[i]); lst_mat.append(A[i])
            einstr+=chr(ord('a')+i) + 'r' + ','
            lst_mat2.append(A[i]); lst_mat3.append(A_[i])
            einstr+=chr(ord('a')+i) + 'z' + ','
            lst_mat2.append(A[i]); lst_mat3.append(A[i])
    einstr2 = einstr[:-1] + "->rz"
    einstr3 = "tr," + einstr[:-1] + "->tz"
    einstr += T_inds + "->"+chr(ord('a')+num)+'rz'
    P2 = np.einsum(einstr2,*lst_mat2,optimize=True)
    lst_mat.append(Omega)
    P = np.einsum(einstr,*lst_mat,optimize=True)
    P3 = np.einsum(einstr3,*lst_mat3,optimize=True)
    o = np.zeros(RHS.shape)

    I = np.eye(R)
    for j in range(A[num].shape[0]):
        o[j,:] = np.linalg.inv(P[j] + alpha*P2 + reg*I) @ (RHS[j,:] + alpha*P3[j,:])
    return o

def iterationCPC(mask, T, A, B, C, A1, A2, A3, alpha=1, reg=1e-5):
    """
    input:
        - mask: I x J x (K+1)
        - T:    I x J x (K+1)
    |omega * (x - Adiag(c)B.T)| + alpha * |[A,B,C] - [A_o, B_o, C_o]| + beta * reg
    """

    mask_ = mask[:,:,-1:]
    T_ = T[:,:,-1:]
    # # get c
    c = Optimizer(mask_, [A1, A2, np.random.random((1,R))], \
        np.einsum('ijk,ir,jr->kr',T_*mask_,A,B,optimize=True), 2, reg)

    _mask = mask[:, :, :-1]

    A = maskOptimizer(mask_, _mask, [A, B, C], [A1, A2, A3, c], np.einsum('ijk,jr,kr->ir',T_*mask_,B,c,optimize=True), np.einsum('ijk,jr,kr->ir',T[:, :, :-1]*_mask,B,C,optimize=True), 0, alpha, reg)
    B = maskOptimizer(mask_, _mask, [A, B, C], [A1, A2, A3, c], np.einsum('ijk,ir,kr->jr',T_*mask_,A,c,optimize=True), np.einsum('ijk,ir,kr->jr',T[:, :, :-1]*_mask,A,C,optimize=True), 1, alpha, reg)
    C = maskOptimizer2(_mask, [A, B, C], [A1, A2, A3], np.einsum('ijk,ir,jr->kr',T[:, :, :-1]*_mask,A,B,optimize=True), 2, alpha, reg)
    C = np.concatenate([C, c], axis=0)

    return A, B, C

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

    import pickle 
    path = './exp-data/iqvia_tensor_data_state_2018.pickle'
    data = pickle.load(open(path, 'rb'))
    X = np.concatenate([data[0], data[1]], axis=2) # 49 x 22 x 52 x 12
    X = np.transpose(X[0,:,:,:], (0, 2, 1))

    I, J, K = X.shape
    R = 10
    # X += np.random.random(X.shape) * 0.1

    # the initial tensor and the mask
    base, preIter = 0.5, 100
    T = int(X.shape[2] * base) # should be larger than L
    mask_base = np.ones((I,J,T))
    for i in range(1,J):
        mask_base[:,i:,-i] = 0
    mask_list = [mask_base]
    for i in range(K-T):
        temp = np.concatenate([np.ones((I,J,1)), mask_list[-1]], axis=2)
        mask_list.append(temp)
    print ('finish data loading')
    print ()

    mask_list = mask_list[1:]


    """
    Oracle CP Decomposition
    """

    A = np.random.random((I,R))
    B = np.random.random((J,R))
    C = np.random.random((K,R))

    tic = time.time()
    result_CPD = []
    for i in range(X.shape[2] - T + preIter):
        A, B, C, rec = iterationCPD(X, A, B, C, reg=1e-5)
        toc = time.time()
        rec, loss, PoF = metric(A, B, C, X); result_CPD.append(PoF)
        print ('loss:{}, PoF: {}, time: {}'.format(loss, PoF, toc - tic))
        tic = time.time()
    print ('finish CPD')
    print ()

    
    """
    Preaparation with base mask
    """

    # initialization
    A = np.random.randn(I,R)
    B = np.random.randn(J,R)
    C = np.random.randn(T,R)

    tic = time.time()
    result_pre = []
    # mask_base = np.ones(X[:,:,:T].shape)
    for i in range(preIter):
        A, B, C = iterationPre2(A, B, C, mask_base, X[:,:,:T])
        toc = time.time()
        rec, loss, PoF = metric(A, B, C, X[:,:,:T], mask_base); result_pre.append(PoF)
        print ('loss:{}, PoF:{}, time:{}'.format(loss, PoF, toc-tic))
        tic = time.time()

    print ('finish preparation')
    print ()

    """
    Common streaming setting
    """
    
    A_, B_, C_ = A.copy(), B.copy(), C.copy()
    T_ = T
    rec = np.einsum('ir,jr,kr->ijk',A_,B_,C_)

    tic = time.time()
    result_stream = result_pre.copy()
    for index, mask_base_ in enumerate(mask_list):
        A_, B_, C_ = iterationStream(mask_base_, X[:,:,:T_+1], A_, B_, C_, alpha=1, reg=1e-5)
        T_ += 1; toc = time.time()
        rec, loss, PoF = metric(A_, B_, C_, X[:,:,:T_], mask_base_); result_stream.append(PoF)
        print ('loss:{}, PoF:{}, time:{}'.format(loss, PoF, toc-tic))
        tic = time.time()

    print ('finish streaming setting')
    print ()


    """
    row-wise streaming CPC
    """

    A_, B_, C_ = A.copy(), B.copy(), C.copy()
    T_ = T

    tic = time.time()
    result_cpc = result_pre.copy()
    for index, mask_base_ in enumerate(mask_list):
        for i in range(1):
            A_, B_, C_ = iterationCPC(mask_base_, X[:,:,:T_+1], A_, B_, C_, A, B, C, 5 / (base * K + index))
            A, B, C = A_.copy(), B_.copy(), C_.copy()
        T_ += 1; toc = time.time()
        rec, loss, PoF = metric(A, B, C, X[:,:,:T_], mask_base_); result_cpc.append(PoF)
        print ('loss:{}, PoF:{}, time:{}'.format(loss, PoF, toc - tic))
        tic = time.time()

    print ('finish CPC-ALS')


    """
    Plot
    """
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.plot(np.array(result_CPD), label="Oracle CPD")
    plt.plot(np.array(result_stream), label="EM CPD (on base tensor) + EM CPD (grow)")
    plt.plot(np.array(result_cpc), label="EM CPD (on base tensor) + row-wise LS (grow)")
    plt.legend()
    plt.ylabel('PoF')
    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.show()
