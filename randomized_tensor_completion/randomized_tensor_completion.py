import numpy as np
from scipy import linalg as la
import time

# configuration
I, J, K, R = 100, 100, 100, 10


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
    T_ = T_ * mask

    A1 = Optimizer(mask, [A1, A2, A3], np.einsum('ijk,jr,kr->ir',T_,A2,A3,optimize=True), 0, reg)
    A2 = Optimizer(mask, [A1, A2, A3], np.einsum('ijk,ir,kr->jr',T_,A1,A3,optimize=True), 1, reg)
    A3 = Optimizer(mask, [A1, A2, A3], np.einsum('ijk,ir,jr->kr',T_,A1,A2,optimize=True), 2, reg)
    return A1, A2, A3

def iterationStream(mask, T, A1, A2, A3, beta, reg=1e-5):
    rec = np.einsum('ir,jr,kr->ijk',A_,B_,C_,optimize=True)

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
    T_ = mask * T_

    A1 = maskOptimizer(mask, [A1, A2, A3], [A, B, C], np.einsum('ijk,jr,kr->ir',T_,A2,A3,optimize=True), 0, alpha, reg)
    A2 = maskOptimizer(mask, [A1, A2, A3], [A, B, C], np.einsum('ijk,ir,kr->jr',T_,A1,A3,optimize=True), 1, alpha, reg)
    A3 = maskOptimizer(mask, [A1, A2, A3], [A, B, C], np.einsum('ijk,ir,jr->kr',T_,A1,A2,optimize=True), 2, alpha, reg)
    return A1, A2, A3


if __name__ == '__main__':
    
    """
    Data Generation
    """

    A0 = np.random.random((I, R))
    B0 = np.random.random((J, R))
    C0 = np.random.random((K, R))

    X = np.einsum('ir,jr,kr->ijk',A0,B0,C0)

    # X = np.random.random((I, J, K))

    # the mask streaming
    base, interval, preIter = 0.2, 0.009, 10
    mask_list = []
    mask_base = np.random.random(X.shape) >= 1 - base
    mask_cumsum = mask_base
    for i in range(int((1-base)/interval)-1):
        mask_tmp = np.random.random(X.shape) * (1 - mask_cumsum) > 1 - interval / (1 - base)
        mask_list.append(mask_tmp)
        mask_cumsum += mask_tmp
        base += interval
    mask_list.append(np.ones(X.shape) - mask_cumsum > 0)

    print ('finish data loading')
    print ()


    """
    Oracle CP Decomposition
    """

    A = np.random.random((I,R))
    B = np.random.random((J,R))
    C = np.random.random((K,R))

    tic = time.time()
    result_CPD = []
    for i in range(100):
        A, B, C, rec = iterationCPD(X, A, B, C, reg=1e-5)
        toc = time.time()
        result_CPD.append(1 - la.norm(rec - X) / la.norm(X))
        print ('PoF: {}, time: {}'.format(1 - la.norm(rec - X) / la.norm(X), toc - tic))
        tic = time.time()
    print ('finish CPD')
    print ()

    
    """
    Preaparation with base mask
    """

    # initialization
    A = np.random.random((I,R))
    B = np.random.random((J,R))
    C = np.random.random((K,R))

    tic = time.time()
    result_pre = []
    for i in range(preIter):
        A, B, C = iterationPre(A, B, C, mask_base, X)
        toc = time.time()
        rec = np.einsum('ir,jr,kr->ijk',A,B,C)
        loss = la.norm(mask_base * (rec - X)) ** 2
        PoF = 1 - la.norm(rec - X) / la.norm(X)
        result_pre.append(PoF)
        print ('loss:{}, PoF:{}, time:{}'.format(loss, PoF, toc-tic))
        tic = time.time()

    print ('finish preparation')
    print ()

    """
    Common streaming setting
    """
    
    A_, B_, C_ = A.copy(), B.copy(), C.copy()

    rec = np.einsum('ir,jr,kr->ijk',A_,B_,C_)

    tic = time.time()
    result_stream = result_pre.copy()
    for mask_item in mask_list:
        mask_base += mask_item

        for i in range(1):
            A_, B_, C_ = iterationStream(mask_item, X, A_, B_, C_, 1e-3)
        toc = time.time()
        rec = np.einsum('ir,jr,kr->ijk',A_,B_,C_)
        loss = la.norm(mask_base * (rec - X)) ** 2
        PoF = 1 - la.norm(rec - X) / la.norm(X)
        result_stream.append(PoF)
        print ('loss:{}, PoF:{}, time:{}'.format(loss, PoF, toc-tic))
        tic = time.time()

    print ('finish streaming setting')
    print ()

    """
    row-wise streaming CPC
    """

    A_, B_, C_ = A.copy(), B.copy(), C.copy()

    tic = time.time()
    result_cpc = result_pre.copy()
    for index, mask_item in enumerate(mask_list):
        mask_base += mask_item
        for i in range(1):
            A_, B_, C_ = iterationCPC(A_, B_, C_, A, B, C, mask_item, X, alpha= interval / (0.2 + index*interval))
            A, B, C = A_.copy(), B_.copy(), C_.copy()
        toc = time.time()
        rec = np.einsum('ir,jr,kr->ijk',A_,B_,C_)
        loss = la.norm(mask_base * (rec - X)) ** 2
        PoF = 1 - la.norm(rec - X) / la.norm(X)
        result_cpc.append(PoF)
        print ('loss:{}, PoF:{}, time:{}'.format(loss, PoF, toc - tic))
        tic = time.time()

    print ('finish CPC-ALS')

    #########################################
    ################# Plot ##################
    #########################################

    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.plot(result_CPD, label="CPD")
    plt.plot(result_stream, label="common streaming")
    plt.plot(result_cpc, label="augmented row-wise CPC")
    plt.legend()
    plt.show()