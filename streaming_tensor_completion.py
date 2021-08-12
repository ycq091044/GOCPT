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
I, J, K, R = 50, 50, 200, 5

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

def iterationStream(mask, T, A, B, C, alpha, reg=1e-5, gamma=0.99):
    """
    input:
        - mask: I x J x K
        - T:    I x J x K
        - |omega * (x - Adiag(c)B.T)| + alpha * |mask * (X - [A, B, C_o])| + beta * reg
    """

    # get c
    c = Optimizer(mask[:,:,-1:], [A, B, np.random.random((1,R))], \
        np.einsum('ijk,ir,jr->kr',(T*mask)[:,:,-1:],A,B,optimize=True), 2, reg)

    coeff = np.array([gamma ** i for i in range(1, mask.shape[2])])[::-1]


    eye = np.eye(R)
    # update A1, A2, A3
    rec_X = mask * T + (1-mask) * np.einsum('ir,jr,kr->ijk',A,B,np.concatenate([C, c], axis=0),optimize=True)
    A = optimize(alpha * np.einsum('ir,im,jr,jm,j->rm',B,B,C,C,coeff,optimize=True) + np.einsum('ir,im,jr,jm->rm',B,B,c,c,optimize=True) + reg * eye, \
                alpha * np.einsum('ijk,jr,kr,k->ri',rec_X[:,:,:-1],B,C,coeff,optimize=True) + np.einsum('ijk,jr,kr->ri',rec_X[:,:,-1:],B,c,optimize=True)).T
    B = optimize(alpha * np.einsum('ir,im,jr,jm,j->rm',A,A,C,C,coeff,optimize=True) + np.einsum('ir,im,jr,jm->rm',A,A,c,c,optimize=True) + reg * eye, \
                alpha * np.einsum('ijk,ir,kr,k->rj',rec_X[:,:,:-1],A,C,coeff,optimize=True) + np.einsum('ijk,ir,kr->rj',rec_X[:,:,-1:],A,c,optimize=True)).T
    # C = optimize(np.einsum('ir,im,jr,jm->rm',A,A,B,B) * coeff.sum() + reg * eye, np.einsum('ijk,ir,jr,k->rk',rec_X[:,:,:-1],A,B,coeff,optimize=True)).T
    # A = optimize(alpha * np.einsum('ir,im,jr,jm->rm',B,B,C,C,optimize=True) + np.einsum('ir,im,jr,jm->rm',B,B,c,c,optimize=True) + reg * eye, \
    #             alpha * np.einsum('ijk,jr,kr->ri',rec_X[:,:,:-1],B,C,optimize=True) + np.einsum('ijk,jr,kr->ri',rec_X[:,:,-1:],B,c,optimize=True)).T
    # B = optimize(alpha * np.einsum('ir,im,jr,jm->rm',A,A,C,C,optimize=True) + np.einsum('ir,im,jr,jm->rm',A,A,c,c,optimize=True) + reg * eye, \
    #             alpha * np.einsum('ijk,ir,kr->rj',rec_X[:,:,:-1],A,C,optimize=True) + np.einsum('ijk,ir,kr->rj',rec_X[:,:,-1:],A,c,optimize=True)).T
    C = optimize(np.einsum('ir,im,jr,jm->rm',A,A,B,B) + reg * eye, np.einsum('ijk,ir,jr->rk',rec_X[:,:,:-1],A,B,optimize=True)).T
    C = np.concatenate([C, c], axis=0)

    return A, B, C

def iterationStream2(mask, T, A, B, C, alpha, reg=1e-5, gamma=0.99):
    """
    input:
        - mask: I x J x K
        - T:    I x J x K
        - |omega * (x - Adiag(c)B.T)| + alpha * |mask * (X - [A, B, C_o])| + beta * reg
    """

    # get c
    c = Optimizer(mask[:,:,-1:], [A, B, np.random.random((1,R))], \
        np.einsum('ijk,ir,jr->kr',(T*mask)[:,:,-1:],A,B,optimize=True), 2, reg)

    coeff = np.array([1.0 ** i for i in range(1, mask.shape[2])])[::-1]


    eye = np.eye(R)
    # update A1, A2, A3
    rec_X = mask * T + (1-mask) * np.einsum('ir,jr,kr->ijk',A,B,np.concatenate([C, c], axis=0),optimize=True)
    A = optimize(alpha * np.einsum('ir,im,jr,jm,j->rm',B,B,C,C,coeff,optimize=True) + np.einsum('ir,im,jr,jm->rm',B,B,c,c,optimize=True) + reg * eye, \
                alpha * np.einsum('ijk,jr,kr,k->ri',rec_X[:,:,:-1],B,C,coeff,optimize=True) + np.einsum('ijk,jr,kr->ri',rec_X[:,:,-1:],B,c,optimize=True)).T
    B = optimize(alpha * np.einsum('ir,im,jr,jm,j->rm',A,A,C,C,coeff,optimize=True) + np.einsum('ir,im,jr,jm->rm',A,A,c,c,optimize=True) + reg * eye, \
                alpha * np.einsum('ijk,ir,kr,k->rj',rec_X[:,:,:-1],A,C,coeff,optimize=True) + np.einsum('ijk,ir,kr->rj',rec_X[:,:,-1:],A,c,optimize=True)).T
    # C = optimize(np.einsum('ir,im,jr,jm->rm',A,A,B,B) * coeff.sum() + reg * eye, np.einsum('ijk,ir,jr,k->rk',rec_X[:,:,:-1],A,B,coeff,optimize=True)).T
    # A = optimize(alpha * np.einsum('ir,im,jr,jm->rm',B,B,C,C,optimize=True) + np.einsum('ir,im,jr,jm->rm',B,B,c,c,optimize=True) + reg * eye, \
    #             alpha * np.einsum('ijk,jr,kr->ri',rec_X[:,:,:-1],B,C,optimize=True) + np.einsum('ijk,jr,kr->ri',rec_X[:,:,-1:],B,c,optimize=True)).T
    # B = optimize(alpha * np.einsum('ir,im,jr,jm->rm',A,A,C,C,optimize=True) + np.einsum('ir,im,jr,jm->rm',A,A,c,c,optimize=True) + reg * eye, \
    #             alpha * np.einsum('ijk,ir,kr->rj',rec_X[:,:,:-1],A,C,optimize=True) + np.einsum('ijk,ir,kr->rj',rec_X[:,:,-1:],A,c,optimize=True)).T
    C = optimize(np.einsum('ir,im,jr,jm->rm',A,A,B,B) + reg * eye, np.einsum('ijk,ir,jr->rk',rec_X[:,:,:-1],A,B,optimize=True)).T
    C = np.concatenate([C, c], axis=0)

    return A, B, C

def iterationStream3(mask, T, A, B, C, alpha, reg=1e-5, index = 1):
    """
    input:
        - mask: I x J x K
        - T:    I x J x K
        - |omega * (x - Adiag(c)B.T)| + alpha * |mask * (X - [A, B, C_o])| + beta * reg
    """

    # get c
    c = Optimizer(mask[:,:,-1:], [A, B, np.random.random((1,R))], \
        np.einsum('ijk,ir,jr->kr',(T*mask)[:,:,-1:],A,B,optimize=True), 2, reg)

    mask_ = mask[:,:,-1:]
    T_ = T[:,:,-1:]


    eye = np.eye(R)
    # update A1, A2, A3
    rec_X = np.einsum('ir,jr,kr->ijk',A,B,c,optimize=True)
    gradA = np.einsum('ijk,jr,kr->ir',mask_*(T_-rec_X),B,c,optimize=True)
    gradB = np.einsum('ijk,ir,kr->jr',mask_*(T_-rec_X),A,c,optimize=True)
    A = (1-alpha*reg/(index+1)) * A - alpha*gradA
    B = (1-alpha*reg/(index+1)) * B - alpha*gradB
    # A = optimize(alpha * np.einsum('ir,im,jr,jm,j->rm',B,B,C,C,coeff,optimize=True) + np.einsum('ir,im,jr,jm->rm',B,B,c,c,optimize=True) + reg * eye, \
    #             alpha * np.einsum('ijk,jr,kr,k->ri',rec_X[:,:,:-1],B,C,coeff,optimize=True) + np.einsum('ijk,jr,kr->ri',rec_X[:,:,-1:],B,c,optimize=True)).T
    # B = optimize(alpha * np.einsum('ir,im,jr,jm,j->rm',A,A,C,C,coeff,optimize=True) + np.einsum('ir,im,jr,jm->rm',A,A,c,c,optimize=True) + reg * eye, \
    #             alpha * np.einsum('ijk,ir,kr,k->rj',rec_X[:,:,:-1],A,C,coeff,optimize=True) + np.einsum('ijk,ir,kr->rj',rec_X[:,:,-1:],A,c,optimize=True)).T
    # C = optimize(np.einsum('ir,im,jr,jm->rm',A,A,B,B) * coeff.sum() + reg * eye, np.einsum('ijk,ir,jr,k->rk',rec_X[:,:,:-1],A,B,coeff,optimize=True)).T
    # A = optimize(alpha * np.einsum('ir,im,jr,jm->rm',B,B,C,C,optimize=True) + np.einsum('ir,im,jr,jm->rm',B,B,c,c,optimize=True) + reg * eye, \
    #             alpha * np.einsum('ijk,jr,kr->ri',rec_X[:,:,:-1],B,C,optimize=True) + np.einsum('ijk,jr,kr->ri',rec_X[:,:,-1:],B,c,optimize=True)).T
    # B = optimize(alpha * np.einsum('ir,im,jr,jm->rm',A,A,C,C,optimize=True) + np.einsum('ir,im,jr,jm->rm',A,A,c,c,optimize=True) + reg * eye, \
    #             alpha * np.einsum('ijk,ir,kr->rj',rec_X[:,:,:-1],A,C,optimize=True) + np.einsum('ijk,ir,kr->rj',rec_X[:,:,-1:],A,c,optimize=True)).T
    # C = optimize(np.einsum('ir,im,jr,jm->rm',A,A,B,B) + reg * eye, np.einsum('ijk,ir,jr->rk',rec_X[:,:,:-1],A,B,optimize=True)).T
    C = np.concatenate([C, c], axis=0)

    return A, B, C


def maskOptimizer(Omega, A, A_, RHS, num, alpha, reg=1e-8):
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
            if i == N-1:
                lst_mat.append(A_[-1]); lst_mat.append(A_[-1])
            else:
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
        o[j,:] = np.linalg.inv(P[j] + alpha*P2 + reg*I) @ (RHS[j,:]  + alpha*P3[j,:])
    return o

def iterationCPC(mask, T, A, B, C, A1, A2, A3, alpha=1, reg=1e-5):
    """
    input:
        - mask: I x J
        - T:    I x J x 1
    |omega * (x - Adiag(c)B.T)| + alpha * |[A,B,C] - [A_o, B_o, C_o]| + beta * reg
    """
    mask = mask[:, :, np.newaxis]
    # # get c
    c = Optimizer(mask, [A1, A2, np.random.random((1,R))], \
        np.einsum('ijk,ir,jr->kr',T*mask,A,B,optimize=True), 2, reg)
    eye = np.eye(R)
    A = maskOptimizer(mask, [A, B, C], [A1, A2, A3, c], np.einsum('ijk,jr,kr->ir',T*mask,B,c,optimize=True), 0, alpha, reg)
    B = maskOptimizer(mask, [A, B, C], [A1, A2, A3, c], np.einsum('ijk,ir,kr->jr',T*mask,A,c,optimize=True), 1, alpha, reg)
    C = optimize(np.einsum('ir,im,jr,jm->rm',A,A,B,B,optimize=True) + reg * eye, np.einsum('kr,ir,im,jr,jm->mk',A3,A1,A,A2,B,optimize=True)).T
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

    A0 = np.random.random((I, R))
    B0 = np.random.random((J, R))
    C0 = np.random.random((K, R))

    X = np.einsum('ir,jr,kr->ijk',A0,B0,C0)
    # X += np.random.random(X.shape) * 0.1

    # import scipy.io as IO
    # path = './exp-data/Indian_pines_corrected.mat'
    # data = IO.loadmat(path)
    # X = data['indian_pines_corrected']
    # I, J, K, R = *X.shape, 5

    # the initial tensor and the mask
    base, sparsity, preIter = 0.1, 0.95, 10
    T = int(X.shape[2] * base)
    mask_base = np.random.random((*X.shape[:2],T)) >= sparsity
    mask_list = []
    for i in range(X.shape[2] - T):
        mask_tmp = np.random.random(X.shape[:2]) >= sparsity
        mask_list.append(mask_tmp)
    print ('finish data loading')
    print ()


    # """
    # Oracle CP Decomposition
    # """

    # A = np.random.random((I,R))
    # B = np.random.random((J,R))
    # C = np.random.random((K,R))

    # tic = time.time()
    # result_CPD = []
    # for i in range(X.shape[2] - T + preIter):
    #     A, B, C, rec = iterationCPD(X, A, B, C, reg=1e-5)
    #     toc = time.time()
    #     rec, loss, PoF = metric(A, B, C, X); result_CPD.append(PoF)
    #     print ('loss:{}, PoF: {}, time: {}'.format(loss, PoF, toc - tic))
    #     tic = time.time()
    # print ('finish CPD')
    # print ()

    
    """
    Preparation with base mask
    """

    # initialization
    A = np.random.random((I,R))
    B = np.random.random((J,R))
    C = np.random.random((T,R))

    tic_pre = time.time()
    tic = time.time()
    result_pre = []
    time_pre = []
    for i in range(preIter):
        A, B, C = iterationPre(A, B, C, mask_base, X[:,:,:T])
        toc = time.time()
        rec, loss, PoF = metric(A, B, C, X[:,:,:T], mask_base); result_pre.append(PoF); time_pre.append(time.time() - tic_pre)
        print ('loss:{}, PoF:{}, time:{}'.format(loss, PoF, toc-tic))
        tic = time.time()

    print ('finish preparation')
    print ()


    """
    Common streaming setting (weighted)
    """
    
    A_, B_, C_ = A.copy(), B.copy(), C.copy()
    T_ = T
    mask_base_ = mask_base.copy()
    rec = np.einsum('ir,jr,kr->ijk',A_,B_,C_)

    tic_method1 = time.time()
    tic = time.time()
    result_stream = [] #result_pre.copy()
    time_stream = []
    for index, mask_item in enumerate(mask_list):
        mask_base_ = np.concatenate([mask_base_, mask_item[:, :, np.newaxis]], axis=2)
        A_, B_, C_ = iterationStream(mask_base_, X[:,:,:T_+1], A_, B_, C_, alpha=1, reg=1e-5, gamma=0.99)
        T_ += 1; toc = time.time()
        rec, loss, PoF = metric(A_, B_, C_, X[:,:,:T_], mask_base_); result_stream.append(PoF); time_stream.append(time.time() - tic_method1)
        print ('loss:{}, PoF:{}, time:{}'.format(loss, PoF, toc-tic))
        tic = time.time()

    # time_stream = time_pre + [i+time_pre[-1] for i in time_stream]
    print ('finish streaming setting')
    print ()


    """
    common streaming setting
    """
    
    A_, B_, C_ = A.copy(), B.copy(), C.copy()
    T_ = T
    mask_base_ = mask_base.copy()
    rec = np.einsum('ir,jr,kr->ijk',A_,B_,C_)

    tic_method2 = time.time()
    tic = time.time()
    result_stream2 = [] #result_pre.copy()
    time_stream2 = []
    for index, mask_item in enumerate(mask_list):
        mask_base_ = np.concatenate([mask_base_, mask_item[:, :, np.newaxis]], axis=2)
        A_, B_, C_ = iterationStream2(mask_base_, X[:,:,:T_+1], A_, B_, C_, alpha=1, reg=1e-5)
        T_ += 1; toc = time.time()
        rec, loss, PoF = metric(A_, B_, C_, X[:,:,:T_], mask_base_); result_stream2.append(PoF); time_stream2.append(time.time() - tic_method2)
        print ('loss:{}, PoF:{}, time:{}'.format(loss, PoF, toc-tic))
        tic = time.time()

    # time_stream = time_pre + [i+time_pre[-1] for i in time_stream]
    print ('finish streaming2 setting')
    print ()


    """
    Recursive Least Squares
    """
    
    A_, B_, C_ = A.copy(), B.copy(), C.copy()
    T_ = T
    mask_base_ = mask_base.copy()
    rec = np.einsum('ir,jr,kr->ijk',A_,B_,C_)

    tic_method3 = time.time()
    tic = time.time()
    result_stream3 = [] #result_pre.copy()
    time_stream3 = []
    for index, mask_item in enumerate(mask_list):
        mask_base_ = np.concatenate([mask_base_, mask_item[:, :, np.newaxis]], axis=2)
        A_, B_, C_ = iterationStream3(mask_base_, X[:,:,:T_+1], A_, B_, C_, alpha=1e-10, reg=1e-5, index=index)
        T_ += 1; toc = time.time()
        rec, loss, PoF = metric(A_, B_, C_, X[:,:,:T_], mask_base_); result_stream3.append(PoF); time_stream3.append(time.time() - tic_method3)
        print ('loss:{}, PoF:{}, time:{}'.format(loss, PoF, toc-tic))
        tic = time.time()

    # time_stream = time_pre + [i+time_pre[-1] for i in time_stream]
    print ('finish streaming2 setting')
    print ()


    """
    row-wise streaming CPC
    """

    A_, B_, C_ = A.copy(), B.copy(), C.copy()
    T_ = T
    mask_base_ = mask_base.copy()

    tic_cpc = time.time()
    tic = time.time()
    result_cpc = [] #result_pre.copy()
    time_cpc = []
    for index, mask_item in enumerate(mask_list):
        mask_base_ = np.concatenate([mask_base_, mask_item[:, :, np.newaxis]], axis=2)
        for i in range(1):
            A_, B_, C_ = iterationCPC(mask_item, X[:,:,T_:T_+1], A_, B_, C_, A, B, C, 0.5 / (base * K + index))
            A, B, C = A_.copy(), B_.copy(), C_.copy()
        T_ += 1; toc = time.time()
        rec, loss, PoF = metric(A, B, C, X[:,:,:T_], mask_base_); result_cpc.append(PoF); time_cpc.append(time.time() - tic_cpc)
        print ('loss:{}, PoF:{}, time:{}'.format(loss, PoF, toc - tic))
        tic = time.time()

    # time_cpc = time_pre + [i+time_pre[-1] for i in time_cpc]
    print ('finish CPC-ALS')


    """
    Plot
    """
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.rcParams.update({"font.size":12})
    plt.plot(np.array(time_stream), np.array(result_stream), label="Initialization + EM-ALS (exponential decay)")
    plt.plot(np.array(time_stream2), np.array(result_stream2), label="Initialization + EM-ALS")
    plt.plot(np.array(time_stream3), np.array(result_stream3), label="Initialization + Recursive Least Square")
    plt.plot(np.array(time_cpc), np.array(result_cpc), label="Initialization + GO-CPC")
    plt.legend()
    plt.ylabel('PoF')
    plt.yscale('log')
    plt.xlabel('Running Time (s)')
    plt.title('Synthetic Data')
    plt.tight_layout()
    plt.show()
