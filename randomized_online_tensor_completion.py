import numpy as np
from scipy import linalg as la
import time
import pickle

"""
For Randomized Online Tensor Completion
    - tensor size does not change
    - tensor is incomplete and the entries are gradually filled
Methods
    - Oracle CPD on the overall complete tensor
    - step by step EM imputation + CP decomposition
    - row-wise least squares method
Note:
    If the mask is sparse, then use our method
    If the mask is dense, then use the EM method
"""
# configuration
I, J, K, R = 100, 100, 100, 5


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

    A1 = optimize(np.einsum('ir,im,jr,jm->rm',A2,A2,A3,A3,optimize=True) + reg * eye, np.einsum('ijk,jr,kr->ri',T,A2,A3,optimize=True)).T
    A2 = optimize(np.einsum('ir,im,jr,jm->rm',A1,A1,A3,A3,optimize=True) + reg * eye, np.einsum('ijk,ir,kr->rj',T,A1,A3,optimize=True)).T
    A3 = optimize(np.einsum('ir,im,jr,jm->rm',A1,A1,A2,A2,optimize=True) + reg * eye, np.einsum('ijk,ir,jr->rk',T,A1,A2,optimize=True)).T

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

def iterationStream(mask, T, A1, A2, A3, reg=1e-5):
    """
    The streaming setting routine
    """
    rec = np.einsum('ir,jr,kr->ijk',A1,A2,A3,optimize=True)

    T_ = mask * T + (1 - mask) * rec
    eye = np.eye(A1.shape[1])

    A1 = optimize(np.einsum('ir,im,jr,jm->rm',A2,A2,A3,A3,optimize=True) + reg * eye, np.einsum('ijk,jr,kr->ri',T_,A2,A3,optimize=True)).T
    A2 = optimize(np.einsum('ir,im,jr,jm->rm',A1,A1,A3,A3,optimize=True) + reg * eye, np.einsum('ijk,ir,kr->rj',T_,A1,A3,optimize=True)).T
    A3 = optimize(np.einsum('ir,im,jr,jm->rm',A1,A1,A2,A2,optimize=True) + reg * eye, np.einsum('ijk,ir,jr->rk',T_,A1,A2,optimize=True)).T
    
    return A1, A2, A3

def iterationStream2(mask, T, A1, A2, A3, lr=1e-3, reg=1e-5):
    """
    The streaming2 setting routine
    """
    rec = np.einsum('ir,jr,kr->ijk',A1,A2,A3,optimize=True)

    grad1 = np.einsum('ijk,jr,kr->ir',mask * (rec - T),A2,A3,optimize=True) + reg * A1
    grad2 = np.einsum('ijk,ir,kr->jr',mask * (rec - T),A1,A3,optimize=True) + reg * A2
    grad3 = np.einsum('ijk,ir,jr->kr',mask * (rec - T),A1,A2,optimize=True) + reg * A3
    
    A1 -= grad1 * lr
    A2 -= grad2 * lr
    A3 -= grad3 * lr

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
    X = pickle.load(open('./exp-data/FACE-3D.pkl', 'rb'))
    I, J, K, R = *X.shape, 5

    # X = np.random.random((I, J, K))

    # the mask streaming
    base, interval, preIter = 0.1, 0.009, 5
    base_ = base
    mask_list = []
    mask_base = np.random.random(X.shape) >= 1 - base
    mask_cumsum = mask_base
    for i in range(int((1-base)/interval)-1):
        mask_tmp = np.random.random(X.shape) * (1 - mask_cumsum) > 1 - interval / (1 - base)
        mask_list.append(mask_tmp)
        mask_cumsum += mask_tmp
        base += interval
    mask_list.append(np.ones(X.shape) -  mask_cumsum > 0)

    print ('finish data loading')
    print ()

    """
    Oracle CP Decomposition
    """

    A = np.random.random((I,R))
    B = np.random.random((J,R))
    C = np.random.random((K,R))

    A_, B_, C_ = A.copy(), B.copy(), C.copy()

    tic_oracle = time.time()
    tic = time.time()
    result_CPD = []
    time_CPD = []
    for i in range(preIter + len(mask_list)):
        A_, B_, C_, rec = iterationCPD(X, A_, B_, C_, reg=1e-5)
        toc = time.time()
        rec, loss, PoF = metric(A_, B_, C_, X); result_CPD.append(PoF); time_CPD.append(time.time() - tic_oracle)
        print ('loss:{}, PoF: {}, time: {}'.format(loss, PoF, toc - tic))
        tic = time.time()
    print ('finish CPD')
    print ()

    
    """
    Preaparation with base mask
    """

    # initialization
    tic_pre = time.time()
    tic = time.time()
    result_pre = []
    time_pre = []
    for i in range(preIter):
        A, B, C = iterationPre(A, B, C, mask_base, X)
        toc = time.time()
        rec, loss, PoF = metric(A, B, C, X, mask_base); result_pre.append(PoF); time_pre.append(time.time() - tic_pre)
        print ('loss:{}, PoF:{}, time:{}'.format(loss, PoF, toc-tic))
        tic = time.time()

    print ('finish preparation')
    print ()

    """
    Common streaming setting
    """
    
    A_, B_, C_ = A.copy(), B.copy(), C.copy()
    mask_base_ = mask_base.copy()

    tic_method1 = time.time()
    tic = time.time()
    result_stream = result_pre.copy()
    time_stream = []
    for mask_item in mask_list:
        mask_base_ += mask_item

        for i in range(1):
            A_, B_, C_ = iterationStream(mask_item, X, A_, B_, C_)
        toc = time.time()
        rec, loss, PoF = metric(A_, B_, C_, X, mask_base_); result_stream.append(PoF); time_stream.append(time.time() - tic_method1)
        print ('loss:{}, PoF:{}, time:{}'.format(loss, PoF, toc-tic))
        tic = time.time()

    time_stream = time_pre + [i+time_pre[-1] for i in time_stream]
    print ('finish streaming setting')
    print ()


    """
    Gradient based method
    """
    
    A_, B_, C_ = A.copy(), B.copy(), C.copy()
    mask_base_ = mask_base.copy()
    tic_method2 = time.time()
    tic = time.time()
    result_stream2 = result_pre.copy()
    time_stream2 = []
    for mask_item in mask_list:
        mask_base_ += mask_item

        for i in range(1):
            A_, B_, C_ = iterationStream2(mask_item, X, A_, B_, C_, 2e-9)
        toc = time.time()
        rec, loss, PoF = metric(A_, B_, C_, X, mask_base_); result_stream2.append(PoF); time_stream2.append(time.time() - tic_method2)
        print ('loss:{}, PoF:{}, time:{}'.format(loss, PoF, toc-tic))
        tic = time.time()

    time_stream2 = time_pre + [i+time_pre[-1] for i in time_stream2]
    print ('finish streaming2 setting')
    print ()

    
    """
    row-wise streaming CPC
    """

    A_, B_, C_ = A.copy(), B.copy(), C.copy()
    mask_base_ = mask_base.copy()

    tic_cpc = time.time()
    tic = time.time()
    result_cpc = result_pre.copy()
    time_cpc = []
    for index, mask_item in enumerate(mask_list):
        mask_base_ += mask_item
        for i in range(1):
            A_, B_, C_ = iterationCPC(A_, B_, C_, A, B, C, mask_item, X, alpha= 5 * interval / (base + index*interval))
            A, B, C = A_.copy(), B_.copy(), C_.copy()
        toc = time.time()
        rec, loss, PoF = metric(A, B, C, X, mask_base_); result_cpc.append(PoF); time_cpc.append(time.time() - tic_cpc)
        print ('loss:{}, PoF:{}, time:{}'.format(loss, PoF, toc - tic))
        tic = time.time()

    time_cpc = time_pre + [i+time_pre[-1] for i in time_cpc]
    print ('finish CPC-ALS')

    """
    Plot
    """

    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.rcParams.update({"font.size":12})
    plt.plot(time_CPD, result_CPD, label="Oracle CPD")
    plt.plot(time_stream, result_stream, label="Initialization + EM-ALS")
    plt.plot(time_stream2, result_stream2, label="Initialization + SGD")
    plt.plot(time_cpc, result_cpc, label="Initialization + GO-CPC")
    plt.legend()
    plt.ylabel('PoF')
    plt.xlabel('Runing Time (s)')
    # plt.title('Synthetic Data')
    plt.title('FACE-3D Shots')
    plt.tight_layout()
    plt.show()
