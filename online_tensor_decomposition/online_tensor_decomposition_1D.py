from typing import Awaitable
import numpy as np
from scipy import linalg as la
import time
import pickle
import scipy.io as IO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='synthetic', help="'synthetic' or 'FACE'")
args = parser.parse_args()

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

def iterationPre(A1, A2, A3, T, reg=1e-5):
    eye = np.eye(A1.shape[1])
    A1 = optimize(np.einsum('ir,im,jr,jm->rm',A2,A2,A3,A3) + reg * eye, np.einsum('ijk,jr,kr->ri',T,A2,A3)).T
    A2 = optimize(np.einsum('ir,im,jr,jm->rm',A1,A1,A3,A3) + reg * eye, np.einsum('ijk,ir,kr->rj',T,A1,A3)).T
    A3 = optimize(np.einsum('ir,im,jr,jm->rm',A1,A1,A2,A2) + reg * eye, np.einsum('ijk,ir,jr->rk',T,A1,A2)).T
    return A1, A2, A3

def iterationStreamOnlineCPD(T, A, B, C, P1, P2, Q1, Q2, reg=1e-5):
    """
    Input:
        - T: I x J x K
        - |(x - Adiag(c)B.T)| + |X - [A, B, C]| + beta * reg
    """

    # get c
    c = optimize(np.einsum('ir,im,jr,jm->rm',A,A,B,B,optimize=True), np.einsum('ijk,ir,jr->rk',T[:,:,-1:],A,B,optimize=True)).T

    eye = np.eye(R)
    C = np.concatenate([C, c], axis=0)

    P1 += np.einsum('ijk,jr,kr->ri',T[:,:,-1:],B,c,optimize=True)
    P2 += np.einsum('ijk,ir,kr->rj',T[:,:,-1:],A,c,optimize=True)
    Q1 += np.einsum('ir,im,jr,jm->rm',B,B,c,c,optimize=True)
    Q2 += np.einsum('ir,im,jr,jm->rm',A,A,c,c,optimize=True)
    A = optimize(Q1 + reg * eye, P1).T
    B = optimize(Q2 + reg * eye, P2).T

    return A, B, C, P1, P2, Q1, Q2

def iterationStreamGOCPC(T, A, B, C, alpha=1, iteration=1, reg=1e-5, gamma=0.995):
    """
    Input
        - T: I x J
        - |(x - Adiag(c)B.T)| + alpha * |[A, B, C] - [A0, B0, C0]| + beta * reg
    """
    A1, A2, A3 = A.copy(), B.copy(), C.copy()
    # get c

    coeff = np.array([gamma ** i for i in range(1, C.shape[0]+1)])[::-1]

    eye = np.eye(R)
    for i in range(iteration):
        reg *= 0.9 + 0.1 * np.random.random()
        c = optimize(np.einsum('ir,im,jr,jm->rm',A,A,B,B,optimize=True), np.einsum('ijk,ir,jr->rk',T,A,B,optimize=True)).T
        A = optimize(np.einsum('ir,im,jr,jm->rm',B,B,c,c,optimize=True) + alpha * np.einsum('ir,im,jr,jm,j->rm',B,B,C,C,coeff,optimize=True) + reg * eye, \
                    np.einsum('ijk,jr,kr->ri',T,B,c,optimize=True) + alpha * np.einsum('kr,ir,im,jr,jm,j->mk',A1,A2,B,A3,C,coeff,optimize=True)).T
        B = optimize(np.einsum('ir,im,jr,jm->rm',A,A,c,c,optimize=True) + alpha * np.einsum('ir,im,jr,jm,j->rm',A,A,C,C,coeff,optimize=True) + reg * eye, \
                    np.einsum('ijk,ir,kr->rj',T,A,c,optimize=True) + alpha * np.einsum('kr,ir,im,jr,jm,j->mk',A2,A1,A,A3,C,coeff,optimize=True)).T
        C = optimize(np.einsum('ir,im,jr,jm->rm',A,A,B,B,optimize=True) + reg * eye, \
                    np.einsum('kr,ir,im,jr,jm->mk',A3,A1,A,A2,B,optimize=True)).T
    return A, B, np.concatenate([C, c], axis=0)


def iterationStreamMAST(T, A, B, C, alpha, alphaN, iteration=1, phi=1.05, etamax=1, etaInit=1e-5):
    """
    KDD2017 settings
    alphaN = [1/5N, 1/5N, 1/5N] = [1/15, 1/15, 1/15]
    phi = 1.05; eta = 1e-4; etamax=1e6 
    """
    # initialize ZA, ZB, ZC and YA, YB, YC, A1, A2, A3
    K, R = C.shape
    ZA, YA = np.zeros(A.shape), np.zeros(A.shape)
    ZB, YB = np.zeros(B.shape), np.zeros(B.shape)
    ZC, YC = np.zeros((K+1, R)), np.zeros((K+1, R))
    c = np.random.random((1,R))
    A1, A2, A3 = A.copy(), B.copy(), C.copy()
    eta = etaInit
    eye = np.eye(R)
    a1, a2, a3 = alphaN
    for k in range(iteration):
        eta = min(eta * phi, etamax)
        # update factors
        C = la.solve(np.einsum('ir,im,jr,jm->rm',A,A,B,B,optimize=True) + eta * eye, \
                np.einsum('kr,ir,im,jr,jm->mk',A3,A1,A,A2,B,optimize=True) + eta * ZC[:-1,:].T + YC[:-1,:].T).T
        c = la.solve(np.einsum('ir,im,jr,jm->rm',A,A,B,B,optimize=True), \
                np.einsum('ijk,ir,jr->rk',T,A,B,optimize=True) + eta * ZC[-1:,:].T + YC[-1:,:].T).T
        A = la.solve(np.einsum('ir,im,jr,jm->rm',B,B,c,c,optimize=True) + alpha * np.einsum('ir,im,jr,jm->rm',B,B,C,C,optimize=True) + eta * eye, \
                np.einsum('ijk,jr,kr->ri',T,B,c,optimize=True) + alpha * np.einsum('kr,ir,im,jr,jm->mk',A1,A2,B,A3,C,optimize=True)+ eta * ZA.T + YA.T).T
        B = la.solve(np.einsum('ir,im,jr,jm->rm',A,A,c,c,optimize=True) + alpha * np.einsum('ir,im,jr,jm->rm',A,A,C,C,optimize=True) + eta * eye, \
                np.einsum('ijk,ir,kr->rj',T,A,c,optimize=True) + alpha * np.einsum('kr,ir,im,jr,jm->mk',A2,A1,A,A3,C,optimize=True)+ eta * ZB.T + YB.T).T
        # update auxiliary factors
        u, s, vh = la.svd(A - YA / eta, full_matrices = False)
        ZA = u @ np.diag(s * (s >= a1 / eta)) @ vh
        YA += eta * (ZA - A)
        u, s, vh = la.svd(B - YB / eta, full_matrices = False)
        ZB = u @ np.diag(s * (s >= a2 / eta)) @ vh
        YB += eta * (ZB - B)
        u, s, vh = la.svd(np.concatenate([C, c], axis=0) - YC / eta, full_matrices = False)
        ZC = u @ np.diag(s * (s >= a3 / eta)) @ vh
        YC += eta * (ZC - np.concatenate([C, c], axis=0))

    return A, B, np.concatenate([C, c], axis=0)

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
    Input
        - T:    I x J x 1
        - mask: I x J 
        - |omega * (x - Adiag(c)B.T)| + alpha * |[A,B,C] - [A_o, B_o, C_o]| + beta * reg
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
    np.random.seed(12345)

    if args.data == 'synthetic':
        # configuration, K is the temporal mode
        I, J, K, R = 50, 50, 500, 5
        A0 = np.random.random((I, R))
        B0 = np.random.random((J, R))
        C0 = np.random.random((K, R))
        X = np.einsum('ir,jr,kr->ijk',A0,B0,C0)
        base, preIter, weight = 0.1, 10, 10

    elif args.data == 'FACE':
        # X = pickle.load(open('../exp-data/FACE-3D.pkl', 'rb'))
        X = pickle.load(open('../exp-data/google.pkl', 'rb'))
        # path = '../exp-data/Indian_pines_corrected.mat'
        # data = IO.loadmat(path)
        # X = data['indian_pines_corrected'][:, :, :100]
        I, J, K, R = *X.shape, 5
        base, preIter, weight = 0.1, 10, 200
    
    elif args.data == 'IQVIA1':
        path = '../exp-data/iqvia_tensor_data_state_2018.pickle'
        data = pickle.load(open(path, 'rb'))
        X = np.concatenate([data[0], data[1]], axis=2) # 49 x 22 x 52 x 12
        X = X.sum(axis=3)
        I, J, K, R = *X.shape, 5
        base, preIter, weight = 0.1, 10, 200
        print (I,J,K)
    else:
        print ('Dataset is not found!')
        exit()

    # the initial tensor and the mask
    T = int(X.shape[2] * base)
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
    #     rec, loss, PoF = metric(A, B, C, X)
    #     if i >= preIter: result_CPD.append(PoF)
    #     print ('loss:{}, PoF: {}, time: {}'.format(loss, PoF, toc - tic))
    #     tic = time.time()
    # print ('finish CPD')
    # print ()

    """
    Preaparation with base mask
    """
    # initialization
    A = np.random.random((I,R))
    B = np.random.random((J,R))
    C = np.random.random((T,R))

    tic = time.time()
    result_pre = []
    for i in range(preIter):
        A, B, C = iterationPre(A, B, C, X[:,:,:T])
        toc = time.time()
        rec, loss, PoF = metric(A, B, C, X[:,:,:T]); result_pre.append(PoF)
        print ('loss:{}, PoF:{}, time:{}'.format(loss, PoF, toc-tic))
        tic = time.time()
    print ('finish preparation')
    print ()


    """
    Common streaming setting (Online CPD)
    """
    A_, B_, C_, T_ = A.copy(), B.copy(), C.copy(), T
    rec = np.einsum('ir,jr,kr->ijk',A_,B_,C_)

    P1 = np.einsum('ijk,jr,kr->ri',X[:, :, :T_],B_,C_,optimize=True)
    P2 = np.einsum('ijk,ir,kr->rj',X[:, :, :T_],A_,C_,optimize=True)
    Q1 = np.einsum('ir,im,jr,jm->rm',B_,B_,C_,C_,optimize=True)
    Q2 = np.einsum('ir,im,jr,jm->rm',A_,A_,C_,C_,optimize=True)

    tic_onlineCPD = time.time()
    tic = time.time()
    # result_stream = result_pre.copy()
    result_stream, time_stream = [], []
    for index in range(K - T):
        A_,B_,C_, P1, P2, Q1, Q2 = iterationStreamOnlineCPD(X[:, :, :T_+1], A_, B_, C_, P1, P2, Q1, Q2)
        T_ += 1; toc = time.time()
        rec, loss, PoF = metric(A_, B_, C_, X[:,:,:T_]); result_stream.append(PoF); time_stream.append(time.time() - tic_onlineCPD)
        print ('loss:{}, PoF:{}, time:{}'.format(loss, PoF, toc-tic))
        tic = time.time()
    print ('finish onlineCPD')
    print ()


    """
    Our GO-CPC
    """
    A_, B_, C_, T_ = A.copy(), B.copy(), C.copy(), T

    tic_GOCPC = time.time()
    tic = time.time()
    # result_stream2 = result_pre.copy()
    result_stream2, time_stream2 = [], []
    for index in range(K - T):
        A_,B_,C_ = iterationStreamGOCPC(X[:, :, T_:T_+1], A_, B_, C_, min(1, weight / (base * K + index)))
        T_ += 1; toc = time.time()
        rec, loss, PoF = metric(A_, B_, C_, X[:,:,:T_]); result_stream2.append(PoF); time_stream2.append(time.time() - tic_GOCPC)
        print ('loss:{}, PoF:{}, time:{}'.format(loss, PoF, toc-tic))
        tic = time.time()
    print ('finish GO-CPC')
    print ()


    """
    Our GO-CPC 2
    """
    A_, B_, C_, T_ = A.copy(), B.copy(), C.copy(), T

    tic_GOCPC2 = time.time()
    tic = time.time()
    # result_stream2 = result_pre.copy()
    result_stream22, time_stream22 = [], []
    for index in range(K - T):
        A_,B_,C_ = iterationStreamGOCPC(X[:, :, T_:T_+1], A_, B_, C_, min(1, weight / (base * K + index)), 3)
        T_ += 1; toc = time.time()
        rec, loss, PoF = metric(A_, B_, C_, X[:,:,:T_]); result_stream22.append(PoF); time_stream22.append(time.time() - tic_GOCPC2)
        print ('loss:{}, PoF:{}, time:{}'.format(loss, PoF, toc-tic))
        tic = time.time()
    print ('finish GO-CPC')
    print ()


    # """
    # row-wise streaming CPC
    # """

    # A_, B_, C_ = A.copy(), B.copy(), C.copy()
    # T_ = T
    # mask_base_ = mask_base.copy()

    # tic = time.time()
    # # result_cpc = result_pre.copy()
    # result_cpc = []
    # for index, mask_item in enumerate(mask_list):
    #     mask_base_ = np.concatenate([mask_base_, mask_item[:, :, np.newaxis]], axis=2)
    #     for i in range(1):
    #         A_, B_, C_ = iterationCPC(mask_item, X[:,:,T_:T_+1], A_, B_, C_, A, B, C, 50 / (base * K + index))
    #         A, B, C = A_.copy(), B_.copy(), C_.copy()
    #     T_ += 1; toc = time.time()
    #     rec, loss, PoF = metric(A, B, C, X[:,:,:T_], mask_base_); result_cpc.append(PoF)
    #     print ('loss:{}, PoF:{}, time:{}'.format(loss, PoF, toc - tic))
    #     tic = time.time()

    # print ('finish CPC-ALS')


    """
    MAST ADMM
    """
    A_, B_, C_, T_ = A.copy(), B.copy(), C.copy(), T

    tic_MAST = time.time()
    tic = time.time()
    # result_stream3 = result_pre.copy()
    result_stream3, time_stream3 = [], []
    for index in range(K - T):
        A_, B_, C_ = iterationStreamMAST(X[:, :, T_:T_+1], A_, B_, C_, 1, [1/15, 1/15, 1/15])
        T_ += 1; toc = time.time()
        rec, loss, PoF = metric(A_, B_, C_, X[:,:,:T_]); result_stream3.append(PoF); time_stream3.append(time.time() - tic_MAST)
        print ('loss:{}, PoF:{}, time:{}'.format(loss, PoF, toc-tic))
        tic = time.time()
    print ('finish MAST')
    print ()


    """
    MAST ADMM 2
    """
    A_, B_, C_, T_ = A.copy(), B.copy(), C.copy(), T

    tic_MAST2 = time.time()
    tic = time.time()
    # result_stream3 = result_pre.copy()
    result_stream32, time_stream32 = [], []
    for index in range(K - T):
        A_, B_, C_ = iterationStreamMAST(X[:, :, T_:T_+1], A_, B_, C_, 1, [1/15, 1/15, 1/15], 3)
        T_ += 1; toc = time.time()
        rec, loss, PoF = metric(A_, B_, C_, X[:,:,:T_]); result_stream32.append(PoF); time_stream32.append(time.time() - tic_MAST2)
        print ('loss:{}, PoF:{}, time:{}'.format(loss, PoF, toc-tic))
        tic = time.time()
    print ('finish MAST')
    print ()


    """
    plot
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":12})

    plt.figure(1)
    # plt.plot(np.array(result_CPD), label="Oracle CPD")
    plt.plot(np.array(time_stream), np.array(result_stream), label="OnlineCPD with 1 cycle")
    # plt.plot(np.array(result_stream2), label="CPD (one base tensor) + Our OnlineCPD:0.0023s/iter")
    plt.plot(np.array(time_stream3), np.array(result_stream3), label="MAST with 1 cycle")
    plt.plot(np.array(time_stream32), np.array(result_stream32), label="MAST with 3 cycles")
    plt.plot(np.array(time_stream2), np.array(result_stream2), label="GO-CPC with 1 cycle")
    plt.plot(np.array(time_stream22), np.array(result_stream22), label="GO-CPC with 3 cycles")
    plt.legend()
    plt.ylabel('PoF')
    plt.yscale('log')
    plt.xlabel('Runing Time (s)')
    if args.data == 'synthetic':
        plt.title('Synthetic Data')
    elif args.data == 'FACE':
        plt.title('FACE-3D Shots')
    elif args.data == 'IQVIA1':
        plt.title('IQVIA 2018')
    plt.tight_layout()
    plt.show()
