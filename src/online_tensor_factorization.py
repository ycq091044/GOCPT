import numpy as np
from scipy import linalg as la
import time
import pickle
import scipy.io as IO
import argparse
from model import cpd_als_iteration, OnlineCPD, MAST, SDT, BiSVD, RLST, CPStream, CPStream2, \
    GOCPT_factorization, GOCPTE_factorization, metric

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='synthetic', help="'synthetic' or 'FACE' or 'GCSS'")
args = parser.parse_args()


# OnlineCPD
def run_OnlineCPD(X, factors, K, T):
    A, B, C, T_ = *factors, T
    P1 = np.einsum('ijk,jr,kr->ri',X[:, :, :T],B,C,optimize=True)
    P2 = np.einsum('ijk,ir,kr->rj',X[:, :, :T],A,C,optimize=True)
    Q1 = np.einsum('ir,im,jr,jm->rm',B,B,C,C,optimize=True)
    Q2 = np.einsum('ir,im,jr,jm->rm',A,A,C,C,optimize=True)

    tic_onlineCPD = time.time()
    result_onlineCPD, time_onlineCPD = [], []
    for index in range(K - T_):
        # call OnlineCPD routine
        A, B, C, P1, P2, Q1, Q2 = OnlineCPD(X[:, :, T:T+1], [A, B, C], [P1, P2], [Q1, Q2]); T += 1
        _, loss, PoF = metric(X[:,:,:T], [A, B, C]); result_onlineCPD.append(PoF); time_onlineCPD.append(time.time() - tic_onlineCPD)
    print ('finish onlineCPD')

    return result_onlineCPD, time_onlineCPD


# MAST + ADMM
def run_MAST(X, factors, K, T):
    A, B, C, T_ = *factors, T
    tic_MAST = time.time()
    result_MAST, time_MAST = [], []
    for index in range(K - T_):
        # call MAST routine
        A, B, C = MAST(X[:, :, T:T+1], [A, B, C], 1, [1/15, 1/15, 1/15], total_iter=20, phi=1.05, etamax=1, etaInit=1e-5, tol=1e-5); T += 1
        _, loss, PoF = metric(X[:,:,:T], [A, B, C]); result_MAST.append(PoF); time_MAST.append(time.time() - tic_MAST)
    print ('finish MAST')

    return result_MAST, time_MAST


# SDT-EW
def run_SDT(X, factors, K, T):
    A, B, C, T_ = *factors, T 
    tic_SDT = time.time()
    result_SDT, time_SDT = [], []

    # preparation
    gamma = 0.99
    coeff = [gamma ** i for i in range(T)][::-1]
    X_pre = np.einsum('ijk,k->ijk',X[:, :, :T],coeff,optimize=True).reshape(A.shape[0]*B.shape[0], -1)
    U, S, V = BiSVD(X_pre, A.shape[1])
    E = U @ S
    C = np.einsum('kr,k->kr',C,coeff,optimize=True)
    W = la.inv(E.T @ E) @ E.T @ la.khatri_rao(B,A)
    Wi = la.inv(W)

    for index in range(K - T_):
        # call SDT routine
        A, B, C, W, Wi, V, S, U = SDT(X[:, :, T:T+1],[A,B,C],W,Wi,V,S,U,gamma); T += 1
        _, loss, PoF = metric(X[:,:,:T], [A, B, C]); result_SDT.append(PoF); time_SDT.append(time.time() - tic_SDT)
    print ('finish SDT-EW')
    
    return result_SDT, time_SDT


# RLST-EW
def run_RLST(X, factors, K, T):
    A, B, C, T_ = *factors, T
    tic_RLST = time.time()
    result_RLST, time_RLST = [], []

    # preparation
    gamma = 0.995
    X_pre = X[:, :, :T].reshape(A.shape[0]*B.shape[0], -1)
    R1 = X_pre @ C
    P1 = C.T @ C
    Z1 = la.inv(R1.T @ R1) @ R1.T
    Q1 = la.inv(P1)

    for index in range(K - T_):
        # call RLST routine
        A, B, C, P1, Q1, R1, Z1 = RLST(X[:, :, T:T+1],[A,B,C],P1,Q1,R1,Z1,T_,gamma); T += 1
        _, loss, PoF = metric(X[:,:,:T], [A, B, C]); result_RLST.append(PoF); time_RLST.append(time.time() - tic_RLST)
        tic = time.time()
    print ('finish RLST-EW')

    return result_RLST, time_RLST


# CPStream
def run_CPStream(X, factors, K, T):
    A, B, C, T_ = *factors, T
    tic_CPStream = time.time()
    result_CPStream, time_CPStream = [], []

    mu = 0.99
    coeff = np.array([mu ** i for i in range(C.shape[0])])[::-1]
    G = np.einsum('kr,km,k->rm',C,C,coeff,optimize=True)
    
    for index in range(K - T_):
        # call CPStream routine
        A, B, C = CPStream2(X[:, :, T:T+1], [A, B, C], total_iter=20, gamma=0.99, tol=1e-5); T += 1
        # A, B, C = CPStream(X[:, :, T:T+1], [A, B, _], G, mu=mu, iteration=20, tol=1e-5)
        _, loss, PoF = metric(X[:,:,:T], [A, B, C]); result_CPStream.append(PoF); time_CPStream.append(time.time() - tic_CPStream)
    print ('finish CP Stream')

    return result_CPStream, time_CPStream


# Our GOCPT_E
def run_GOCPTE_factorization(X, factors, K, T):
    A, B, C, T_ = *factors, T
    tic_GOCPTE = time.time()
    result_GOCPTE, time_GOCPTE = [], []
    for index in range(K - T_):
        # call GOCPT_factorization routine
        A, B, C = GOCPTE_factorization(X[:, :, T:T+1], [A, B, C], min(1,weight / (base * K + index))); T += 1
        _, loss, PoF = metric(X[:,:,:T], [A, B, C]); result_GOCPTE.append(PoF); time_GOCPTE.append(time.time() - tic_GOCPTE)
    print ('finish GOCPT_E')

    return result_GOCPTE, time_GOCPTE


# Our GOCPT
def run_GOCPT_factorization(X, factors, K, T):
    A, B, C, T_ = *factors, T
    tic_GOCPT = time.time()
    result_GOCPT, time_GOCPT = [], []
    for index in range(K - T_):
        # call GOCPT_factorization routine
        A, B, C = GOCPT_factorization(X[:, :, :T+1], [A, B, C], 2 / (base * K + index)); T += 1
        _, loss, PoF = metric(X[:,:,:T], [A, B, C]); result_GOCPT.append(PoF); time_GOCPT.append(time.time() - tic_GOCPT)
    print ('finish GOCPT')

    return result_GOCPT, time_GOCPT



if __name__ == '__main__':
    np.random.seed(np.random.randint(1e8))

    if args.data == 'synthetic':
        # configuration, K is the temporal mode
        I, J, K, R = 100, 100, 500, 5
        A0 = np.random.random((I, R))
        B0 = np.random.random((J, R))
        C0 = np.random.random((K, R))
        X = np.einsum('ir,jr,kr->ijk',A0,B0,C0)
        base, pre_iter, weight = 0.1, 10, 2

    elif args.data == 'FACE':
        X = pickle.load(open('../exp-data/FACE-3D.pkl', 'rb'))
        I, J, K, R = *X.shape, 5
        base, pre_iter, weight = 0.1, 10, 200
        
    elif args.data == 'GCSS':
        X = pickle.load(open('../exp-data/GCSS.pkl', 'rb'))
        I, J, K, R = *X.shape, 5
        base, pre_iter, weight = 0.1, 10, 200

    else:
        print ('Dataset is not found!')
        exit()


    # show my the size
    print (I, J, K, R)

    # the initial tensor and the mask
    T = int(X.shape[2] * base)
    print ('finish data loading')


    """
    Initalize with preparation data
    """
    A = np.random.random((I,R))
    B = np.random.random((J,R))
    C = np.random.random((T,R))

    for i in range(pre_iter):
        A, B, C = cpd_als_iteration(X[:,:,:T], [A, B, C])
    print ('finish preparation')
    print ('------------------')

    # onlineCPD
    result_onlineCPD, time_onlineCPD = run_OnlineCPD(X, [A.copy(), B.copy(), C.copy()], K, T)

    # MAST
    result_MAST, time_MAST = run_MAST(X, [A.copy(), B.copy(), C.copy()], K, T)

    # # SDT
    # result_SDT, time_SDT = run_SDT(X, [A.copy(), B.copy(), C.copy()], K, T)

    # RLST
    result_RLST, time_RLST = run_RLST(X, [A.copy(), B.copy(), C.copy()], K, T)

    # CPStream
    result_CPStream, time_CPStream = run_CPStream(X, [A.copy(), B.copy(), C.copy()], K, T)

    # GOCPT_E
    result_GOCPTE, time_GOCPTE = run_GOCPTE_factorization(X, [A.copy(), B.copy(), C.copy()], K, T)

    # GOCPT
    result_GOCPT, time_GOCPT = run_GOCPT_factorization(X, [A.copy(), B.copy(), C.copy()], K, T)



    """
    report
    """
    model_name = ['OnlineCPD', 'MAST', 'RLST', 'CPStream', 'GOCPTE', 'GOCPT']
    RESULT = [time_onlineCPD, result_onlineCPD, time_MAST, result_MAST, time_RLST, result_RLST, time_CPStream, result_CPStream, \
                time_GOCPTE, result_GOCPTE, time_GOCPT, result_GOCPT]

    if args.data == 'synthetic':

        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size":9})

        color = ['#1b9e77',
                '#d95f02',
                '#7570b3',
                '#e7298a',
                '#66a61e',
                '#e6ab02'][::-1]

        for index in range(len(RESULT) // 2):
            if index == 2:
                plt.plot(np.arange(len(RESULT[2*index]))[:-1], RESULT[2*index+1][1:], \
                        label="{} (time: {:.4}s)".format(model_name[index], RESULT[2*index][-1]), color=color[-6+index])
            else:
                plt.plot(np.arange(len(RESULT[2*index])), RESULT[2*index+1], \
                        label="{} (time: {:.4}s)".format(model_name[index], RESULT[2*index][-1]), color=color[-6+index])

        plt.legend(loc='lower right')
        plt.ylabel('PoF')
        plt.yscale('log')
        plt.xlabel('Time Step $t$')
        plt.title('Online Tensor Factorization (synthetic)')
        plt.tight_layout()
        plt.show()
    
    else:

        for index in range(len(RESULT) // 2):
            print ('Model: {}, Time: {}, Avg. PoF: {}'.format(model_name[index], RESULT[2*index][-1], np.mean(RESULT[2*index+1][1:])))

