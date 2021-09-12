import numpy as np
from scipy import linalg as la
import time
import pickle
import scipy.io as IO
import argparse
from model import sparse_strategy_iteration, em_als_completion, OnlineSGD, OLSTEC, GOCPTE_completion, GOCPT_completion, metric

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='synthetic', help="'synthetic' or 'Indian' or 'CovidHT'")
args = parser.parse_args()


# EMALS
def run_EMALS(X, factors, K, T, mask_base, mask_list):
    A, B, C = factors

    tic_EMALS = time.time()
    result_EMALS, time_EMALS = [], []
    for index, mask_item in enumerate(mask_list):
        mask_base = np.concatenate([mask_base, mask_item[:, :, np.newaxis]], axis=2)
        # call EMALS routine
        A, B, C = em_als_completion(X[:, :, :T+1]*mask_base, mask_base, [A, B, C], alpha=1, gamma=1.0); T += 1
        _, loss, PoF = metric(X[:,:,:T]*mask_base, [A, B, C], mask_base); result_EMALS.append(PoF); time_EMALS.append(time.time() - tic_EMALS)
    print ('finish EMALS')

    return result_EMALS, time_EMALS


# EMALS (exponential decay)
def run_EMALS_decay(X, factors, K, T, mask_base, mask_list):
    A, B, C = factors

    tic_EMALS_w = time.time()
    result_EMALS_w, time_EMALS_w = [], []
    for index, mask_item in enumerate(mask_list):
        mask_base = np.concatenate([mask_base, mask_item[:, :, np.newaxis]], axis=2)
        # call EMALS routine
        A, B, C = em_als_completion(X[:, :, :T+1]*mask_base, mask_base, [A, B, C], alpha=1, gamma=0.99); T += 1
        _, loss, PoF = metric(X[:,:,:T]*mask_base, [A, B, C], mask_base); result_EMALS_w.append(PoF); time_EMALS_w.append(time.time() - tic_EMALS_w)
    print ('finish EMALS (decay)')

    return result_EMALS_w, time_EMALS_w


# Online SGD
def run_OnlineSGD(X, factors, K, T, mask_base, mask_list):
    A, B, C = factors

    tic_OnlineSGD = time.time()
    result_OnlineSGD, time_OnlineSGD = [], []
    for index, mask_item in enumerate(mask_list):
        mask_base = np.concatenate([mask_base, mask_item[:, :, np.newaxis]], axis=2)
        # call OnlineSGD routine
        A, B, C = OnlineSGD(X[:,:,T:T+1] * mask_item[:, :, np.newaxis], mask_item[:, :, np.newaxis], [A, B, C], alpha=1e-10, index=index); T += 1
        _, loss, PoF = metric(X[:,:,:T]*mask_base, [A, B, C], mask_base); result_OnlineSGD.append(PoF), time_OnlineSGD.append(time.time() - tic_OnlineSGD)
    print ('finish OnlineSGD')

    return result_OnlineSGD, time_OnlineSGD


# OLSTEC
def run_OLSTEC(X, factors, K, T, mask_base, mask_list):
    A, B, C = factors
    tic_OLSTEC = time.time()
    Lambda = 0.98
    coeff = np.array([Lambda ** i for i in range(C.shape[0])])[::-1]
    RA = np.einsum('br,bz,cr,cz,c,abc->arz',B,B,C,C,coeff,mask_base,optimize=True)
    SA = np.einsum('ijk,jr,kr,k->ir',X[:,:,:T]*mask_base,B,C,coeff,optimize=True)
    RB = np.einsum('ar,az,cr,cz,c,abc->brz',A,A,C,C,coeff,mask_base,optimize=True)  
    SB = np.einsum('ijk,ir,kr,k->jr',X[:,:,:T]*mask_base,A,C,coeff,optimize=True)
    result_OLSTEC, time_OLSTEC = [], []

    for index, mask_item in enumerate(mask_list):
        mask_base = np.concatenate([mask_base, mask_item[:, :, np.newaxis]], axis=2)
        # call OLSTEC routine
        A, B, C, RA, RB, SA, SB = OLSTEC(X[:,:,T:T+1] * mask_item[:, :, np.newaxis], mask_item[:, :, np.newaxis], [A, B, C], RA, RB, SA, SB, mu=1e-9, Lambda=Lambda); T += 1
        _, loss, PoF = metric(X[:,:,:T]*mask_base, [A, B, C], mask_base); result_OLSTEC.append(PoF); time_OLSTEC.append(time.time() - tic_OLSTEC)
    print ('finish OLSTEC')

    return result_OLSTEC, time_OLSTEC


# GOCPTE
def run_GOCPTE(X, factors, K, T, weight, mask_base, mask_list):
    A, B, C = factors
    tic_GOCPTE = time.time()
    result_GOCPTE, time_GOCPTE = [], []
    for index, mask_item in enumerate(mask_list):
        mask_base = np.concatenate([mask_base, mask_item[:, :, np.newaxis]], axis=2)
        A, B, C = GOCPTE_completion(X[:,:,T:T+1] * mask_item[:, :, np.newaxis], mask_item[:, :, np.newaxis], [A, B, C], weight / T); T += 1
        _, loss, PoF = metric(X[:,:,:T]*mask_base, [A, B, C], mask_base); result_GOCPTE.append(PoF); time_GOCPTE.append(time.time() - tic_GOCPTE)
    print ('finish GOCPTE')

    return result_GOCPTE, time_GOCPTE
    
    
# GOCPT
def run_GOCPT(X, factors, K, T, weight, mask_base, mask_list):
    A, B, C = factors
    tic_GOCPT = time.time()
    result_GOCPT, time_GOCPT = [], []
    for index, mask_item in enumerate(mask_list):
        mask_base = np.concatenate([mask_base, mask_item[:, :, np.newaxis]], axis=2)
        A, B, C = GOCPT_completion(X[:,:,:T+1] * mask_base, mask_base, [A, B, C], weight / T); T += 1
        _, loss, PoF = metric(X[:,:,:T]*mask_base, [A, B, C], mask_base); result_GOCPT.append(PoF); time_GOCPT.append(time.time() - tic_GOCPT)
    print ('finish GOCPT')

    return result_GOCPT, time_GOCPT


if __name__ == '__main__':
    np.random.seed(np.random.randint(1e8))

    if args.data == 'synthetic':
        # configuration, K is the temporal mode
        I, J, K, R = 50, 50, 500, 5
        A0 = np.random.random((I, R))
        B0 = np.random.random((J, R))
        C0 = np.random.random((K, R))
        X = np.einsum('ir,jr,kr->ijk',A0,B0,C0)
        base, sparsity, preIter, weight = 0.1, 0.98, 10, 0.5

    elif args.data == 'Indian':
        import scipy.io as IO
        path = '../exp-data/Indian_pines_corrected.mat'
        data = IO.loadmat(path)
        X = data['indian_pines_corrected']
        I, J, K, R = *X.shape, 5
        base, sparsity, preIter, weight = 0.1, 0.98, 10, 0.5
        
    else:
        print ('the data is not found! Please check')
        exit()


    # show my the size
    print (I, J, K, R)

    # prepare the mask (one slice at a time)
    T = int(X.shape[2] * base)
    mask_base = np.random.random((*X.shape[:2],T)) >= sparsity
    mask_list = []
    for i in range(X.shape[2] - T):
        mask_tmp = np.random.random(X.shape[:2]) >= sparsity
        mask_list.append(mask_tmp)
    print ('finish data loading')
    # print ()


    """
    Preparation with base mask
    """
    # initialization
    A = np.random.random((I,R))
    B = np.random.random((J,R))
    C = np.random.random((T,R))

    for i in range(preIter):
        A, B, C = sparse_strategy_iteration(X[:,:,:T] * mask_base, mask_base, [A, B, C])
    print ('finish preparation')


    
    # run EMALS
    result_EMALS, time_EMALS = run_EMALS(X, [A.copy(), B.copy(), C.copy()], K, T, mask_base.copy(), mask_list)

    # run EMALS (decay)
    result_EMALS_w, time_EMALS_w = run_EMALS_decay(X, [A.copy(), B.copy(), C.copy()], K, T, mask_base.copy(), mask_list)

    # run Online SGD
    result_OnlineSGD, time_OnlineSGD = run_OnlineSGD(X, [A.copy(), B.copy(), C.copy()], K, T, mask_base.copy(), mask_list)

    # run OLSTEC
    result_OLSTEC, time_OLSTEC = run_OLSTEC(X, [A.copy(), B.copy(), C.copy()], K, T, mask_base.copy(), mask_list)

    # run GOCPTE
    result_GOCPTE, time_GOCPTE = run_GOCPTE(X, [A.copy(), B.copy(), C.copy()], K, T, weight, mask_base.copy(), mask_list)

    # run GOCPT
    result_GOCPT, time_GOCPT = run_GOCPT(X, [A.copy(), B.copy(), C.copy()], K, T, weight, mask_base.copy(), mask_list)



    """
    report
    """
    model_name = ['EM-ALS', 'EM-ALS (decay)', 'OnlineSGD', 'OLSTEC', 'GOCPTE', 'GOCPT']
    RESULT = [time_EMALS, result_EMALS, time_EMALS_w, result_EMALS_w, time_OnlineSGD, result_OnlineSGD, \
                    time_OLSTEC, result_OLSTEC, time_GOCPTE, result_GOCPTE, time_GOCPT, result_GOCPT]

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
        # plt.yscale('log')
        plt.xlabel('Time Step $t$')
        plt.title('Streaming Tensor Completion (synthetic)')
        plt.tight_layout()
        plt.show()
    
    else:

        for index in range(len(RESULT) // 2):
            print ('Model: {}, Time: {}, Avg. PoF: {}'.format(model_name[index], RESULT[2*index][-1], np.mean(RESULT[2*index+1][1:])))

