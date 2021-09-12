import numpy as np
from scipy import linalg as la
import time
import pickle
import argparse
from model import em_als_other1, OnlineSGD_other1, GOCPT_other1, metric, sparse_strategy_iteration
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='synthetic', help="'synthetic'")
args = parser.parse_args()

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

# EM-ALS
def run_EMALS(X, factors, mask_base, mask_list):
    A, B, C = factors
    tic_EMALS = time.time()
    result_EMALS, time_EMALS = [], []
    for mask_item in mask_list:
        mask_base += mask_item
        A, B, C = em_als_other1(X * mask_base, mask_base, [A, B, C])
        rec, loss, PoF = metric(X * mask_base, [A, B, C], mask_base); result_EMALS.append(PoF); time_EMALS.append(time.time() - tic_EMALS)
    print ('finish EM-ALS')

    return result_EMALS, time_EMALS


# CPC-ALS
def run_CPCALS(X, factors, mask_base, mask_list):
    A, B, C = factors
    tic_CPCALS = time.time()
    result_CPCALS, time_CPCALS = [], []
    for mask_item in mask_list:
        mask_base += mask_item
        A, B, C = sparse_strategy_iteration(X * mask_base, mask_base, [A, B, C])
        rec, loss, PoF = metric(X * mask_base, [A, B, C], mask_base); result_CPCALS.append(PoF); time_CPCALS.append(time.time() - tic_CPCALS)
    print ('finish CPC-ALS')

    return result_CPCALS, time_CPCALS


# onlineSGD
def run_OnlineSGD(X, factors, mask_base, mask_list, lr=1e-3):
    A, B, C = factors
    tic_onlineSGD = time.time()
    result_onlineSGD, time_onlineSGD = [], []
    for mask_item in mask_list:
        mask_base += mask_item
        A, B, C = OnlineSGD_other1(X * mask_item, mask_item, [A, B, C], lr=lr)
        rec, loss, PoF = metric(X * mask_base, [A, B, C], mask_base); result_onlineSGD.append(PoF); time_onlineSGD.append(time.time() - tic_onlineSGD)
    print ('finish onlineSGD')

    return result_onlineSGD, time_onlineSGD


# GOCPTE
def run_GOCPTE(X, factors, mask_base, weight, interval, base, mask_list):
    A, B, C = factors
    tic_GOCPTE = time.time()
    result_GOCPTE, time_GOCPTE = [], []
    for index, mask_item in enumerate(mask_list):
        mask_base += mask_item
        A, B, C = GOCPT_other1(X * mask_item, mask_item, [A, B, C], alpha=weight * interval / (base + index*interval))
        rec, loss, PoF = metric(X * mask_base, [A, B, C], mask_base); result_GOCPTE.append(PoF); time_GOCPTE.append(time.time() - tic_GOCPTE)
    print ('finish GOCPTE')

    return result_GOCPTE, time_GOCPTE


# GOCPT
def run_GOCPT(X, factors, mask_base, weight, interval, base, mask_list):
    A, B, C = factors
    tic_GOCPT = time.time()
    result_GOCPT, time_GOCPT = [], []
    for index, mask_item in enumerate(mask_list):
        mask_base += mask_item
        A, B, C = GOCPT_other1(X * mask_base, mask_base, [A, B, C], alpha=weight * interval / 10 / (base + index*interval))
        rec, loss, PoF = metric(X * mask_base, [A, B, C], mask_base); result_GOCPT.append(PoF); time_GOCPT.append(time.time() - tic_GOCPT)
    print ('finish GOCPT')

    return result_GOCPT, time_GOCPT


if __name__ == '__main__':
    
    np.random.seed(np.random.randint(1e8))
    if args.data == 'synthetic':
        # configuration
        # I, J, K, R = 100, 100, 100, 5
        I, J, K, R = 100, 100, 100, 5
        A0 = np.random.random((I, R))
        B0 = np.random.random((J, R))
        C0 = np.random.random((K, R))
        X = np.einsum('ir,jr,kr->ijk',A0,B0,C0)
        base, interval, preIter, weight, total, lr = 0.02, 0.01, 10, 2, 1.0, 1e-5
        # base, interval, preIter, weight, total, lr = 0.005, 0.001, 10, 0.5, 0.05, 1e-3

    else:
        print ('Dataset is not found!')
        exit()


    # show my the size
    print (I, J, K, R)


    # the mask streaming
    base_ = base
    mask_list = []
    mask_base = np.random.random(X.shape) >= 1 - base
    mask_cumsum = mask_base
    for i in range(int((total-base)/interval)-1):
        mask_tmp = np.random.random(X.shape) * (1 - mask_cumsum) > 1 - interval / (1 - base)
        mask_list.append(mask_tmp)
        mask_cumsum += mask_tmp
        base += interval
    print ('finish data loading')


    """
    Preaparation with base mask
    """
    A = np.random.random((I,R))
    B = np.random.random((J,R))
    C = np.random.random((K,R))

    for i in range(preIter):
        A, B, C = sparse_strategy_iteration(X * mask_base, mask_base, [A, B, C])
    print ('finish preparation')


    # EM-ALS
    result_EMALS, time_EMALS = run_EMALS(X, [A.copy(), B.copy(), C.copy()], mask_base.copy(), mask_list)

    # CPC-ALS
    result_CPCALS, time_CPCALS = run_CPCALS(X, [A.copy(), B.copy(), C.copy()], mask_base.copy(), mask_list)

    # onlineSGD
    result_onlineSGD, time_onlineSGD = run_OnlineSGD(X, [A.copy(), B.copy(), C.copy()], mask_base.copy(), mask_list, lr)

    # GOCPTE
    result_GOCPTE, time_GOCPTE = run_GOCPTE(X, [A.copy(), B.copy(), C.copy()], mask_base.copy(), weight, interval, base, mask_list)

    # GOCPT
    result_GOCPT, time_GOCPT = run_GOCPT(X, [A.copy(), B.copy(), C.copy()], mask_base.copy(), weight, interval, base, mask_list)



    """
    report
    """
    model_name = ['EM-ALS', 'CPC-ALS', 'OnlineSGD', 'GOCPTE', 'GOCPT']
    RESULT = [time_EMALS, result_EMALS, time_CPCALS, result_CPCALS, time_onlineSGD, \
                    result_onlineSGD, time_GOCPTE, result_GOCPTE, time_GOCPT, result_GOCPT]

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

    plt.legend()
    plt.ylabel('PoF')
    plt.yscale('log')
    plt.xlabel('Time Step $t$')
    if args.data == 'synthetic':
        # plt.title('Online Tensor Completion (synthetic)')
        plt.title('Synthetic (start:{:.2}%, interval:{:.2}%, total:{:.2}%)'.format(base_*100,interval*100,total*100))
    plt.tight_layout()
    plt.show()
