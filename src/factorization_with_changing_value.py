from typing import Awaitable
import numpy as np
from scipy import linalg as la
import time
import pickle
import argparse
from model import cpd_als_iteration, GOCPTE_other2, GOCPT_other2, metric
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='synthetic', help="'synthetic'")
args = parser.parse_args()

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


# CPD-ALS
def run_CPDALS(X, factors, mask_tensor, mask_list):
    A, B, C = factors
    tic_CPD = time.time()
    result_CPD, time_CPD = [], []
    for mask, tensor in zip(mask_list, mask_tensor):
        X = X + X * mask * tensor
        A, B, C = cpd_als_iteration(X, [A, B, C])
        rec, loss, PoF = metric(X, [A, B, C]); result_CPD.append(PoF); time_CPD.append(time.time() - tic_CPD)
    print ('finish CPD-ALS')

    return result_CPD, time_CPD


# GOCPTE
def run_GOCPTE(X, factors, mask_tensor, mask_list):
    A, B, C = factors
    tic_GOCPTE = time.time()
    result_GOCPTE, time_GOCPTE = [], []
    for mask, tensor in zip(mask_list, mask_tensor):
        A, B, C = GOCPTE_other2(X * mask, mask, [A, B, C], alpha=1)
        X = X + X * mask * tensor
        rec, loss, PoF = metric(X, [A, B, C]); result_GOCPTE.append(PoF); time_GOCPTE.append(time.time() - tic_GOCPTE)
    print ('finish GOCPTE')

    return result_GOCPTE, time_GOCPTE


# GOCPT
def run_GOCPT(X, factors, mask_tensor, mask_list):
    A, B, C = factors
    tic_GOCPT = time.time()
    result_GOCPT, time_GOCPT = [], []
    for mask, tensor in zip(mask_list, mask_tensor):
        X = X + X * mask * tensor
        A, B, C = GOCPT_other2(X, [A, B, C], alpha=0.05)
        rec, loss, PoF = metric(X, [A, B, C]); result_GOCPT.append(PoF); time_GOCPT.append(time.time() - tic_GOCPT)
    print ('finish GOCPT')

    return result_GOCPT, time_GOCPT



if __name__ == '__main__':
    
    np.random.seed(np.random.randint(1e8))
    if args.data == 'synthetic':
        # configuration
        I, J, K, R = 100, 100, 100, 5
        A0 = np.random.random((I, R))
        B0 = np.random.random((J, R))
        C0 = np.random.random((K, R))
        X = np.einsum('ir,jr,kr->ijk',A0,B0,C0)
        amplitude, timestamp, sparsity, preIter = 0.01, 100, 0.95, 10

    elif args.data == 'Indian':
        # configuration
        import scipy.io as IO
        path = '../exp-data/Indian_pines_corrected.mat'
        data = IO.loadmat(path)
        X = data['indian_pines_corrected']
        I, J, K, R = *X.shape, 5
        amplitude, timestamp, sparsity, preIter = 0.005, 100, 0.95, 10

    else:
        print ('Dataset is not found!')
        exit()


    # show my the size
    print (I, J, K, R)

    # the initial tensor and the mask
    mask_list = []
    mask_tensor = []
    for i in range(timestamp):
        tmp = np.random.random(X.shape)
        mask_list.append(tmp >= sparsity)
        mask_tensor.append(np.random.random(X.shape) * amplitude)
    print ('finish data loading')


    """
    Preaparation
    """
    A = np.random.random((I,R))
    B = np.random.random((J,R))
    C = np.random.random((K,R))

    for i in range(preIter):
        A, B, C = cpd_als_iteration(X, [A, B, C])
    print ('finish preparation')


    # CPD-ALS
    result_CPD, time_CPD = run_CPDALS(X, [A.copy(), B.copy(), C.copy()], mask_tensor, mask_list)

    # GOCPTE
    result_GOCPTE, time_GOCPTE = run_GOCPTE(X, [A.copy(), B.copy(), C.copy()], mask_tensor, mask_list)

    # GOCPT
    result_GOCPT, time_GOCPT = run_GOCPT(X, [A.copy(), B.copy(), C.copy()], mask_tensor, mask_list)


    """
    report
    """
    model_name = ['CPD-ALS', 'GOCPTE', 'GOCPT']
    RESULT = [time_CPD, result_CPD, time_GOCPTE, result_GOCPTE, time_GOCPT, result_GOCPT]

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
        plt.title('Factorization with Changing Values (synthetic)')
    elif args.data == 'Indian':
        plt.title('Indian Pines')
    plt.tight_layout()
    plt.show()
