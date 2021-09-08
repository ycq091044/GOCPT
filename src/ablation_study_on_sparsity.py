import numpy as np
from scipy import linalg as la
import time
import pickle
from model import sparse_strategy_iteration, dense_strategy_iteration
from utils import metric


def run(X, mask, factors, total_iter, strategy):
    A, B, C = factors

    # setup
    tic_start = time.time()
    Result, Time = [], []

    if strategy == 'dense':
        func = dense_strategy_iteration
    elif strategy == 'sparse':
        func = sparse_strategy_iteration
    else:
        print ('strategy is not found!')
        exit()

    # iterations
    for _ in range(total_iter):
        A, B, C = func(mask * X, mask, [A, B, C])
        _, loss, PoF = metric(mask * X, [A, B, C], mask)
        Result.append(PoF); Time.append(time.time() - tic_start)
    return Result, Time


if __name__ == '__main__':

    """
    This scripts provides the comparison between sparse and dense strategy.
    We store the sparsity list in "sparse_list".
    """

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5)); plt.rcParams.update({"font.size":12})

    sparse_list = [0.05, 0.2, 0.8, 0.95, 0.98, 0.995]
    I, J, K, R, total_iter = 100, 100, 100, 5, 100

    for index, sparsity in enumerate(sparse_list):
        plt.subplot(2,len(sparse_list)//2,index+1)

        # results will be stored here
        time_sparse_list, time_dense_list = [], []
        result_sparse_list, result_dense_list = [], []

        # for each sparsity, run 5 random seends
        for j in range(5):

            print ('start {}-{}, sparsity {}'.format(index, j, sparsity))

            # construct the synthetic tensor
            A0 = np.random.random((I, R))
            B0 = np.random.random((J, R))
            C0 = np.random.random((K, R))
            X = np.einsum('ir,jr,kr->ijk',A0,B0,C0)
            # create the mask
            mask = np.random.random(X.shape) >= sparsity

            # random initialization
            A = np.random.random(A0.shape)
            B = np.random.random(B0.shape)
            C = np.random.random(C0.shape)

            # run sparse strategy
            result_sparse, time_sparse = run(X, mask, [A, B, C], total_iter, 'sparse')
            result_sparse_list.append(result_sparse); time_sparse_list.append(time_sparse)

            # run sense strategy
            result_dense, time_dense = run(X, mask, [A, B, C], total_iter, 'dense')
            result_dense_list.append(result_dense); time_dense_list.append(time_dense)

        plt.plot(np.mean(time_sparse_list, axis=0), np.mean(result_sparse_list, axis=0), color='firebrick', label="Sparse Strategy")
        plt.fill_between(np.mean(time_sparse_list, axis=0), np.mean(result_sparse_list, axis=0) - np.std(result_sparse_list, axis=0), \
                np.mean(result_sparse_list, axis=0) + np.std(result_sparse_list, axis=0), alpha=0.3, color='firebrick')
        plt.plot(np.mean(time_dense_list, axis=0), np.mean(result_dense_list, axis=0), color='grey', label="Dense Strategy")
        plt.fill_between(np.mean(time_dense_list, axis=0), np.mean(result_dense_list, axis=0) - np.std(result_dense_list, axis=0), \
                np.mean(result_dense_list, axis=0) + np.std(result_dense_list, axis=0), alpha=0.3, color='grey')

        if index == 0:
            plt.legend(loc='lower right')
        plt.ylabel('PoF'); plt.xlabel('Runing Time (s)')
        plt.title('Density: {}%'.format(100 - sparsity*100))
    plt.tight_layout()
    plt.show()
