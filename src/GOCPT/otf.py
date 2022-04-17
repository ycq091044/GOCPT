import numpy as np
from regex import P
from .utils import cpd_als_iteration, OnlineCPD_iteration, get_new_LHS_RHS
from .metrics import PoF

def random_factors(X, R):
    """
    This function is to obtain the random initialization of the factors given the tensor
    INPUT:
        - <tensor> X: this is the input tensor of size (I1, I2, I3, ..., In)
        - <int> R: the target rank
    OUTPUT:
        - <matrix> A1, A2, ..., An: the random factor matrices of size (In, R) 
    """
    # obtain the size of the tensor
    In = X.shape
    # randomly initialize each factor
    random_factors = [np.random.random((i, R)) for i in In]
    return random_factors


def cpd(X, R, iters=None, verbose=False):
    """
    This function is to obtain the preparation factors for the initial tensor
    INPUT:
        - <tensor> X: this is the input tensor of size (I1, I2, I3, ..., In)
        - <int> R: the target rank
        - iter: the number of iterations or optimize under converge
    OUTPUT:
        - <matrix> A1, A2, ..., An: the preparation factor matrices of size (In, R)
        - <list> pof_score_list: contains the PoF metric during iterations 
    """
     
    factors = random_factors(X, R)
    pof_score_list = []

    if iters is not None:
        for i in range(iters):
            factors, run_time = cpd_als_iteration(X, factors)
            pof_score = PoF(X, factors)
            if verbose and (i % 10 == 0):
                print ("{}-th iters, PoF: {}, time: {}s".format(i, pof_score, run_time))
            pof_score_list.append(pof_score)
    else:
        pof_score = PoF(X, factors)
        max_iters = 50
        for i in range(max_iters):
            factors, run_time = cpd_als_iteration(X, factors)
            new_pof_score = PoF(X, factors)
            if verbose and (i % 10 == 0):
                print ("{}-th iters, PoF: {}, time: {}s".format(i, PoF(X, factors, run_time)))
            # whether early stop
            if (new_pof_score - pof_score) / (pof_score + 1e-8) < 1e-5:
                break
            else:
                pof_score = new_pof_score
            pof_score_list.append(pof_score)
    return factors, pof_score_list


def draw_pof(pof_score):
    import matplotlib.pyplot as plt
    _ = plt.plot(pof_score)
    _ = plt.xlabel('Iterations')
    _ = plt.ylabel('PoF metric')
    plt.show()



class OnlineCPD():
    def __init__(self, R):
        self.P = None
        self.Q = None
        self.factors = None
        self.R = R
        self.counter = 0
        self.X = None
        self.pof_update_list = []
    
    def update(self, X, verbose=False):
        # for calculating pof, we store X
        self.X = np.concatenate([self.X, X], 2)
        self.factors, self.P, self.Q, run_time = \
                        OnlineCPD_iteration(X, self.factors, self.P, self.Q)
        pof_score = PoF(self.X, self.factors)
        if verbose:
            print ("{}-th update, PoF: {}, run_time: {}s".\
                            format(self.counter, pof_score, run_time))
        self.pof_update_list.append(pof_score)
        self.counter += 1

    def initialize(self, X):
        # update X and counter
        self.X = X
        self.counter += 1

        # update factors, P and Q
        In = X.shape
        factors, _ = cpd(X, self.R)
        self.factors = factors
        self.P = [None for _ in range(len(In))]
        self.Q = [None for _ in range(len(In))]

        for i in range(len(factors)-1):
            Qn, Pn = get_new_LHS_RHS(X, factors, i)
            self.P[i] = Pn; self.Q[i] = Qn

        pof = PoF(self.X, self.factors)
        self.pof_update_list.append(pof)
        print ('Init PoF: {}'.format(pof))
        

