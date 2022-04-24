import numpy as np
from regex import P
from .utils import generate_random_factors, cpc_als_iteration, OnlineSGD_update, OLSTEC_update, \
    get_lhs_rhs_mask_weighted, GOCPTE_comp_update
from .metrics import PoF
from numpy import linalg as la

def cpc(Omega, mask_X, R, iters=None, verbose=False):
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
     
    factors = generate_random_factors(Omega, R)
    pof_score_list = []

    if iters is not None:
        for i in range(iters):
            factors, run_time = cpc_als_iteration(mask_X, factors, Omega)
            pof_score = PoF(mask_X, factors, Omega)
            if verbose and (i % 10 == 0):
                print ("{}-th iters, PoF: {}, time: {}s".format(i, pof_score, run_time))
            pof_score_list.append(pof_score)
    else:
        pof_score = PoF(mask_X, factors, Omega)
        max_iters = 50
        for i in range(max_iters):
            factors, run_time = cpc_als_iteration(mask_X, factors, Omega)
            new_pof_score = PoF(mask_X, factors, Omega)
            if verbose and (i % 10 == 0):
                print ("{}-th iters, PoF: {}, time: {}s".format(i, PoF(mask_X, factors, Omega), run_time))
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


class BASE_ONLINE_TENSOR_COMP:
    def __init__(self, base_mask, base_X, R, iters=50):
        self.factors = None
        self.N = None
        self.R = R
        self.mask = base_mask
        self.X = base_X
        self.pof_update_list = []
        self.counter = 0

        self.initialize(iters)
    
    def initialize(self, iters):
        # update X and counter
        self.N = self.X.ndim
        self.counter += 1

        # update factors
        factors, _ = cpc(self.mask, self.X, self.R, iters, verbose=False)
        self.factors = factors

        pof = PoF(self.X, self.factors, self.mask)
        self.pof_update_list.append(pof)
        print ('Initial PoF Metric: {}'.format(pof))

    def cal_aux(self):
        print ('This model does not need to prepare aux!')
        print ()

    def collect_X_and_mask(self, X, mask):
        self.X = np.concatenate([self.X, X], -1)
        self.mask = np.concatenate([self.mask, mask], -1)


class OnlineSGD(BASE_ONLINE_TENSOR_COMP):
    """
    Online Tensor Completion based on Stochastic Gradient Descent
    """
    def __init__(self, base_mask, base_X, R, iters=50):
        super(OnlineSGD, self).__init__(base_mask, base_X, R, iters)
        self.cal_aux()
    
    def update(self, mask_X, mask, lr=1e-10, index=1, verbose=False):
        # for calculating pof, we store X and the mask
        self.collect_X_and_mask(mask_X, mask)

        self.factors, run_time = OnlineSGD_update(mask_X, mask, self.factors, lr, index)
        pof_score = PoF(self.X, self.factors, self.mask)
        if verbose:
            print ("{}-th update, PoF: {}, run_time: {}s".\
                            format(self.counter, pof_score, run_time))
        self.pof_update_list.append(pof_score)
        self.counter += 1


class OLSTEC(BASE_ONLINE_TENSOR_COMP):
    """
    Kasai, Online Low-Rank Tensor Subspace Tracking from Incomplete Data by CP Decomposition \
    using Recursive Least Squares, ICASSP 2016
    """
    def __init__(self, base_mask, base_X, R, iters=50):
        super(OLSTEC, self).__init__(base_mask, base_X, R, iters)
        self.R = []
        self.S = []
        self.cal_aux()
    
    def update(self, mask_X, mask, lr=1e-10, index=1, verbose=False):
        # for calculating pof, we store X and the mask
        self.collect_X_and_mask(mask_X, mask)

        self.factors, self.R, self.S, run_time = OLSTEC_update(mask_X, mask, self.factors, self.R, \
            self.S, mu=1e-9, Lambda=0.88)
        pof_score = PoF(self.X, self.factors, self.mask)
        if verbose:
            print ("{}-th update, PoF: {}, run_time: {}s".\
                            format(self.counter, pof_score, run_time))
        self.pof_update_list.append(pof_score)
        self.counter += 1

    def cal_aux(self):
        Lambda = 0.98
        coeff = np.array([Lambda ** i for i in range(self.factors[-1].shape[0])])[::-1]
        for i in range(self.X.ndim - 1):
            Ri, Si = get_lhs_rhs_mask_weighted(self.mask, self.X, self.factors, coeff, i)
            self.R.append(Ri); self.S.append(Si)
        
        print ('aux variables prepared!')
        print ()


class GOCPTE(BASE_ONLINE_TENSOR_COMP):
    """
    Our efficient version for online tensor completion 
    """
    def __init__(self, base_mask, base_X, R, iters=50):
        super(GOCPTE, self).__init__(base_mask, base_X, R, iters)
        self.cal_aux()
    
    def update(self, mask_X, mask, alpha=1, verbose=False):
        # for calculating pof, we store X and the mask
        self.collect_X_and_mask(mask_X, mask)

        self.factors, run_time = GOCPTE_comp_update(mask_X, mask, self.factors, alpha)
        pof_score = PoF(self.X, self.factors, self.mask)
        if verbose:
            print ("{}-th update, PoF: {}, run_time: {}s".\
                            format(self.counter, pof_score, run_time))
        self.pof_update_list.append(pof_score)
        self.counter += 1