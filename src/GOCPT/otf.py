import numpy as np
from regex import P
from .utils import cpd_als_iteration, OnlineCPD_update, get_lhs_rhs_from_tensor, \
        MAST_update, BiSVD, SDT_update, RLST_update, CPStream_update, GOCPTE_fac_update, \
        generate_random_factors
from .metrics import PoF
from scipy import linalg as la

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
     
    factors = generate_random_factors(X, R)
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


class BASE_ONLINE_TENSOR_FAC:
    def __init__(self, base_X, R, iters=50):
        self.factors = None
        self.N = None
        self.R = R
        self.X = base_X
        self.pof_update_list = []
        self.counter = 0

        self.initialize(iters)
    
    def initialize(self, iters):
        # update X and counter
        self.N = self.X.ndim
        self.counter += 1

        # update factors
        factors, _ = cpd(self.X, self.R, iters, verbose=False)
        self.factors = factors

        pof = PoF(self.X, self.factors)
        self.pof_update_list.append(pof)
        print ('Initial PoF Metric: {}'.format(pof))

    def cal_aux(self):
        print ('This model does not need to prepare aux!')
        print ()

    def collect_X(self, X):
        self.X = np.concatenate([self.X, X], -1)
        

class MAST(BASE_ONLINE_TENSOR_FAC):
    """
    refer to Song et al. Multi-Aspect Streaming Tensor Completion. KDD 2017
    """
    def __init__(self, base_X, R, iters=50):
        super(MAST, self).__init__(base_X, R, iters)
        self.cal_aux()
    
    def update(self, X, verbose=False):
        # for calculating pof, we store X
        self.collect_X(X)

        self.factors, run_time = MAST_update(X, self.factors, alphaN=[1/15]*self.N, \
            alpha=1, iters=20, phi=1.05, eta_max=1, eta_init=1e-5, tol=1e-5)

        pof_score = PoF(self.X, self.factors)
        if verbose:
            print ("{}-th update, PoF: {}, run_time: {}s".\
                            format(self.counter, pof_score, run_time))
        self.pof_update_list.append(pof_score)
        self.counter += 1


class OnlineCPD(BASE_ONLINE_TENSOR_FAC):
    """
    refer to Zhou et al. Accelerating Online CP Decompositions for \
        Higher Order Tensors. KDD 2016
    """
    def __init__(self, base_X, R, iters=50):
        super(OnlineCPD, self).__init__(base_X, R, iters)
        self.P = None
        self.Q = None
        self.cal_aux()
    
    def update(self, X, verbose=False):
        # for calculating pof, we store X
        self.collect_X(X)

        self.factors, self.P, self.Q, run_time = \
                        OnlineCPD_update(X, self.factors, self.P, self.Q)
        pof_score = PoF(self.X, self.factors)
        if verbose:
            print ("{}-th update, PoF: {}, run_time: {}s".\
                            format(self.counter, pof_score, run_time))
        self.pof_update_list.append(pof_score)
        self.counter += 1

    def cal_aux(self):
        # update factors, P and Q
        self.P = [None for _ in range(self.N)]
        self.Q = [None for _ in range(self.N)]

        for i in range(self.N-1):
            Qn, Pn = get_lhs_rhs_from_tensor(self.X, self.factors, i)
            self.P[i] = Pn; self.Q[i] = Qn
        
        print ('aux variables prepared!')
        print ()


class SDT(BASE_ONLINE_TENSOR_FAC):
    """
    refer to Nion et al. Adaptive Algorithms to Track the PARAFAC Decomposition of a Third-Order Tensor. TSP 2009
    CAN ONLY BE USED FOR THIRD-ORDER TENSOR
    """
    def __init__(self, base_X, R, iters=50):
        super(SDT, self).__init__(base_X, R, iters)
        self.aux = []
        self.cal_aux()

    def update(self, X, verbose=False):
        # for calculating pof, we store X
        self.collect_X(X)

        self.factors, self.aux, run_time = SDT_update(X,self.factors, self.aux, gamma=0.99)

        pof_score = PoF(self.X, self.factors)
        if verbose:
            print ("{}-th update, PoF: {}, run_time: {}s".\
                            format(self.counter, pof_score, run_time))
        self.pof_update_list.append(pof_score)
        self.counter += 1

    def cal_aux(self):
        [A, B, C] = self.factors #[A, B, C]

        # auxilaries
        gamma = 0.99
        coeff = [gamma ** i for i in range(self.X.shape[-1])][::-1] # exponential weight
        X_pre = np.einsum('ijk,k->ijk',self.X,coeff,optimize=True).reshape(-1, self.X.shape[-1])
        U, S, V = BiSVD(X_pre, self.R)
        E = U @ S
        C = np.einsum('kr,k->kr',C,coeff,optimize=True)
        W = la.inv(E.T @ E) @ E.T @ la.khatri_rao(B,A)
        Wi = la.inv(W)

        self.aux = [W, Wi, V, S, U]
        print ('aux variables prepared!')
        print ()
        

class RLST(BASE_ONLINE_TENSOR_FAC):
    """
    refer to Nion et al. Adaptive Algorithms to Track the PARAFAC Decomposition of a Third-Order Tensor. TSP 2009
    CAN ONLY BE USED FOR THIRD-ORDER TENSOR
    """
    def __init__(self, base_X, R, iters=50):
        super(RLST, self).__init__(base_X, R, iters)
        self.aux = []
        self.cal_aux()

    def update(self, X, verbose=False):
        # for calculating pof, we store X
        self.collect_X(X)

        self.factors, self.aux, run_time = RLST_update(X, self.factors, self.aux, gamma=0.995)
        pof_score = PoF(self.X, self.factors)
        if verbose:
            print ("{}-th update, PoF: {}, run_time: {}s".\
                            format(self.counter, pof_score, run_time))
        self.pof_update_list.append(pof_score)
        self.counter += 1

    def cal_aux(self):
        [A, B, C] = self.factors #[A, B, C]

        # auxilaries
        X_pre = self.X.reshape(-1, C.shape[0])
        R1 = X_pre @ C
        P1 = C.T @ C
        Z1 = la.inv(R1.T @ R1) @ R1.T
        Q1 = la.inv(P1)

        self.aux = [P1, Q1, R1, Z1]
        print ('aux variables prepared!')
        print ()


class CPStream(BASE_ONLINE_TENSOR_FAC):
    """
    refer to Smith et al. Streaming Tensor Factorization for Infinite Data Sources. SDM 2018 
    """
    def __init__(self, base_X, R, iters=50):
        super(CPStream, self).__init__(base_X, R, iters)
        self.aux = None
        self.cal_aux()

    def update(self, X, verbose=False):
        # for calculating pof, we store X
        self.collect_X(X)

        self.factors, self.aux, run_time = CPStream_update(X, self.factors, self.aux, \
            mu=2, iters=20, tol=1e-5)
        pof_score = PoF(self.X, self.factors)
        if verbose:
            print ("{}-th update, PoF: {}, run_time: {}s".\
                            format(self.counter, pof_score, run_time))
        self.pof_update_list.append(pof_score)
        self.counter += 1

    def cal_aux(self):
        [A, B, C] = self.factors #[A, B, C]
        # auxilaries
        mu = 0.99
        coeff = np.array([mu ** i for i in range(C.shape[0])])[::-1]
        G = np.einsum('kr,km,k->rm',C,C,coeff,optimize=True)
        self.aux = G
        print ('aux variables prepared!')
        print ()


class GOCPTE(BASE_ONLINE_TENSOR_FAC):
    """
    Our effective version for factorization
    """
    def __init__(self, base_X, R, iters=50):
        super(GOCPTE, self).__init__(base_X, R, iters)
        self.cal_aux()

    def update(self, X, verbose=False):
        # for calculating pof, we store X
        self.collect_X(X)
        
        self.factors, run_time = GOCPTE_fac_update(X, self.factors, alpha=5)

        pof_score = PoF(self.X, self.factors)
        if verbose:
            print ("{}-th update, PoF: {}, run_time: {}s".\
                            format(self.counter, pof_score, run_time))
        self.pof_update_list.append(pof_score)
        self.counter += 1
