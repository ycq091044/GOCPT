import numpy as np
from .utils import generate_random_factors, cpc_als_iteration, GOCPTE_general_comp_update, \
    GOCPT_general_comp_update
from .metrics import PoF
from .otc import BASE_ONLINE_TENSOR_COMP

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


class GOCPTE(BASE_ONLINE_TENSOR_COMP):
    """
    Our efficient version for generalized online tensor completion (GTC)
    """
    def __init__(self, base, R, iters=50):
        super(GOCPTE, self).__init__(base, R, iters)
        self.cal_aux()
    
    def update(self, increment, alpha=1, iters=3, verbose=True, **kwargs):
        mask_X, mask = increment

        # process rank change R
        if ('new_R' in kwargs):
            new_R = kwargs['new_R']
            if new_R > 1:
                self.R = new_R
            else:
                print ('{}-th update Unsuccess! R should be an >=2 integer, you input is invalid\n'.format(self.counter)); return
        else:
            pass
        kwargs['new_R'] = self.R
    
        # for calculating pof, we store X and the mask
        self.collect_X_and_mask(mask_X, mask, kwargs)

        self.factors, run_time = GOCPTE_general_comp_update([mask_X, mask, kwargs], self.factors, alpha, iters)
        pof_score = PoF(self.X, self.factors, self.mask)
        if verbose:
            print ("{}-th update Success! PoF: {:.4}, run_time: {:.4}s\n".\
                            format(self.counter, pof_score, run_time))
        self.pof_update_list.append(pof_score)
        self.counter += 1

    def collect_X_and_mask(self, X, mask, kwargs):
        self.X = np.concatenate([self.X, X], -1)
        self.mask = np.concatenate([self.mask, mask], -1)
        if 'value_update' in kwargs:
            vupdate_coords, vupdate_values = kwargs['value_update']
            self.X[vupdate_coords] = vupdate_values
            self.mask[vupdate_coords] = 1
        if 'miss_fill' in kwargs:
            fill_coords, fill_values = kwargs['miss_fill']
            self.X[fill_coords] = fill_values
            self.mask[fill_coords] = 1


class GOCPT(BASE_ONLINE_TENSOR_COMP):
    """
    Our full version for generalized online tensor completion (GTC)
    """
    def __init__(self, base, R, iters=50):
        super(GOCPT, self).__init__(base, R, iters)
        self.cal_aux()
    
    def update(self, increment, alpha=1, iters=3, verbose=True, **kwargs):
        mask_X, mask = increment

        # process rank change R
        if ('new_R' in kwargs):
            new_R = kwargs['new_R']
            if new_R > 1:
                self.R = new_R
            else:
                print ('{}-th update Unsuccess! R should be an >=2 integer, you input is invalid\n'.format(self.counter)); return
        else:
            pass
    
        # for calculating pof, we store X and the mask
        self.collect_X_and_mask(mask_X, mask, kwargs)

        self.factors, run_time = GOCPT_general_comp_update([self.X, self.mask, self.R], self.factors, alpha, iters)
        pof_score = PoF(self.X, self.factors, self.mask)
        if verbose:
            print ("{}-th update Success! PoF: {:.4}, run_time: {:.4}s\n".\
                            format(self.counter, pof_score, run_time))
        self.pof_update_list.append(pof_score)
        self.counter += 1

    def collect_X_and_mask(self, X, mask, kwargs):
        self.X = np.concatenate([self.X, X], -1)
        self.mask = np.concatenate([self.mask, mask], -1)
        if 'value_update' in kwargs:
            vupdate_coords, vupdate_values = kwargs['value_update']
            self.X[vupdate_coords] = vupdate_values
            self.mask[vupdate_coords] = 1
        if 'miss_fill' in kwargs:
            fill_coords, fill_values = kwargs['miss_fill']
            self.X[fill_coords] = fill_values
            self.mask[fill_coords] = 1