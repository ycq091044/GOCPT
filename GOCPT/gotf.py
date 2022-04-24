from re import L
import numpy as np
from .utils import GOCPTE_fac_update, GOCPT_fac_update
from .metrics import PoF
from .otf import BASE_ONLINE_TENSOR_FAC


class GOCPTE(BASE_ONLINE_TENSOR_FAC):
    """
    Our effective version for factorization
    """
    def __init__(self, base_X, R, iters=50):
        super(GOCPTE, self).__init__(base_X, R, iters)
        self.cal_aux()

    def update(self, X, iters=3, alpha=1, verbose=True, **kwargs):
        # update class params
        # for calculating pof, we store X
        self.collect_X(X)

        if ('new_R' in kwargs):
            new_R = kwargs['new_R']
            if new_R > 1:
                self.R = new_R
            else:
                print ('{}-th update Unsuccess! R should be an >=2 integer, you input is invalid'.format(self.counter)); return
        else:
            pass
        
        self.factors, run_time = GOCPTE_fac_update([X, self.R], alpha, self.factors, iters)
        pof_score = PoF(self.X, self.factors)
        if verbose:
            print ("{}-th update Success! PoF: {:.4}, run_time: {:.4}s".\
                            format(self.counter, pof_score, run_time))
        self.pof_update_list.append(pof_score)
        self.counter += 1

    
class GOCPT(BASE_ONLINE_TENSOR_FAC):
    """
    Our effective version for factorization
    """
    def __init__(self, base_X, R, iters=50):
        super(GOCPT, self).__init__(base_X, R, iters)
        self.cal_aux()

    def update(self, X, iters=3, verbose=True, **kwargs):
        # update class params
        # for calculating pof, we store X
        self.collect_X(X)

        if ('new_R' in kwargs):
            new_R = kwargs['new_R']
            if new_R > 1:
                self.R = new_R
            else:
                print ('{}-th update Unsuccess! R should be an >=2 integer, you input is invalid'.format(self.counter)); return
        else:
            pass
        
        self.factors, run_time = GOCPT_fac_update([self.X, self.R], self.factors, iters)
        pof_score = PoF(self.X, self.factors)
        if verbose:
            print ("{}-th update Success! PoF: {:.4}, run_time: {:.4}s".\
                            format(self.counter, pof_score, run_time))
        self.pof_update_list.append(pof_score)
        self.counter += 1
        