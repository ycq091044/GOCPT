import numpy as np
from scipy import linalg as la
import time

def get_new_factor(X, factors, i):
    ein_str = ""
    ein_mat = []
    RHS_str = "".join([chr(j+97) for j in range(len(factors))])
    RHS_mat = [X]
    for j in range(len(factors)):
        if j != i:
            ein_str += '{}r,{}z,'.format(chr(j+97), chr(j+97))
            ein_mat.append(factors[j]); ein_mat.append(factors[j])
            RHS_str += ',{}r'.format(chr(j+97))
            RHS_mat.append(factors[j])
    ein_str = ein_str[:-1] + '->rz'
    RHS_str += '->r{}'.format(chr(i+97))
    result = optimize(np.einsum(ein_str,*ein_mat,optimize=True), \
            np.einsum(RHS_str,*RHS_mat,optimize=True)).T
    return result

def rec(factors):
    """
    Using the factors to reconstruct the tensor
    INPUT:
        - <matrix list> factors: the low rank factors (A1, A2, ..., An)
    OUTPUT:
        - <tensor> X: the reconstructed tensor 
    """
    ein_str = ""
    for i in range(len(factors)):
        if i == len(factors) - 1:
            ein_str += "{}r->{}".format(chr(i+97), \
                        "".join([chr(j+97) for j in range(i + 1)]))
        else:
            ein_str += "{}r,".format(chr(i+97))
    X = np.einsum(ein_str, *factors, optimize=True)
    return X

def optimize(A, B, reg=1e-6):
    """
    The least squares solver: AX = B
    INPUT:
        - <matrix> A: the coefficient matrix (R,R)
        - <matrix> B: the RHS matrix (R, In)
    OUTPUT:
        - <matrix> X: the solution (R, In)
    """
    try:
        L = la.cholesky(A + np.eye(A.shape[1]) * reg)
        y = la.solve_triangular(L.T, B, lower=True)
        x = la.solve_triangular(L, y, lower=False)
    except:
        x = la.solve(A + np.eye(A.shape[1]) * reg, B)
    return x

def cpd_als_iteration(X, factors):
    """
    The CPD-ALS algorithm
    INPUT:
        - <tensor> X: this is the input tensor of size (I1, I2, ..., In)
        - <matrix list> factors: the initalized factors (A1, A2, ..., An)
    OUTPUT:
        - <matrix list> factors: the optimized factors (A1, A2, ..., An)
    
    --- EXAMPLE ---
        INPUT:
            - X: (I1, I2, I3)
            - factors: [A1, A2, A3] 
        INTERMEDIATE (first iteration):
            - <string> ein_str: "ar,az,br,bz->rz"
            - <matrix list> ein_mat: [A1, A1, A2, A2]
            - <string> RHS_str: "abc,ar,br->rc"
            - <matrix list> RHS_mat: [X, A1, A2]
        OUTPUT
            - factors: [A1', A2', A3']
    """
    tic = time.time()
    for i in range(len(factors)):
        factors[i] = get_new_factor(X, factors, i)
    toc = time.time()
    return factors, toc - tic


def get_new_LHS_RHS(X, factors, i):
    ein_str = ""
    ein_mat = []
    RHS_str = "".join([chr(j+97) for j in range(len(factors))])
    RHS_mat = [X]
    for j in range(len(factors)):
        if j != i:
            ein_str += '{}r,{}z,'.format(chr(j+97), chr(j+97))
            ein_mat.append(factors[j]); ein_mat.append(factors[j])
            RHS_str += ',{}r'.format(chr(j+97))
            RHS_mat.append(factors[j])
    ein_str = ein_str[:-1] + '->rz'
    RHS_str += '->r{}'.format(chr(i+97))
    result1 = np.einsum(ein_str,*ein_mat,optimize=True)
    result2 = np.einsum(RHS_str,*RHS_mat,optimize=True)
    return result1, result2


def OnlineCPD_iteration(X, factors, P, Q):
    """
    refer to Zhou et al. Accelerating Online CP Decompositions for Higher Order Tensors. KDD 2016
    Note: X_new: I x J x 1
    """
    tic = time.time()
    # get the augmented part of the last factor
    aug_last_factor = get_new_factor(X, factors, len(factors) - 1)
    new_last_factor = np.concatenate([factors[-1], aug_last_factor], axis=0)
    factors[-1] = aug_last_factor 

    # update P and Q
    for i in range(len(factors)-1):
        Qn, Pn = get_new_LHS_RHS(X, factors, i)
        P[i] += Pn; Q[i] += Qn
        factors[i] = optimize(Q[i],P[i]).T 

    factors[-1] = new_last_factor
    toc = time.time()
    return factors, P, Q, toc - tic