import numpy as np
from scipy import linalg as la
import time

def optimize(A, B, reg=1e-5):
    """
    Least Squares Solver: Au = B
    """
    L = la.cholesky(A + np.eye(A.shape[1]) * reg)
    y = la.solve_triangular(L.T, B, lower=True)
    u = la.solve_triangular(L, y, lower=False)
    return u

def Optimizer(Omega, A, RHS, num):
    """
    masked least square optimizer:
        A @ u.T = Omega * RHS
        number: which factor
        reg: 2-norm regulizer
    """
    N = len(A)
    lst_mat = []
    T_inds = "".join([chr(ord('a')+i) for i in range(Omega.ndim)])
    einstr=""
    for i in range(N):
        if i != num:
            einstr+=chr(ord('a')+i) + 'r' + ','
            lst_mat.append(A[i])
            einstr+=chr(ord('a')+i) + 'z' + ','
            lst_mat.append(A[i])
    einstr+= T_inds + "->"+chr(ord('a')+num)+'rz'
    lst_mat.append(Omega)
    P = np.einsum(einstr,*lst_mat,optimize=True)
    o = np.zeros(RHS.shape)
    for j in range(A[num].shape[0]):
        o[j,:] = optimize(P[j], RHS[j,:])
    return o

def metric(X, factors, mask=None):
    A, B, C = factors
    rec = np.einsum('ir,jr,kr->ijk',A,B,C)
    if mask is not None:
        loss = la.norm(mask * rec - X) ** 2
        PoF = 1 - la.norm(mask * rec - X) / la.norm(X)
    else:
        loss = la.norm(rec - X) ** 2
        PoF = 1 - la.norm(rec - X) / la.norm(X) 
    return rec, loss, PoF