import numpy as np
from scipy import linalg as la
import time


def generate_random_factors(X, R):
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


def rec_from_factors(factors):
    """
    Using the factors to reconstruct the tensor
    INPUT:
        - <matrix list> factors: the low rank factors (A1, A2, ..., An)
    OUTPUT:
        - <tensor> X: the reconstructed tensor 
    """
    lhs_str = ""
    for i in range(len(factors)):
        if i == len(factors) - 1:
            lhs_str += "{}r->{}".format(chr(i+97), \
                        "".join([chr(j+97) for j in range(i + 1)]))
        else:
            lhs_str += "{}r,".format(chr(i+97))
    X = np.einsum(lhs_str, *factors, optimize=True)
    return X


def optimize(A, B, reg=1e-6):
    """
    The least squares solver: AX = B
    INPUT:
        - <matrix> A: the coefficient matrix (R,R)
        - <matrix> B: the rhs matrix (R, In)
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


# dense strategy iteration
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
            - <string> lhs_str: "ar,az,br,bz->rz"
            - <matrix list> lhs_mat: [A1, A1, A2, A2]
            - <string> rhs_str: "abc,ar,br->rc"
            - <matrix list> rhs_mat: [X, A1, A2]
        OUTPUT
            - factors: [A1', A2', A3']
    """
    tic = time.time()
    for i in range(len(factors)):
        lhs, rhs = get_lhs_rhs_from_tensor(X, factors, i)
        factors[i] = optimize(lhs, rhs).T
    toc = time.time()
    return factors, toc - tic


def get_lhs_rhs_from_tensor(X, factors, i):
    """
    get the lhs and rhs einsum results of the i-th factor
    INPUT:
        - <tensor> X: the tensor
        - <matrix list> factors: the list of current factors
        - <int> i: the indices of the target factor
    OUTPUT:
        - <matrix> lhs: size (R,R), is the self-product of khatri-Rao product
        - <matrix> rhs: size (R, Ii), is the MTTKRP result 
    """
    lhs_str = ""
    lhs_mat = []
    rhs_str = "".join([chr(j+97) for j in range(len(factors))])
    rhs_mat = [X]
    for j in range(len(factors)):
        if j != i:
            lhs_str += '{}r,{}z,'.format(chr(j+97), chr(j+97))
            lhs_mat.append(factors[j]); lhs_mat.append(factors[j])
            rhs_str += ',{}r'.format(chr(j+97))
            rhs_mat.append(factors[j])
    lhs_str = lhs_str[:-1] + '->rz'
    rhs_str += '->r{}'.format(chr(i+97))
    lhs = np.einsum(lhs_str,*lhs_mat,optimize=True)
    rhs = np.einsum(rhs_str,*rhs_mat,optimize=True)
    return lhs, rhs


def get_lhs_rhs_from_copy(As, factors, i):
    """
    get the lhs and rhs einsum results of the i-th factor from the nested copy factors
    INPUT:
        - <matrix list> As: the list of copied factors
        - <matrix list> factors: the list of current factors
        - <int> i: the indices of the target factor
    OUTPUT:
        - <matrix> lhs: size (R,R), is the self-product of khatri-Rao product
        - <matrix> rhs: size (R, Ii), is the MTTKRP result 
    """
    lhs_str = ""
    lhs_mat = []
    rhs_str = "{}r".format(chr(i+97))
    rhs_mat = [As[i]]
    for j in range(len(factors)):
        if j != i:
            lhs_str += '{}r,{}z,'.format(chr(j+97), chr(j+97))
            lhs_mat.append(factors[j]); lhs_mat.append(factors[j])
            rhs_str += ',{}r,{}z'.format(chr(j+97), chr(j+97))
            rhs_mat.append(As[j]); rhs_mat.append(factors[j])
    lhs_str = lhs_str[:-1] + '->rz'
    rhs_str += '->z{}'.format(chr(i+97))
    
    lhs = np.einsum(lhs_str,*lhs_mat,optimize=True)
    rhs = np.einsum(rhs_str,*rhs_mat,optimize=True)
    return lhs, rhs


def OnlineCPD_update(X, factors, P, Q):
    """
    refer to Zhou et al. Accelerating Online CP Decompositions for \
        Higher Order Tensors. KDD 2016
    INPUT:
        - <tensor> X: the given tensor slice
        - <matrix list> factors: current factor list
        - <matrix list> P: a list of rhs for Ai
        - <matrix list> Q: a list of lhs for Ai
    OUTPUT:
        - <matrix list> factors: updated current factor list
        - <matrix list> P: updated list of rhs for Ai
        - <matrix list> Q: updated list of lhs for Ai
        - <float> toc - tic: the time span
    """
    tic = time.time()
    # get the augmented part of the last factor
    lhs, rhs = get_lhs_rhs_from_tensor(X, factors, len(factors) - 1)
    aug_last_factor = la.solve(lhs, rhs).T

    new_last_factor = np.concatenate([factors[-1], aug_last_factor], axis=0)
    factors[-1] = aug_last_factor 

    # update P and Q
    for i in range(len(factors)-1):
        Qn, Pn = get_lhs_rhs_from_tensor(X, factors, i)
        P[i] += Pn; Q[i] += Qn
        factors[i] = optimize(Q[i],P[i]).T 

    factors[-1] = new_last_factor
    return factors, P, Q, time.time() - tic


def MAST_update(X, factors, alphaN, alpha=1, iters=20, phi=1.05, \
                eta_max=0.001, eta_init=1e-5, tol=1e-5):
    """
    refer to Song et al. Multi-Aspect Streaming Tensor Completion. KDD 2017
    INPUT:
        - <tensor> X: the given tensor slice
        - <matrix list> factors: current factor list
        - <list> alphaN: weight list for auxiliaries, default: [1/15] * len(factors)
        - <float> alpha: weight for balancing new and history data, default: 1
        - <float> phi: default from the original paper: 1.05
        - <float> eta_init: default from the original paper: 1e-5
        - <float> eta: default from the original paper: 1e-4
        - <float> eta_max: default from the original paper: 1
        - <float> tol: default1e-5
    OUTPUT:
        - <matrix list> factors: updated current factor list
        - <float> toc - tic: the time span
    """
    tic = time.time()

    R = factors[0].shape[1]
    # initialize Z, Y, copy A factor lists
    Zs, Ys, As = [], [], []
    for idx, factor in enumerate(factors):
        if idx != len(factors) - 1:
            Zs.append(np.zeros_like(factor))
            Ys.append(np.zeros_like(factor))
            As.append(factor.copy())
        else:
            Zs.append(np.zeros((factor.shape[0]+1, R)))
            Ys.append(np.zeros((factor.shape[0]+1, R)))
            As.append(factor.copy())
    aug_last_factor = np.random.random((1,R))    
    
    # preparation
    eta = eta_init
    eye = np.eye(R)
    rec_old = rec_from_factors(factors[:-1] + [np.concatenate([factors[-1], aug_last_factor], 0)])

    for _ in range(iters):
        eta = min(eta * phi, eta_max)
        # update the last factors
        lhs, rhs = get_lhs_rhs_from_copy(As, factors, len(factors)-1)
        factors[-1] = optimize(lhs + eta * eye, rhs + eta * Zs[-1][:-1,:].T + \
            Ys[-1][:-1,:].T, reg=0).T 
        lhs, rhs = get_lhs_rhs_from_tensor(X, factors, len(factors)-1)
        aug_last_factor = optimize(lhs + eta * eye, rhs + eta * Zs[-1][-1:,:].T + \
            Ys[-1][-1:,:].T, reg=0).T
        
        # update other factors
        for j in range(len(factors) - 1):
            # update other factors
            lhs1, rhs1 = get_lhs_rhs_from_tensor(X, factors[:-1] + [aug_last_factor], j)
            lhs2, rhs2 = get_lhs_rhs_from_copy(As, factors, j)
            factors[j] = optimize(lhs1 + alpha * lhs2 + eta * eye, rhs1 + alpha * rhs2 + \
                eta * Zs[j].T + Ys[j].T, reg=0).T
        
        # update other auxiliaries for other factors
        for j in range(len(factors) - 1):
            u, s, vh = la.svd(factors[j] - Ys[j] / eta, full_matrices=False)
            Zs[j] = u @ np.diag(s * (s >= alphaN[j] / eta)) @ vh
            Ys[j] += eta * (Zs[j] - factors[j])
        
        # update the auxiliaries
        u, s, vh = la.svd(np.concatenate([factors[-1], aug_last_factor], 0) - Ys[-1] / eta, \
            full_matrices=False)
        Zs[-1] = u @ np.diag(s * (s >= alphaN[-1] / eta)) @ vh
        Ys[-1] += eta * (Zs[-1] - np.concatenate([factors[-1], aug_last_factor], 0))

        rec_new = rec_from_factors(factors[:-1] + [np.concatenate([factors[-1], aug_last_factor], 0)])
        if la.norm(rec_old - rec_new) / la.norm(rec_old) < tol:
            break
        else:
            rec_old = rec_new

    return factors[:-1] + [np.concatenate([factors[-1], aug_last_factor], 0)], time.time() - tic


# used for SDT
def BiSVD(M, r, epsilon=1e-5, maxIter=50):
    """
    Bi-Singular value decomposition
    input M: I x N
    output U, S, Vt of the top-r components
    """
    Qa = np.concatenate([np.eye(r), np.zeros((M.shape[1]-r,r))],axis=0)
    Qb = np.random.random((M.shape[0], r))
    Rb = np.random.random((r, r)) 
    
    t = 0
    while (la.norm(Qb @ Rb @ Qa.T - M) / la.norm(M) > epsilon) and (t < maxIter):
        B = M @ Qa
        Qb, Rb = la.qr(B, mode='economic')
        A = M.T @ Qb
        Qa, Ra = la.qr(A, mode='economic')
        t += 1
    return Qb, Rb, Qa


# used for SDT
def SWASVD(T, U, S, V, gamma):
    """
    U: IJ x r
    T: IJ x 1
    """
    # update V
    h = U.T @ T
    B = np.concatenate([gamma ** (0.5) * V @ S.T, h.T], axis=0)
    V_new, Rb = la.qr(B[1:, :], mode='economic')
    # update U
    xo = T - U @ h
    E_new = U @ Rb.T + xo @ V_new[-1:,:]
    U_new, S_new = la.qr(E_new, mode='economic')
    return U_new, S_new, V_new, E_new


def SDT_update(X, factors, aux, gamma):
    """
    refer to Nion et al. Adaptive Algorithms to Track the PARAFAC Decomposition of a Third-Order Tensor. TSP 2009
    also: https://github.com/thanhtbt/ROLCP/tree/main/Algorithms/PAFARAC(2009)
    """

    tic = time.time()
    A0, B0, C0 = factors
    W0, Wi0, V0, S0, U0 = aux
    I, J, _ = X.shape
    _, R = A0.shape

    # step 1:
    U1, S1, V1, E1 = SWASVD(X.reshape(-1,1), U0, S0, V0, gamma)
    # step 2:
    Z = V1[:-1,:].T @ V0[1:, :]
    v0 = V0[:1,:]; v1 = V1[-1:,:]
    W1 = gamma**(-0.5) * Z @ (np.eye(R) + v0.T @ v0 / (1 - v0 @ v0.T)) @ W0
    Wi1 = gamma**(0.5) * Wi0 @ Z.T @ (np.eye(R) + v1.T @ v1 / (1 - v1 @ v1.T))
    c = Wi1 @ v1.T
    # step 3:
    H1 = E1 @ W1
    A1 = np.zeros((I, R))
    B1 = np.zeros((J, R))
    for r in range(R):
        Hr1 = H1[:, r].reshape(I, J)
        B1[:, r] = Hr1.T @ A0[:, r]
        a1 = Hr1 @ B1[:, r]
        A1[:, r] = a1 / la.norm(a1)
    # step 4:
    C1 = np.concatenate([C0, c.T], axis=0)
    return [A1, B1, C1], [W1, Wi1, V1, S1, U1], time.time() - tic


# used for RLST
def Pinv_update(A, P, c, d, idcase):
    """
    refer to Nion et al. Adaptive Algorithms to Track the PARAFAC Decomposition of a Third-Order Tensor. TSP 2009
    also: https://github.com/thanhtbt/ROLCP/tree/main/Algorithms/PAFARAC(2009)
    """

    if idcase == '5':
        k = P @ c       # R x 1
        h = d.T @ P     # 1 x T
        bet = 1 + d.T @ k # scalar
        u = c - A @ k
        nu = u.T @ u
        nh = h @ h.T

        s2 = nh * nu + bet * bet
        z2 = P @ h.T
        p2 = -(nu / bet) * z2 - k
        q2h = -(nh / bet) * u.T - h
        A = A + c @ d.T
        P = P + ((1 / bet) * z2) @ u.T - ((bet / s2) * p2) @ q2h

    elif idcase == '7':
        k = P @ c       # R x 1
        bet = 1 + d.T @ k # scalar
        h = d.T @ P     # 1 x T
        A = A + c @ d.T
        P = P - ((1/bet) * k) @ h

    return A, P


def RLST_update(X,factors,aux,gamma):
    """
    refer to Nion et al. Adaptive Algorithms to Track the PARAFAC Decomposition of a Third-Order Tensor. TSP 2009
    also: https://github.com/thanhtbt/ROLCP/tree/main/Algorithms/PAFARAC(2009)
    """
    tic = time.time()
    A0, B0, C0 = factors
    P0,Q0,R0,Z0 = aux
    I, J, _ = X.shape
    _, R = A0.shape
    x = X.reshape(-1,1)

    # step 1:
    c1 = P0 @ (Z0 @ x)

    # step 2:
    R1, Z1 = Pinv_update(gamma * R0, gamma**(-1)*Z0, x, c1, '5')
    P1, Q1 = Pinv_update(gamma * P0, gamma**(-1)*Q0, c1, c1, '7')

    # step 3:
    c1 = P1 @ (Z1 @ x)
    R1, Z1 = Pinv_update(R1, Z1, x, c1, '5')
    P1, Q1 = Pinv_update(P1, Q1, c1, c1, '7')
    c1 = P1 @ (Z1 @ x)

    # step 4:
    H1 = R1 @ Q1
    A1 = np.zeros((I, R))
    B1 = np.zeros((J, R))
    for r in range(R):
        Hr1 = H1[:, r].reshape(I, J)
        B1[:, r] = Hr1.T @ A0[:, r]
        a1 = Hr1 @ B1[:, r]
        A1[:, r] = a1 / la.norm(a1)
    C1 = np.concatenate([C0, c1.T], axis=0)
    return [A1, B1, C1], [P1, Q1, R1, Z1], time.time() - tic


def CPStream_update(X, factors, G, mu=0.99, iters=20, tol=1e-5):
    """
    refer to Smith et al. Streaming Tensor Factorization for Infinite Data Sources. SDM 2018 
    """
    tic = time.time()

    A, B, C = factors
    A_, U = np.zeros(A.shape), np.zeros(A.shape)
    B_, V = np.zeros(B.shape), np.zeros(B.shape)
    C_, W = np.zeros(C.shape), np.zeros(C.shape)
    A1 = A.copy(); A2 = B.copy()
    
    # get the new row
    c = optimize(np.einsum('ir,im,jr,jm->rm',A,A,B,B,optimize=True), np.einsum('ijk,ir,jr->rk',X,A,B,optimize=True)).T
    
    R = A.shape[1]
    eye = np.eye(R)
    tmpA, tmpB = A.copy(), B.copy()
    
    for i in range(iters):
        # for A
        Si = (B.T @ B) * (mu * G + c.T @ c)

        Phi = B.T @ X[:,:,0].T + np.einsum('kr,ir,im,rm->mk',A1,A2,B,mu*G,optimize=True)
        rho = np.trace(Si) / R
        
        A_ = optimize(Si + rho * eye, Phi + rho * (A.T + U.T), reg = 0).T
        A = A_ - U; A = A / np.sqrt((A**2).sum(axis=0))
        U += A_ - A
        
        # for B
        Si = (A.T @ A) * (mu * G + c.T @ c)
        Phi = A.T @ X[:,:,0] + np.einsum('kr,ir,im,rm->mk',A2,A1,A,mu*G,optimize=True)
        rho = np.trace(Si) / R
        
        B_ = optimize(Si + rho * eye, Phi + rho * (B.T + V.T), reg = 0).T
        B = B_ - V; B = B / np.sqrt((B**2).sum(axis=0))
        V += B_ - B
        
        if (i > 2) and (la.norm(A - tmpA) / la.norm(tmpA) + la.norm(B - tmpB) / la.norm(tmpB) < tol):
            break
        tmpA = A.copy(); tmpB = B.copy()
        
    # update G
    G += mu * G + c.T @ c

    toc = time.time()
    return [A, B, np.concatenate([C, c], axis=0)], G, toc - tic



def GOCPTE_fac_update(X, factors, alpha=1):
    """
    Our efficient version for online tensor factorization
    INPUT:
        - <tensor> X: the input new tensor slice, (..., ..., ..., 1)
        - <matrix list> factors: the current factor matrix list
        - <float> alpha: the weight for balancing the new and past information
    OUTPUT:
        - <matrix list> factors: the current factor matrix list
    """
    tic = time.time()

    As = [factor.copy() for factor in factors]
    
    # get the last dim
    lhs, rhs = get_lhs_rhs_from_tensor(X, factors, len(factors)-1)
    aug_last_factor = optimize(lhs, rhs).T
    
    # update other factors
    for j in range(len(factors) - 1):
        lhs1, rhs1 = get_lhs_rhs_from_tensor(X, factors[:-1] + [aug_last_factor], j)
        lhs2, rhs2 = get_lhs_rhs_from_copy(As, factors, j)
        factors[j] = optimize(lhs1 + alpha * lhs2, rhs1 + alpha * rhs2).T
    
    # update the last factor
    lhs, rhs = get_lhs_rhs_from_copy(As, factors, len(factors)-1)
    factors[-1] = optimize(lhs, rhs).T

    return factors[:-1] + [np.concatenate([factors[-1], aug_last_factor], axis=0)], time.time() - tic 


# ----------- the following is for online tensor completion -----------

def get_lhs_rhs_mask(Omega, mask_X, factors, i):
    """
    get the lhs and rhs einsum results of the i-th factor with tensor mask
    INPUT:
        - <tensor> Omega: the tensor mask
        - <tensor> mask_X: the masked tensor
        - <matrix list> factors: the list of current factors
        - <int> i: the indices of the target factor
    OUTPUT:
        - <matrix> lhs: size (Ii, R, R), is the self-product of khatri-Rao product
        - <matrix> rhs: size (R, Ii), is the MTTKRP result 
    """
    lhs_mat, rhs_mat = [], [mask_X]
    lhs_str = ""
    rhs_str = "".join([chr(j+97) for j in range(len(factors))])
    for j in range(Omega.ndim):
        if j != i:
            lhs_str+='{}r,{}z,'.format(chr(97+j), chr(97+j))
            lhs_mat.append(factors[j]); lhs_mat.append(factors[j])
            rhs_str += ',{}r'.format(chr(j+97))
            rhs_mat.append(factors[j])
    lhs_str += "".join([chr(97+i) for i in range(Omega.ndim)]) + "->"+chr(97+i)+'rz'
    rhs_str += '->r{}'.format(chr(i+97))
    lhs_mat.append(Omega)
    lhs_ls = np.einsum(lhs_str, *lhs_mat, optimize=True)
    rhs = np.einsum(rhs_str, *rhs_mat, optimize=True)

    return lhs_ls, rhs


def get_lhs_rhs_mask_weighted(Omega, mask_X, factors, coeff, i):
    """
    get the lhs and rhs einsum results of the i-th factor with tensor mask
    INPUT:
        - <tensor> Omega: the tensor mask
        - <tensor> mask_X: the masked tensor
        - <matrix list> factors: the list of current factors
        - <list> coeff: exponential weights (in reverse order) for each time step
        - <int> i: the indices of the target factor
    OUTPUT:
        - <matrix> lhs: size (Ii, R, R), is the self-product of khatri-Rao product
        - <matrix> rhs: size (R, Ii), is the MTTKRP result 
    """
    lhs_mat, rhs_mat = [], [mask_X]
    lhs_str = ""
    rhs_str = "".join([chr(j+97) for j in range(len(factors))])
    for j in range(Omega.ndim):
        if j != i:
            lhs_str+='{}r,{}z,'.format(chr(97+j), chr(97+j))
            lhs_mat.append(factors[j]); lhs_mat.append(factors[j])
            rhs_str += ',{}r'.format(chr(j+97))
            rhs_mat.append(factors[j])
    lhs_str += "{},".format(chr(97+Omega.ndim-1)) + \
        "".join([chr(97+i) for i in range(Omega.ndim)]) + "->"+chr(97+i)+'rz'
    lhs_mat.append(coeff); lhs_mat.append(Omega)
    rhs_str += ",{}".format(chr(97+Omega.ndim-1)) + "->r{}".format(chr(i+97))
    rhs_mat.append(coeff)

    lhs_ls = np.einsum(lhs_str, *lhs_mat, optimize=True)
    rhs = np.einsum(rhs_str, *rhs_mat, optimize=True)

    return lhs_ls, rhs


# sparse strategy iteration
def cpc_als_iteration(mask_X, factors, Omega):
    """
    The CPD-ALS algorithm
    INPUT:
        - <tensor> mask_X: this is the input tensor of size (I1, I2, ..., In)
        - <matrix list> factors: the initalized factors (A1, A2, ..., An)
        - <tensor> Omega: this is the tensor mask of size (I1, I2, ..., In)
    OUTPUT:
        - <matrix list> factors: the optimized factors (A1, A2, ..., An)
    """
    tic = time.time()
    for i in range(len(factors)):
        lhs, rhs = get_lhs_rhs_mask(Omega, mask_X, factors, i)
        for k in range(factors[i].shape[0]):
            factors[i][k] = optimize(lhs[k], rhs[:, k]).T
    toc = time.time()
    return factors, toc - tic


def OnlineSGD_update(mask_X, mask, factors, lr, index=1, reg=1e-5):
    """
    This method uses stochastic gradient descent to update each factor
    INPUT:
        - <tensor> mask_X: this is the input tensor of size (I1, I2, ..., In)
        - <matrix list> factors: the initalized factors (A1, A2, ..., An)
        - <tensor> mask: this is the tensor mask of size (I1, I2, ..., In)
        - <float> lr: the learning rate
        - <int> index: in which step? the index will become large with charging weight
        - <float> reg: L2 regularization coefficient
    OUTPUT:
        - <matrix list> factors: the optimized factors (A1, A2, ..., An)
    """
    tic = time.time()
    _, R = factors[0].shape

    # get new row c
    lhs, rhs = get_lhs_rhs_mask(mask, mask_X, factors, mask.ndim-1)
    aug_last_factor = optimize(lhs[0], rhs[:, :1]).T

    # update A1, A2, A3
    rec_X = rec_from_factors(factors[:-1] + [aug_last_factor])
    grad = []
    for i in range(mask.ndim - 1):
        lhs, rhs = get_lhs_rhs_from_tensor(mask_X - mask * rec_X, factors, i)
        grad.append(optimize(lhs, rhs).T)
        
    for i in range(mask.ndim - 1):
        factors[i] = (1 - lr * reg / (index+1)) * factors[i] - lr * grad[i]
    return factors[:-1] + [np.concatenate([factors[-1], aug_last_factor], 0)], time.time() - tic


def OLSTEC_update(mask_X, mask, factors, R_ls, S_ls, mu=1e-9, Lambda=0.88):
    """
    Kasai, Online Low-Rank Tensor Subspace Tracking from Incomplete Data by CP Decomposition \
        using Recursive Least Squares, ICASSP 2016
    INPUT:
        - <tensor> mask_X: this is the input tensor of size (I1, I2, ..., In)
        - <matrix list> factors: the initalized factors (A1, A2, ..., An)
        - <tensor> mask: this is the tensor mask of size (I1, I2, ..., In)
        - <matrix list> R_ls: aux variables for LHS
        - <matrix list> L_ls: aux variables for RHS
        - <float> mu, Lambda: two coefficient
    OUTPUT:
        - <matrix list> factors: the optimized factors (A1, A2, ..., An)
        - <matrix list> R_ls: aux variables for LHS
        - <matrix list> L_ls: aux variables for RHS
    """
    tic = time.time()

    _, R = factors[0].shape
    eye = np.eye(R)

    # obtain the aug_last_factor
    lhs, rhs = get_lhs_rhs_mask(mask, mask_X, factors, mask.ndim-1)
    aug_last_factor = optimize(lhs[0], rhs[:, :1]).T

    # update the other factors

    for i in range(mask.ndim - 1):
        lhs, rhs = get_lhs_rhs_mask(mask, mask_X, factors[:-1] + [aug_last_factor], i)
        R_ls[i] = Lambda * R_ls[i] + lhs + mu * (1 - Lambda) * eye
        S_ls[i] = Lambda * S_ls[i] + rhs

        for k in range(factors[i].shape[0]):
            factors[i][k] = optimize(R_ls[i][k], S_ls[i][:, k]).T
    
    return factors[:-1] + [np.concatenate([factors[-1], aug_last_factor], 0)], \
        R_ls, S_ls, time.time() - tic


def GOCPTE_comp_update(mask_X, mask, factors, alpha=1):
    """
    Our efficient version for online tensor completion
    INPUT:
        - <tensor> X: the input new tensor slice, (..., ..., ..., 1)
        - <matrix list> factors: the current factor matrix list
        - <float> alpha: the weight for balancing the new and past information
    OUTPUT:
        - <matrix list> factors: the current factor matrix list
    """
    tic = time.time()
    As = [factor.copy() for factor in factors]

    # get a new row in the last factor
    lhs, rhs = get_lhs_rhs_mask(mask, mask_X, factors, mask.ndim-1)
    aug_last_factor = optimize(lhs[0], rhs[:, :1]).T

    for i in range(mask.ndim - 1):
        lhs1, rhs1 = get_lhs_rhs_mask(mask, mask_X, factors[:-1] + [aug_last_factor], i)
        lhs2, rhs2 = get_lhs_rhs_from_copy(As, factors, i)

        for k in range(factors[i].shape[0]):
            factors[i][k] = optimize(lhs1[k] + alpha * lhs2, \
                    rhs1[:, k] + alpha * rhs2[:, k]).T

    toc = time.time()
    return factors[:-1] + [np.concatenate([factors[-1], aug_last_factor], 0)], toc - tic