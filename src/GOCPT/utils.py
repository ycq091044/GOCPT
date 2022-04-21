import numpy as np
from scipy import linalg as la
import time

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
        LHS, RHS = get_new_LHS_RHS_from_tensor(X, factors, i)
        factors[i] = la.solve(LHS, RHS).T
    toc = time.time()
    return factors, toc - tic


def get_new_LHS_RHS_from_tensor(X, factors, i):
    """
    get the LHS and RHS einsum results of the i-th factor
    INPUT:
        - <tensor> X: the tensor
        - <matrix list> factors: the list of current factors
        - <int> i: the indices of the target factor
    OUTPUT:
        - <matrix> LHS: size (R,R), is the self-product of khatri-Rao product
        - <matrix> RHS: size (R, Ii), is the MTTKRP result 
    """
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
    LHS = np.einsum(ein_str,*ein_mat,optimize=True)
    RHS = np.einsum(RHS_str,*RHS_mat,optimize=True)
    return LHS, RHS


def OnlineCPD_update(X, factors, P, Q):
    """
    refer to Zhou et al. Accelerating Online CP Decompositions for \
        Higher Order Tensors. KDD 2016
    INPUT:
        - <tensor> X: the given tensor slice
        - <matrix list> factors: current factor list
        - <matrix list> P: a list of RHS for Ai
        - <matrix list> Q: a list of LHS for Ai
    OUTPUT:
        - <matrix list> factors: updated current factor list
        - <matrix list> P: updated list of RHS for Ai
        - <matrix list> Q: updated list of LHS for Ai
        - <float> toc - tic: the time span
    """
    tic = time.time()
    # get the augmented part of the last factor
    LHS, RHS = get_new_LHS_RHS_from_tensor(X, factors, len(factors) - 1)
    aug_last_factor = la.solve(LHS, RHS).T

    new_last_factor = np.concatenate([factors[-1], aug_last_factor], axis=0)
    factors[-1] = aug_last_factor 

    # update P and Q
    for i in range(len(factors)-1):
        Qn, Pn = get_new_LHS_RHS_from_tensor(X, factors, i)
        P[i] += Pn; Q[i] += Qn
        factors[i] = optimize(Q[i],P[i]).T 

    factors[-1] = new_last_factor
    toc = time.time()
    return factors, P, Q, toc - tic


def get_new_LHS_RHS_from_copy(As, factors, i):
    """
    get the LHS and RHS einsum results of the i-th factor from the nested copy factors
    INPUT:
        - <matrix list> As: the list of copied factors
        - <matrix list> factors: the list of current factors
        - <int> i: the indices of the target factor
    OUTPUT:
        - <matrix> LHS: size (R,R), is the self-product of khatri-Rao product
        - <matrix> RHS: size (R, Ii), is the MTTKRP result 
    """
    ein_str = ""
    ein_mat = []
    RHS_str = "{}r".format(chr(i+97))
    RHS_mat = [As[i]]
    for j in range(len(factors)):
        if j != i:
            ein_str += '{}r,{}z,'.format(chr(j+97), chr(j+97))
            ein_mat.append(factors[j]); ein_mat.append(factors[j])
            RHS_str += ',{}r,{}z'.format(chr(j+97), chr(j+97))
            RHS_mat.append(As[j]); RHS_mat.append(factors[j])
    ein_str = ein_str[:-1] + '->rz'
    RHS_str += '->z{}'.format(chr(i+97))
    
    LHS = np.einsum(ein_str,*ein_mat,optimize=True)
    RHS = np.einsum(RHS_str,*RHS_mat,optimize=True)
    return LHS, RHS


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
    rec_old = rec(factors[:-1] + [np.concatenate([factors[-1], aug_last_factor], 0)])

    for _ in range(iters):
        eta = min(eta * phi, eta_max)
        # update the last factors
        LHS, RHS = get_new_LHS_RHS_from_copy(As, factors, len(factors)-1)
        factors[-1] = optimize(LHS + eta * eye, RHS + eta * Zs[-1][:-1,:].T + \
            Ys[-1][:-1,:].T, reg=0).T 
        LHS, RHS = get_new_LHS_RHS_from_tensor(X, factors, len(factors)-1)
        aug_last_factor = optimize(LHS + eta * eye, RHS + eta * Zs[-1][-1:,:].T + \
            Ys[-1][-1:,:].T, reg=0).T
        
        # update other factors
        for j in range(len(factors) - 1):
            # update other factors
            LHS1, RHS1 = get_new_LHS_RHS_from_tensor(X, factors[:-1] + [aug_last_factor], j)
            LHS2, RHS2 = get_new_LHS_RHS_from_copy(As, factors, j)
            factors[j] = optimize(LHS1 + alpha * LHS2 + eta * eye, RHS1 + alpha * RHS2 + \
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

        rec_new = rec(factors[:-1] + [np.concatenate([factors[-1], aug_last_factor], 0)])
        if la.norm(rec_old - rec_new) / la.norm(rec_old) < tol:
            break
        else:
            rec_old = rec_new
    toc = time.time()

    return factors[:-1] + [np.concatenate([factors[-1], aug_last_factor], 0)], toc - tic


# used for SDT
def BiSVD(M, r, epsilon=1e-5, maxIter=50):
    """
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


def SDT_update(T, factors, aux, gamma):
    """
    refer to Nion et al. Adaptive Algorithms to Track the PARAFAC Decomposition of a Third-Order Tensor. TSP 2009
    also: https://github.com/thanhtbt/ROLCP/tree/main/Algorithms/PAFARAC(2009)
    """

    tic = time.time()
    A0, B0, C0 = factors
    W0, Wi0, V0, S0, U0 = aux
    I, J, _ = T.shape
    _, R = A0.shape

    # step 1:
    U1, S1, V1, E1 = SWASVD(T.reshape(-1,1), U0, S0, V0, gamma)
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

    toc = time.time()
    return [A1, B1, C1], [W1, Wi1, V1, S1, U1], toc - tic


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

def RLST_update(T,factors,aux,gamma):
    """
    refer to Nion et al. Adaptive Algorithms to Track the PARAFAC Decomposition of a Third-Order Tensor. TSP 2009
    also: https://github.com/thanhtbt/ROLCP/tree/main/Algorithms/PAFARAC(2009)
    """
    tic = time.time()
    A0, B0, C0 = factors
    P0,Q0,R0,Z0 = aux
    I, J, _ = T.shape
    _, R = A0.shape
    x = T.reshape(-1,1)

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

    toc = time.time()
    return [A1, B1, C1], [P1, Q1, R1, Z1], toc - tic


def CPStream_update(X, factors, G, mu=0.99, iters=20, tol=1e-5):
    """
    refer to Smith et al. Streaming Tensor Factorization for Infinite Data Sources. SDM 2018 
    Note: X_new: I x J x 1
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



def GOCPT_update(X, factors, alpha=1):
    """
    Our efficient version
    Note: X_new: I x J x 1
    """
    tic = time.time()

    As = [factor.copy() for factor in factors]
    
    # get the last dim
    LHS, RHS = get_new_LHS_RHS_from_tensor(X, factors, len(factors)-1)
    aug_last_factor = optimize(LHS, RHS).T
    
    # update other factors
    for j in range(len(factors) - 1):
        LHS1, RHS1 = get_new_LHS_RHS_from_tensor(X, factors[:-1] + [aug_last_factor], j)
        LHS2, RHS2 = get_new_LHS_RHS_from_copy(As, factors, j)
        factors[j] = optimize(LHS1 + alpha * LHS2, RHS1 + alpha * RHS2).T
    
    # update the last factor
    LHS, RHS = get_new_LHS_RHS_from_copy(As, factors, len(factors)-1)
    factors[-1] = optimize(LHS, RHS).T

    toc = time.time()
    return factors[:-1] + [np.concatenate([factors[-1], aug_last_factor], axis=0)], toc - tic 