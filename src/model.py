import numpy as np
from utils import optimize, Optimizer
from scipy import linalg as la

def sparse_strategy_iteration(X, mask, factors):
    A1, A2, A3 = factors

    A1 = Optimizer(mask, [A1, A2, A3], np.einsum('ijk,jr,kr->ir',X,A2,A3,optimize=True), 0)
    A2 = Optimizer(mask, [A1, A2, A3], np.einsum('ijk,ir,kr->jr',X,A1,A3,optimize=True), 1)
    A3 = Optimizer(mask, [A1, A2, A3], np.einsum('ijk,ir,jr->kr',X,A1,A2,optimize=True), 2)
    return A1, A2, A3


def dense_strategy_iteration(X, mask, factors):
    A1, A2, A3 = factors
    X = X + (1 - mask) * np.einsum('ir,jr,kr->ijk',A1,A2,A3,optimize=True)

    # call cpd-als
    A1, A2, A3 = cpd_als_iteration(X, [A1, A2, A3])
    return A1, A2, A3


def cpd_als_iteration(X, factors):
    A1, A2, A3 = factors
    A1 = optimize(np.einsum('ir,im,jr,jm->rm',A2,A2,A3,A3), np.einsum('ijk,jr,kr->ri',X,A2,A3)).T
    A2 = optimize(np.einsum('ir,im,jr,jm->rm',A1,A1,A3,A3), np.einsum('ijk,ir,kr->rj',X,A1,A3)).T
    A3 = optimize(np.einsum('ir,im,jr,jm->rm',A1,A1,A2,A2), np.einsum('ijk,ir,jr->rk',X,A1,A2)).T
    return A1, A2, A3


# for factorization 
def OnlineCPD(X_new, factors, P, Q):
    """
    refer to Zhou et al. Accelerating Online CP Decompositions for Higher Order Tensors. KDD 2016
    Note: X_new: I x J x 1
    """
    A, B, C = factors
    P1, P2 = P; Q1, Q2 = Q

    # get the new row
    c = optimize(np.einsum('ir,im,jr,jm->rm',A,A,B,B,optimize=True), np.einsum('ijk,ir,jr->rk',X_new,A,B,optimize=True)).T
    C = np.concatenate([C, c], axis=0)

    # update P and Q
    P1 += np.einsum('ijk,jr,kr->ri',X_new,B,c,optimize=True)
    P2 += np.einsum('ijk,ir,kr->rj',X_new,A,c,optimize=True)
    Q1 += np.einsum('ir,im,jr,jm->rm',B,B,c,c,optimize=True)
    Q2 += np.einsum('ir,im,jr,jm->rm',A,A,c,c,optimize=True)

    A = optimize(Q1, P1).T
    B = optimize(Q2, P2).T
    return A, B, C, P1, P2, Q1, Q2


def CPStream(X_new, factors, G, mu=0.99, total_iter=20, tol=1e-5):
    """
    refer to Smith et al. Streaming Tensor Factorization for Infinite Data Sources. SDM 2018 
    Note: X_new: I x J x 1
    """
    A, B, C = factors
    A_, U = np.zeros(A.shape), np.zeros(A.shape)
    B_, V = np.zeros(B.shape), np.zeros(B.shape)
    C_, W = np.zeros(C.shape), np.zeros(C.shape)
    A1 = A.copy(); A2 = B.copy()
    
    # get the new row
    c = optimize(np.einsum('ir,im,jr,jm->rm',A,A,B,B,optimize=True), np.einsum('ijk,ir,jr->rk',T,A,B,optimize=True)).T
    
    R = A.shape[1]
    tmpA, tmpB = A.copy(), B.copy()
    
    for i in range(total_iter):
        # for A
        Si = (B.T @ B) * (mu * G + c.T @ c)

        Phi = B.T @ X_new[:,:,0].T + np.einsum('kr,ir,im,rm->mk',A1,A2,B,mu*G,optimize=True)
        rho = np.trace(Si) / R
        
        A_ = optimize(Si + rho * eye, Phi + rho * (A.T + U.T), reg = 0).T
        A = A_ - U; A = A / np.sqrt((A**2).sum(axis=0))
        U += A_ - A
        
        # for B
        Si = (A.T @ A) * (mu * G + c.T @ c)
        Phi = A.T @ X_new[:,:,0] + np.einsum('kr,ir,im,rm->mk',A2,A1,A,mu*G,optimize=True)
        rho = np.trace(Si) / R
        
        B_ = optimize(Si + rho * eye, Phi + rho * (B.T + V.T), reg = 0).T
        B = B_ - V; B = B / np.sqrt((B**2).sum(axis=0))
        V += B_ - B
        
        if (i > 2) and (la.norm(A - tmpA) / la.norm(tmpA) + la.norm(B - tmpB) / la.norm(tmpB) < tol):
            break
        tmpA = A.copy(); tmpB = B.copy()
        
    # update G
    G += mu * G + c.T @ c
    return A, B, np.concatenate([C, c], axis=0), G


def CPStream2(X_new, factors, total_iter=20, gamma=0.99, tol=1e-5):
    A, B, C = factors
    """
    refer to Smith et al. Streaming Tensor Factorization for Infinite Data Sources. SDM 2018
    We customize an optimizer 
    Note: X_new: I x J x 1
    """
    A1, A2, A3 = A.copy(), B.copy(), C.copy()
    coeff = np.array([gamma**i for i in range(1, C.shape[0]+1)])[::-1]

    # get the new row
    c = optimize(np.einsum('ir,im,jr,jm->rm',A,A,B,B,optimize=True), np.einsum('ijk,ir,jr->rk',X_new,A,B,optimize=True)).T
    tmpA, tmpB = A.copy(), B.copy()

    for i in range(total_iter):
        A = optimize(np.einsum('ir,im,jr,jm->rm',B,B,c,c,optimize=True) + np.einsum('ir,im,jr,jm,j->rm',B,B,C,C,coeff,optimize=True), \
                    np.einsum('ijk,jr,kr->ri',X_new,B,c,optimize=True) + np.einsum('kr,ir,im,jr,jm,j->mk',A1,A2,B,A3,C,coeff,optimize=True)).T
        B = optimize(np.einsum('ir,im,jr,jm->rm',A,A,c,c,optimize=True) + np.einsum('ir,im,jr,jm,j->rm',A,A,C,C,coeff,optimize=True), \
                    np.einsum('ijk,ir,kr->rj',X_new,A,c,optimize=True) + np.einsum('kr,ir,im,jr,jm,j->mk',A2,A1,A,A3,C,coeff,optimize=True)).T
        if (i > 2) and (la.norm(A - tmpA) / la.norm(tmpA) + la.norm(B - tmpB) / la.norm(tmpB) < tol):
            break
        tmpA = A.copy(); tmpB = B.copy()
        
    return A, B, np.concatenate([C, c], axis=0)
        

def GOCPTE_factorization(X_new, factors, alpha=1):
    """
    Our efficient version
    Note: X_new: I x J x 1
    """
    A, B, C = factors
    A1, A2, A3 = A.copy(), B.copy(), C.copy()
    
    # get c
    c = optimize(np.einsum('ir,im,jr,jm->rm',A,A,B,B,optimize=True), np.einsum('ijk,ir,jr->rk',X_new,A,B,optimize=True)).T
    A = optimize(np.einsum('ir,im,jr,jm->rm',B,B,c,c,optimize=True) + alpha * np.einsum('ir,im,jr,jm->rm',B,B,C,C,optimize=True), \
                np.einsum('ijk,jr,kr->ri',X_new,B,c,optimize=True) + alpha * np.einsum('kr,ir,im,jr,jm->mk',A1,A2,B,A3,C,optimize=True)).T
    B = optimize(np.einsum('ir,im,jr,jm->rm',A,A,c,c,optimize=True) + alpha * np.einsum('ir,im,jr,jm->rm',A,A,C,C,optimize=True), \
                np.einsum('ijk,ir,kr->rj',X_new,A,c,optimize=True) + alpha * np.einsum('kr,ir,im,jr,jm->mk',A2,A1,A,A3,C,optimize=True)).T
    C = optimize(np.einsum('ir,im,jr,jm->rm',A,A,B,B,optimize=True), np.einsum('kr,ir,im,jr,jm->mk',A3,A1,A,A2,B,optimize=True)).T
    return A, B, np.concatenate([C, c], axis=0)


def GOCPT_factorization(X, factors, alpha=1):
    """
    Our model
    Note: X: I x J x Kt
    """
    A, B, C = factors
    A1, A2, A3 = A.copy(), B.copy(), C.copy()
    
    # get c
    c = optimize(np.einsum('ir,im,jr,jm->rm',A,A,B,B,optimize=True), np.einsum('ijk,ir,jr->rk',X[:,:,-1:],A,B,optimize=True)).T
    C_ = np.concatenate([C, c], axis=0)
    A = optimize(np.einsum('ir,im,jr,jm->rm',B,B,C_,C_,optimize=True) + alpha * np.einsum('ir,im,jr,jm->rm',B,B,C,C,optimize=True), \
                np.einsum('ijk,jr,kr->ri',X,B,C_,optimize=True) + alpha * np.einsum('kr,ir,im,jr,jm->mk',A1,A2,B,A3,C,optimize=True)).T
    B = optimize(np.einsum('ir,im,jr,jm->rm',A,A,C_,C_,optimize=True) + alpha * np.einsum('ir,im,jr,jm->rm',A,A,C,C,optimize=True), \
                np.einsum('ijk,ir,kr->rj',X,A,C_,optimize=True) + alpha * np.einsum('kr,ir,im,jr,jm->mk',A2,A1,A,A3,C,optimize=True)).T
    C = optimize(np.einsum('ir,im,jr,jm->rm',A,A,B,B,optimize=True), np.einsum('kr,ir,im,jr,jm->mk',A3,A1,A,A2,B,optimize=True)).T
    return A, B, np.concatenate([C, c], axis=0)


def MAST(X_new, factors, alpha, alphaN, total_iter=20, phi=1.05, etamax=1, etaInit=1e-5, tol=1e-5):
    """
    refer to Song et al. Multi-Aspect Streaming Tensor Completion. KDD 2017
    Their settings
    alphaN = [1/5N, 1/5N, 1/5N] = [1/15, 1/15, 1/15]
    phi = 1.05; eta = 1e-4; etamax=1e6
    Note: X_new: I x J x 1 
    """
    A, B, C = factors

    # initialize ZA, ZB, ZC and YA, YB, YC, A1, A2, A3
    K, R = C.shape
    ZA, YA = np.zeros(A.shape), np.zeros(A.shape)
    ZB, YB = np.zeros(B.shape), np.zeros(B.shape)
    ZC, YC = np.zeros((K+1, R)), np.zeros((K+1, R))
    c = np.random.random((1,R))
    A1, A2, A3 = A.copy(), B.copy(), C.copy()
    eta = etaInit
    eye = np.eye(R)
    a1, a2, a3 = alphaN
    
    rec = np.einsum('ir,jr,kr->ijk',A,B,np.concatenate([C, c], axis=0),optimize=True)
    for k in range(total_iter):
        eta = min(eta * phi, etamax)
        # update factors
        C = optimize(np.einsum('ir,im,jr,jm->rm',A,A,B,B,optimize=True) + eta * eye, \
                np.einsum('kr,ir,im,jr,jm->mk',A3,A1,A,A2,B,optimize=True) + eta * ZC[:-1,:].T + YC[:-1,:].T, reg=0).T
        c = optimize(np.einsum('ir,im,jr,jm->rm',A,A,B,B,optimize=True), \
                np.einsum('ijk,ir,jr->rk',X_new,A,B,optimize=True) + eta * ZC[-1:,:].T + YC[-1:,:].T, reg=0).T
        A = optimize(np.einsum('ir,im,jr,jm->rm',B,B,c,c,optimize=True) + alpha * np.einsum('ir,im,jr,jm->rm',B,B,C,C,optimize=True) + eta * eye, \
                np.einsum('ijk,jr,kr->ri',X_new,B,c,optimize=True) + alpha * np.einsum('kr,ir,im,jr,jm->mk',A1,A2,B,A3,C,optimize=True)+ eta * ZA.T + YA.T, reg=0).T
        B = optimize(np.einsum('ir,im,jr,jm->rm',A,A,c,c,optimize=True) + alpha * np.einsum('ir,im,jr,jm->rm',A,A,C,C,optimize=True) + eta * eye, \
                np.einsum('ijk,ir,kr->rj',X_new,A,c,optimize=True) + alpha * np.einsum('kr,ir,im,jr,jm->mk',A2,A1,A,A3,C,optimize=True)+ eta * ZB.T + YB.T, reg=0).T
        
        # update auxiliary factors
        u, s, vh = la.svd(A - YA / eta, full_matrices = False)
        ZA = u @ np.diag(s * (s >= a1 / eta)) @ vh
        YA += eta * (ZA - A)
        u, s, vh = la.svd(B - YB / eta, full_matrices = False)
        ZB = u @ np.diag(s * (s >= a2 / eta)) @ vh
        YB += eta * (ZB - B)
        u, s, vh = la.svd(np.concatenate([C, c], axis=0) - YC / eta, full_matrices = False)
        ZC = u @ np.diag(s * (s >= a3 / eta)) @ vh
        YC += eta * (ZC - np.concatenate([C, c], axis=0))
        
        rec1 = np.einsum('ir,jr,kr->ijk',A,B,np.concatenate([C, c], axis=0),optimize=True)
        if la.norm(rec - rec1) / la.norm(rec) < tol:
            break

    return A, B, np.concatenate([C, c], axis=0)


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


def ConsistentApprox(T, Qb, tmp_b, Rb, Qa, tmp_a, Ra, gamma):
    """
    refer to Nion et al. Adaptive Algorithms to Track the PARAFAC Decomposition of a Third-Order Tensor. TSP 2009
    also: https://github.com/thanhtbt/ROLCP/tree/main/Algorithms/PAFARAC(2009)
    """
    # update A
    X_ = Qb @ Ra @ Qa.T
    A = X_.T @ Qb
    h_a = Qb.T @ T.reshape(-1, 1)
    theta_a = tmp_b.T @ Qb
    aug_a = np.concatenate([h_a.T, gamma * Ra @ theta_a])
    Gat, Rat = la.qr(aug_a)
    Ra = Rat[:-1, :]
    emp_a = np.zeros((Qa.shape[0]+1, Qa.shape[1]+1))
    emp_a[1:, 1:] = Qa; emp_a[0,0] = 1
    tmp_a = Qa; Qa = (emp_a @ Gat)[:, :-1]
    # update B
    aug_b = Qb @ Rb @ Qa.T
    X_ = aug_b @ Qa
    tmp_b = Qb
    Qb, Rb = la.qr(X_, mode='economic')
    aug_X = np.concatenate([h_a.T, gamma * Ra @ theta_a])
    # aug_X = np.concatenate([Qb @ Ra @ Qa.T, T.reshape(-1, 1)], axis=1)
    # tmp_b = Qb; tmp_a = Qa
    # Qb, _, Rb, Qa, _, Ra = BiSVD(aug_X, Qb.shape[1])
    # return Qb, tmp_b, Rb, Qa, tmp_a, Ra
    return Qb, tmp_b, Rb, Qa, tmp_a, Ra


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


def SDT(T, factors, W0, Wi0, V0, S0, U0, gamma):
    """
    refer to Nion et al. Adaptive Algorithms to Track the PARAFAC Decomposition of a Third-Order Tensor. TSP 2009
    also: https://github.com/thanhtbt/ROLCP/tree/main/Algorithms/PAFARAC(2009)
    """

    A0, B0, C0 = factors
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

    return A1, B1, C1, W1, Wi1, V1, S1, U1


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


def RLST(T,factors,P0,Q0,R0,Z0,N,gamma):
    """
    refer to Nion et al. Adaptive Algorithms to Track the PARAFAC Decomposition of a Third-Order Tensor. TSP 2009
    also: https://github.com/thanhtbt/ROLCP/tree/main/Algorithms/PAFARAC(2009)
    """
    A0, B0, C0 = factors
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

    return A1, B1, C1, P1, Q1, R1, Z1