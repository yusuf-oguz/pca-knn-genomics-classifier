import numpy as np


def _qr_decompose(A):
    # QR decomposition using modified Gram-Schmidt
    n = A.shape[0]
    Q = A.copy().astype(float)
    R = np.zeros((n, n), dtype=float)
    for j in range(n):
        nrm = np.sqrt(Q[:, j] @ Q[:, j])
        if nrm < 1e-14:
            continue
        Q[:, j] /= nrm
        R[j, j] = nrm
        if j < n - 1:
            R[j, j + 1:] = Q[:, j] @ Q[:, j + 1:]
            Q[:, j + 1:] -= np.outer(Q[:, j], R[j, j + 1:])
    return Q, R


def _eig_symmetric(A, max_iter=500, tol=1e-8):
    # QR iteration to find eigenvalues and eigenvectors of a symmetric matrix
    n = A.shape[0]
    Ak = A.copy().astype(float)
    Qk = np.eye(n)
    for _ in range(max_iter):
        Q, R = _qr_decompose(Ak)
        Ak = R @ Q
        Qk = Qk @ Q
        off_diag = np.sum(np.abs(np.tril(Ak, -1)))
        if off_diag < tol:
            break
    return np.diag(Ak), Qk


def center_fit(Xtr: np.ndarray):
    # compute mean and center the training matrix
    mu = Xtr.mean(axis=0)
    Xtr_c = Xtr - mu
    return Xtr_c, mu


def center_apply(X: np.ndarray, mu: np.ndarray):
    # apply pre-computed mean to center X
    X_c = X - mu
    return X_c


def pca_svd(Xc: np.ndarray):
    N, M = Xc.shape
    # use N x N covariance (cheaper when N << M)
    C = (Xc @ Xc.T) / (N - 1)
    raw_eigvals, U = _eig_symmetric(C)
    # sort by descending eigenvalue
    idx = np.argsort(-raw_eigvals)
    eigvals = np.maximum(raw_eigvals[idx], 0.0)
    U = U[:, idx]
    # recover V: Xc = U @ diag(sigma) @ V.T => V[:,k] = Xc.T @ U[:,k] / sigma_k
    sigmas = np.sqrt(eigvals * (N - 1))
    K = min(N, M)
    sigmas_k = sigmas[:K]
    valid = sigmas_k > 1e-10
    V = np.zeros((M, K), dtype=float)
    V[:, valid] = (Xc.T @ U[:, :K][:, valid]) / sigmas_k[valid]
    scores = Xc @ V
    return eigvals, V, scores


def pca_correlated_snp_scores(V: np.ndarray, k: int) -> np.ndarray:
    # score each SNP by summing squared loadings over the top k PCs
    scores = np.sum(V[:, :k] ** 2, axis=1)
    return scores
