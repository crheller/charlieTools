"""
decoding tools for natural sounds analysis. Most is based on 
analysis shown in Rumyantsev et al., 2020, Nature

CRH 04/10/2020
"""
import numpy as np

def compute_dprime(A, B, diag=False):
    """
    Compute discriminability between matrix A and matrix B
    where both are shape N neurons X N reps.

    Return:
        dprime
        decoding axis (with norm 1)
        evals: (of mean covariance matrix)
        evecs: (of mean covariance matrix)
        dU: <A> - <B>
    """
    if A.shape[0] > A.shape[1]:
        raise ValueError("Number of dimensions greater than number of observations. Unstable")

    if A.shape[0] != B.shape[0]:
        raise ValueError("Number of dimensions do not match between conditions")

    if diag:
        dprime, wopt, evals, evecs, dU = _dprime_diag(A, B)

    else:
        dprime, wopt, evals, evecs, dU = _dprime(A, B)

    return dprime, wopt, evals, evecs, dU 


def _dprime(A, B):
    """
    See Rumyantsev et. al 2020, Nature for nice derivation
    """
    usig = 0.5 * (np.cov((A.T - A.mean(axis=-1)).T) + np.cov((B.T - B.mean(axis=-1)).T))
    u_vec = (A.mean(axis=-1) - B.mean(axis=-1))[np.newaxis, :]

    try:
        wopt = np.matmul(np.linalg.inv(usig), u_vec.T)
    except:
        print('WARNING, Singular Covariance, dprime infinite, set to np.nan')
        return np.nan, np.nan, np.nan, np.nan, np.nan

    dp2 = np.matmul(u_vec, wopt)[0][0]
    if dp2 < 0:
        dp2 = -dp2

    evals, evecs = np.linalg.eig(usig)

    return np.sqrt(dp2), wopt, evals, evecs, u_vec


def _dprime_diag(A, B):
    """
    See Rumyantsev et. al 2020, Nature for nice derivation
    """

    # get numerator (optimal dprime)
    dp, _, evals, evecs, _ = _dprime(A, B)
    numerator = dp ** 2

    usig = 0.5 * (np.cov((A.T - A.mean(axis=-1)).T) + np.cov((B.T - B.mean(axis=-1)).T))
    u_vec = (A.mean(axis=-1) - B.mean(axis=-1))[np.newaxis, :]

    # get denominator
    usig_diag = np.zeros(usig.shape)
    np.fill_diagonal(usig_diag, np.diagonal(usig))
    denominator = u_vec @ np.linalg.inv(usig_diag) @ (usig @ np.linalg.inv(usig_diag)) @ u_vec.T
    denominator = denominator[0][0]
    if denominator < 0:
        denominator = -denominator

    dp2 = numerator / denominator

    # best decoding axis ignoring correlations (reduces to direction of u_vec)
    wopt_diag = np.linalg.inv(usig_diag) @ u_vec.T

    return np.sqrt(dp2), wopt_diag, evals, evecs, u_vec