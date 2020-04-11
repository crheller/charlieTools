"""
Some very basic tools for matrix reshaping, used to get data 
in the correct shape for doing decoding analysis smoothly.

(and for some preprocessing normalization / est/val splitting steps)

CRH 04/10/2020
"""

from itertools import combinations
import numpy as np

def dict_to_X(d):
    """
    Transform dictionary of spike counts (returned by nems.recording.extract_epochs)
    into a matrix of shape: (Neuron X Repetition X Stimulus X Time)
    """
    
    epochs = d.keys()
    for i, epoch in enumerate(epochs):
        r_epoch = d[epoch].transpose(1, 0, -1)[:, :, np.newaxis, :]
        if i == 0:
            X = r_epoch
        else:
            # stack on stimuli (epochs)
            X = np.append(X, r_epoch, axis=2)
    return X


def flatten_X(X):
    """
    Transform X matrix (Neuron X Stimulus X Repetition X Time) into 
    matrix of Neurons X Repetition * Stimulus * Time
    """
    if len(X.shape) != 4: raise ValueError("Input has unexpected shape")
    return X.reshape(X.shape[0], -1)


def fold_X(X_flat, nstim, nreps, nbins):
    """
    Invert the transformation done by cat_stimuli. 
        X_flat is shape (Neurons X Stimulus * Repetition * Time)
        and this fn returns matrix of shape (Neurons X Repetion X Stimulus X Time)
    """
    if len(X_flat.shape) != 2: raise ValueError("Input has unexpected shape")
    return X_flat.reshape(X_flat.shape[0], nreps, nstim, nbins)


def get_est_val_sets(X, njacks=10):
    """
    njacks 50-50 splits of the data matrix X.

    X is shape Neurons X Reps X Stim.
    """
    nreps = X.shape[1]
    n_test_reps = int(nreps / 2)
    all_idx = np.arange(0, nreps)

    # get list of all possible est / val rep idxs
    test_idx = list(combinations(all_idx, n_test_reps))

    # choose njacks random combos
    test_idx = np.array(test_idx)[np.random.choice(range(0, len(test_idx)), njacks)]
    train_idx = np.array([list(set(all_idx) - set(t)) for t in test_idx])

    est = []
    val = []

    for jk in range(njacks):
        est.append(X[:, train_idx[jk], :])
        val.append(X[:, test_idx[jk], :])

    return est, val


def scale_est_val(est, val, mean=True, sd=True):
    """
    Scale est / val datasets and return normalized data.
    If mean true, mean center data
    If sd true, normalize by std

    val gets normalized according to stats measured in est
    """
    est_new = est.copy()
    val_new = val.copy()

    if len(est) != len(val):
        raise ValueError("len of est and val must be the same")

    if (len(est[0].shape) != 3) | (len(val[0].shape) != 3):
        raise ValueError("est / val elements must be shape neuron x rep x stim")

    sets = len(est)

    for i in range(sets):
        u = est[i].mean(axis=(1, 2), keepdims=True)
        std = est[i].std(axis=(1, 2), keepdims=True)

        if mean:
            est_new[i] = est_new[i] - u
            val_new[i] = val_new[i] - u

        if sd:
            est_new[i] = est_new[i] / std
            val_new[i] = val_new[i] / std

    return est_new, val_new