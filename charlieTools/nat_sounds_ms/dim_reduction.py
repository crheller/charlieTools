"""
Tools for reducing dimensionality of natural sounds data prior to decoding.

Implementing PCA and PLS for dimensionality reduction. 

Meant to deal with trial limitation issues (e.g. where n_trials << n_neurons)

CRH 04/10/2020
"""
import numpy as np

def get_one_hot_matrix(ncategories, nreps):
    # build Y matrix of one hot vectors
    Y = np.zeros((ncategories, ncategories * nreps))
    for stim in range(ncategories):
        yt = np.zeros((nreps, ncategories))
        yt[:, stim] = 1
        yt = yt.reshape(1, -1)
        Y[stim, :] = yt
    return Y