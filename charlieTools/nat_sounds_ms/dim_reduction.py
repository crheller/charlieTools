"""
Tools for reducing dimensionality of natural sounds data prior to decoding.

Implementing PCA and PLS for dimensionality reduction. 

Meant to deal with trial limitation issues (e.g. where n_trials << n_neurons)

CRH 04/10/2020
"""
import numpy as np
from sklearn.decomposition import PCA

def get_one_hot_matrix(ncategories, nreps):
    # build Y matrix of one hot vectors
    Y = np.zeros((ncategories, ncategories * nreps))
    for stim in range(ncategories):
        yt = np.zeros((nreps, ncategories))
        yt[:, stim] = 1
        yt = yt.reshape(1, -1)
        Y[stim, :] = yt
    return Y


class TDR():
    """
    Custom dimensionality reduction method. 
    Finds axis 1 = difference between centroids of two stimuli
    Finds first PC of noise covariance matrix
    Then, find axis 2, the axis orthogonal to axis1 that completes the plane
    defined by PC1 and axis 1
    """
    def __init__(self):
        return None

    def fit(self, x, y):
        """
        x is shape observation x dim (neuron)
        y is one-hot vector of shape observation x 2 and specifies the ID of each stimulus
        """
        # get stim A and stim B matrices
        A = x[y[:,0]==1, :]
        B = x[y[:,1]==1, :]

        # get dU
        dU = A.mean(axis=0, keepdims=True) - B.mean(axis=0, keepdims=True)
        dU /= np.linalg.norm(dU)

        # get first PC of mean centered data
        pca = PCA(n_components=1)
        A0 = A - A.mean(axis=0, keepdims=True)
        B0 = B - B.mean(axis=0, keepdims=True)
        Xcenter = np.concatenate((A0, B0), axis=0)
        pca.fit(Xcenter)
        noise_axis = pca.components_

        # figure out the axis that spans the plane with dU
        noise_on_dec = (np.dot(noise_axis, dU.T)) * dU
        orth_ax = noise_axis - noise_on_dec
        orth_ax /= np.linalg.norm(orth_ax)

        weights = np.concatenate((dU, orth_ax), axis=0)

        self.weights = weights