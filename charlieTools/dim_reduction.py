import numpy as np
from sklearn.decomposition import PCA

class TDR():
    """
    Custom dimensionality reduction method. 
    Finds axis 1 = difference between centroids of two stimuli
    Finds first PC of noise covariance matrix
    Then, find axis 2, the axis orthogonal to axis1 that completes the plane
    defined by PC1 and axis 1
    """
    def __init__(self, tdr2_init=None, n_additional_axes=None):
        # if tdr2_init is NOT none, then use this to define TDR2 relative to dU
        self.tdr2_init = tdr2_init
        self.n_additional_axes = n_additional_axes
        return None

    def fit(self, A, B):
        """
        A and B are observation x neuron matrices. A/B are two stimuli
        """

        # get dU
        dU = A.mean(axis=0, keepdims=True) - B.mean(axis=0, keepdims=True)
        dU = dU / np.linalg.norm(dU)

        # get first PC of mean centered data
        if self.tdr2_init is None:
            pca = PCA(n_components=1)
            A0 = A - A.mean(axis=0, keepdims=True)
            B0 = B - B.mean(axis=0, keepdims=True)
            Xcenter = np.concatenate((A0, B0), axis=0)
            pca.fit(Xcenter)
            noise_axis = pca.components_
        else:
            noise_axis = self.tdr2_init

        # figure out the axis that spans the plane with dU
        noise_on_dec = (np.dot(noise_axis, dU.T)) * dU
        orth_ax = noise_axis - noise_on_dec
        orth_ax /= np.linalg.norm(orth_ax)

        weights = np.concatenate((dU, orth_ax), axis=0)

        if self.n_additional_axes is not None:
            # remove TDR projection
            A0 = A - A.mean(axis=0, keepdims=True)
            B0 = B - B.mean(axis=0, keepdims=True)
            Xcenter = np.concatenate((A0, B0), axis=0)
            Xresidual = Xcenter - Xcenter.dot(weights.T).dot(weights)

            # find n additional axes, orthogonal to TDR plane
            pca = PCA(n_components=self.n_additional_axes)
            pca.fit(Xresidual)
            noise_weights = pca.components_
            weights = np.concatenate((weights, noise_weights), axis=0)

        self.weights = weights