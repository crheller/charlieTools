from sklearn.decomposition import PCA 
import numpy as np 


def get_noise_axis_per_est(d, keys=None, njacks=None):
    """
    Get pooled noise data first PC for each est set. If keys is not 
    None, only use data in d[keys]

    d is a resp dictionary with est / val sets. Compute an axis for each est set
    """

    if keys is None:
        keys = d.keys()
    
    if njacks is None:
        raise ValueError

    noise_axes = []
    for j in range(0, njacks):
        for i, k in enumerate(keys):
            try:
                if i == 0:
                    X = d[k]['est'][j].squeeze()
                else:
                    X = np.concatenate((X, d[k]['est'][j].squeeze()), axis=0)
            except:
                pass

        # get first pc
        pca = PCA(n_components=1)
        noise_axis = pca.fit(X).components_
        noise_axes.append(noise_axis)

    return noise_axes


class TDR():
    """
    Custom dimensionality reduction method. 
    Finds axis 1 = difference between centroids of two stimuli
    Finds first PC of noise covariance matrix
    Then, find axis 2, the axis orthogonal to axis1 that completes the plane
    defined by PC1 and axis 1
    """
    def __init__(self, tdr2_init=None):
        # if tdr2_init is NOT none, then use this to define TDR2 relative to dU
        self.tdr2_init = tdr2_init
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

        self.weights = weights
