import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import pandas as pd
from itertools import permutations
import scipy.fftpack as fp
import scipy.signal as ss
import itertools
import nems.xforms as xforms
from nems.recording import Recording
import os
import nems.db as nd
import logging
import sys

log = logging.getLogger(__name__)

def euclidean_dist(x, y):
    d = np.sqrt(sum((x-y)**2))
    return d

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def normalize_variance(x, y, z=None):
    """
    Normalize such that denominator of dprime calculation goes
    to 1.

    z, if specified, is a list of 2 other vecotrs to be normalized in
    the same manner. So, for example, z could contain [x_big_pupil_trials,
    y_small_pupil_trials] and x/y could contain the data over all
    trials (big and small pupil). We can figure out normalization factor
    for all data, then apply it to the state-dependent (z) case.
    """
    s1 = x.copy()
    s2 = y.copy()

    if (len(x.shape)==1):
        s1 = s1[:, np.newaxis]
        s2 = s2[:, np.newaxis]

    # normalize x and y
    s_all = np.concatenate((s1, s2), axis=0)
    std = np.std(s_all, axis=0)
    std[std==0] = 1
    s1 /= std
    s2 /= std

    # normalize z, if it exists
    if z is not None:
        z1 = z[0].copy()
        z2 = z[1].copy()
        z1 /= std
        z2 /= std

    if z is not None:
        return s1, s2, z1, z2
    else:
        return s1, s2

def get_epoch_combinations(d):
    """
    Given a dictionary of spikes, determine all combinations of time bins for a pairwise
    stimulus analysis. Assumes same number of time bins in all epochs
    """
    e = list(d.keys())[0]
    segs_per_epoch = d[e].shape[-1]

    epochs = list(d.keys())
    nEpochs = len(epochs)
    segments = np.arange(0, segs_per_epoch).tolist()
    repped_epochs = list(itertools.chain.from_iterable(itertools.repeat(x, segs_per_epoch) for x in epochs))
    repped_segments = segments * nEpochs
    stim_idx = [i for i in zip(repped_epochs, repped_segments)]
    combos = list(itertools.combinations(stim_idx, 2))

    return combos


def _compute_dprime(x, y):

    if (x.mean() - y.mean()) == 0:
        dprime = 0
    elif (np.var(x) + np.var(y)) == 0:
        dprime = np.nan #(x.mean() - y.mean())
    else:
        dprime = (x.mean() - y.mean()) / np.sqrt(0.5 * (np.var(x) + np.var(y)))

    return dprime

def compute_dprime_from_dicts(d1, d2=None, norm=True, LDA=True, spont_bins=None):
    """
    For case when you've folded response into epoch dictionary with keys
    = epoch name and values = numpy arrays (reps x neuron x time).

    d2 is optional. If included, it's concatenated with d1
    to find LDA/NULL axis (ex use case: state agnostic decoding axis)

    spont_bins is optional. If specified, refers to number of prestimulus bins
    of silence that precede each epoch. These will be labeled as such in the
    output data frame
    """

    if d2 is not None:
        # concatenate both dictionaries on trials by epochs
        if d1.keys() != d2.keys():
            raise ValueError("two dictionaries passed must have same epochs")
        else:
            decoding_axis_data = dict.fromkeys(d1.keys())
            for e in decoding_axis_data.keys():
                decoding_axis_data[e] = np.concatenate((d1[e], d2[e]), axis=0)
    else:
        decoding_axis_data = d1.copy()

    # get all possible combinations of epochs/segments and calculate dprime for each
    combos = get_epoch_combinations(d1)
    df_idx = [x[0][0]+'_'+str(x[0][1])+'_'+x[1][0]+'_'+str(x[1][1]) for x in combos]

    # Compute dprime for each comparison (pairs of stim_idxs) and save in df
    DP = pd.DataFrame(index=df_idx, columns=['dprime', 'similarity', 'difference',
                        'category', 'pc1_var_explained', 'pc1_proj_on_dec',
                        'pc1_var_explained_all', 'pc1_proj_on_dec_all'])
    for i, combo in enumerate(combos):
        ep1 = combo[0][0]
        seg1 = combo[0][1]
        ep2 = combo[1][0]
        seg2 =  combo[1][1]
        X_st = d1[ep1][:, :, seg1]
        Y_st = d1[ep2][:, :, seg2]
        X_all = decoding_axis_data[ep1][:, :, seg1]
        Y_all = decoding_axis_data[ep2][:, :, seg2]

        if norm:
                X_all, Y_all, X_st, Y_st = normalize_variance(X_all, Y_all, [X_st, Y_st])

        # in this (normalized) space, compute the variance explained by the first
        # pc of the residuals (for the single trial data, in this state)
        pca = PCA(n_components=1)
        residuals = np.concatenate((X_st - X_st.mean(axis=0), Y_st - Y_st.mean(axis=0)), axis=0)
        pca.fit(residuals)
        pc_ratio = pca.explained_variance_ratio_

        DP.loc[df_idx[i], 'pc1_var_explained'] = pc_ratio

        if LDA:
            d = get_LDA_axis(X_all, Y_all)
        else:
            d = get_null_axis(X_all, Y_all)

        # also, compute the magntidue of the projection of the first PC onto the decoding axis
        # both the PC and the decoding axis are normed to mag 1, so this should be standardized
        pca_norm = pca.components_ / np.linalg.norm(pca.components_) # (just to double check)
        pca_proj = np.matmul(pca_norm, d)
        DP.loc[df_idx[i], 'pc1_proj_on_dec'] = pca_proj

        # do the same as above for all data (sort of a "baseline measure of correlation")
        pca = PCA(n_components=1)
        residuals = np.concatenate((X_all - X_all.mean(axis=0), Y_all - Y_all.mean(axis=0)), axis=0)
        pca.fit(residuals)
        pc_ratio = pca.explained_variance_ratio_
        DP.loc[df_idx[i], 'pc1_var_explained_all'] = pc_ratio

        pca_norm = pca.components_ / np.linalg.norm(pca.components_) # (just to double check)
        pca_proj = np.matmul(pca_norm, d)
        DP.loc[df_idx[i], 'pc1_proj_on_dec_all'] = pca_proj

        # decide if should do X - Y or Y - X based on X_all and Y_all
        # over ALL data, force dprime to postivie. Then if sign switches
        # between subsets of data, you know you have a "noisy" discrimination
        s = np.sign(_compute_dprime(np.matmul(X_all, d), np.matmul(Y_all, d)))
        if s < 0:
            # overall dprime is negative, flip stimulus order to force positive
            X_new = Y_st.copy()
            Y_new = X_st.copy()
            Y_st = Y_new
            X_st = X_new
        else:
            # overall dprime is positive. Keep stimulus order
            pass

        # project on decoding axis
        X = np.matmul(X_st, d)
        Y = np.matmul(Y_st, d)

        # compute dprime between X and Y
        dprime = _compute_dprime(X, Y)

        # save value
        DP.loc[df_idx[i], 'dprime'] = dprime

        # compute the similarity (cosine distance)
        uX = X_all.mean(axis=0)
        uY = Y_all.mean(axis=0)
        similarity = np.dot(uX / np.linalg.norm(uX),
                            uY / np.linalg.norm(uY))
        DP.loc[df_idx[i], 'similarity'] = similarity

        # compute the abs difference in mag (normalized by overall variance
        # along mean evoked axis)
        difference = abs(np.linalg.norm(uX) - np.linalg.norm(uY))
        uStim = np.concatenate((uX[:, np.newaxis], uY[:, np.newaxis]), axis=1).mean(axis=-1)
        st_all = np.concatenate((X_all, Y_all), axis=0)
        proj = np.matmul(st_all, uStim / np.linalg.norm(uStim))
        std = np.std(proj)
        difference /= std
        DP.loc[df_idx[i], 'difference'] = difference

        # determine if a spont vs. sound, spont vs. spont, sound vs. sound comparison
        if spont_bins is not None:
            if (seg1 < spont_bins) & (seg2 < spont_bins):
                cat = 'spont_spont'
            elif (seg1 < spont_bins) & (seg2 >= spont_bins):
                cat = 'spont_sound'
            elif (seg1 >= spont_bins) & (seg2 < spont_bins):
                cat = 'spont_sound'
            else:
                cat = 'sound_sound'
            DP.loc[df_idx[i], 'category'] = cat

    return DP

def get_null_axis(x, y):
    '''
    Return unit vector from centroid of x to centroid of y
    x and y must be of dimensions: O x N, where O are observations and N are
    number of dimensions. For example, this could be trials x neurons
    '''
    ux = x.mean(axis=0)
    uy = y.mean(axis=0)

    d = ux - uy

    return unit_vector(d)

def get_LDA_axis(x, y):
    '''
    x and y must be of dimensions: O x N, where O are observations and N are
    number of dimensions. For example, this could be trials x neurons
    '''
    n_classes = 2
    if x.shape[0] != y.shape[0]:
        if x.shape[0] < y.shape[0]:
            n = x.shape[0]
            idx = np.random.choice(np.arange(0, y.shape[0]), n, replace=False)
            y = y[idx, :]
        else:
            n = y.shape[0]
            idx = np.random.choice(np.arange(0, x.shape[0]), n, replace=False)
            x = x[idx, :]

    X = np.concatenate((x[np.newaxis, :, :], y[np.newaxis, :, :]), axis=0)

    # find best axis using LDA
    # STEP 1: compute mean vectors for each category
    mean_vectors = []
    for cl in range(0, n_classes):
        mean_vectors.append(np.mean(X[cl], axis=0))

    # STEP 2.1: Compute within class scatter matrix
    n_units = X.shape[-1]
    S_W = np.zeros((n_units, n_units))
    n_observations = X.shape[1]
    for cl, mv in zip(range(0, n_classes), mean_vectors):
        class_sc_mat = np.zeros((n_units, n_units))
        for r in range(0, n_observations):
            row, mv = X[cl, r, :].reshape(n_units, 1), mv.reshape(n_units, 1)
            class_sc_mat += (row-mv).dot((row-mv).T)
        S_W += class_sc_mat

    # STEP 2.2: Compute between class scatter matrix
    overall_mean = np.mean(X, axis=0).mean(axis=0)[:, np.newaxis]
    S_B = np.zeros((n_units, n_units))
    X_fl = X.reshape(-1, n_units)
    for i in range(X_fl.shape[0]):
        S_B += (X_fl[i, :].reshape(n_units, 1) - overall_mean).dot((X_fl[i, :].reshape(n_units, 1) - overall_mean).T)

    # STEP 3: Solve the generalized eigenvalue problem for the matrix S_W(-1) S_B
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.pinv(S_W).dot(S_B))
    if np.iscomplexobj(eig_vecs):
        eig_vals, eig_vecs = np.linalg.eigh(np.linalg.pinv(S_W).dot(S_B))
    #if np.any(eig_vals<0):
    #    import pdb; pdb.set_trace()
    # STEP 4: Sort eigenvectors and find the best axis (number of nonzero eigenvalues
    # will be at most number of categories - 1)
    sorted_idx = np.argsort(eig_vals)[::-1]
    eig_vecs = eig_vecs[:, sorted_idx]
    eig_vals = eig_vals[sorted_idx]

    # STEP 5: Project data onto the top axis
    discrimination_axis = eig_vecs[:, 0]

    return discrimination_axis
