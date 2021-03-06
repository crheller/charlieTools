import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import pandas as pd
from itertools import permutations
import scipy.fftpack as fp
import scipy.signal as ss
import itertools
import os
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

    if np.isclose(x.mean() - y.mean(), 0, atol=0.01):
        dprime = 0
    elif np.isclose(np.var(x) + np.var(y), 0, atol=0.01):
        dprime = np.nan #(x.mean() - y.mean())
    else:
        dprime = (x.mean() - y.mean()) / np.sqrt(0.5 * (np.var(x) + np.var(y)))

    return dprime

def compute_dprime_from_dicts(d1, d2=None, norm=True, LDA=True, spont_bins=None, equal_tbin=False, verbose=False):
    """
    For case when you've folded response into epoch dictionary with keys
    = epoch name and values = numpy arrays (reps x neuron x time).

    d2 is optional. If included, it's used to compute the decoding axis.

    spont_bins is optional. If specified, refers to number of prestimulus bins
    of silence that precede each epoch. These will be labeled as such in the
    output data frame

    equal_tbin. Optional. If True, only compare epochs of equivalent time bins. 
        For example: compare REF_0 to TAR_0 but NOT REF_0 to TAR_1

    CRH update docstring 02/03/2020
    returns df with the following fields per pair of stimuli:

        d1 (or other) results:
            dprime: discriminability 
            category: specify if sound vs. sound comparison or sound vs. spont etc.
            pc1_var_explained: variance explained of first PC of residuals for *d1*
            pc1_proj_on_dec: cosine similarity between first PC of residuals for *d1* and decoding axis
            stim1_pc1_proj_on_dec: same as ^ for only data from first stimulus in pair
            stim2_pc1_proj_on_dec: "^" for second stimulus in pair
            stim1_pc1_proj_on_u1: cosine similarity between pc1 of stim1 and mean of stim1
            stim2_pc1_proj_on_u2: cosine similarity between pc1 of stim2 and mean of stim2

        d2 (decoding data) stats (or d1 if d2 is None):
            similarity: cosine similarity between pairs of stimuli in *d2* (if d2 is none, this is d1)
            difference: normalized euc. distance between pairs of stimuli in *d2* (if d2 is none, this is d1)
            pc1_var_explained_all: variance explained of first PC of residuals for *d2*
            pc1_proj_on_dec_all: cosine similarity between first PC of residuals for *d2* and decoding axis
    """

    if d2 is not None:
        # d2 is the data to be used for defining the decoding axis
        if d1.keys() != d2.keys():
            raise ValueError("two dictionaries passed must have same epochs")
        else:
            decoding_axis_data = d2.copy()
    else:
        decoding_axis_data = d1.copy()

    # get all possible combinations of epochs/segments and calculate dprime for each
    combos = get_epoch_combinations(d1)
    if equal_tbin:
        combos = [c for c in combos if c[0][1] == c[1][1]] 
    print(len(combos))
    df_idx = [x[0][0]+'_'+str(x[0][1])+'_'+x[1][0]+'_'+str(x[1][1]) for x in combos]

    # Compute dprime for each comparison (pairs of stim_idxs) and save in df
    DP = pd.DataFrame(index=df_idx, columns=['dprime', 'similarity', 'difference',
                        'category', 'pc1_var_explained', 'pc1_proj_on_dec',
                        'pc1_var_explained_all', 'pc1_proj_on_dec_all',
                        'stim1_pc1_proj_on_dec', 'stim2_pc1_proj_on_dec',
                        'stim1_pc1_proj_on_u1', 'stim2_pc1_proj_on_u2'])
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
        # this will only be different than above if decoding axis is defined over different data
        # i.e. 'all' refers to the data used to compute decoding axis
        pca = PCA(n_components=1)
        residuals = np.concatenate((X_all - X_all.mean(axis=0), Y_all - Y_all.mean(axis=0)), axis=0)
        pca.fit(residuals)
        pc_ratio = pca.explained_variance_ratio_
        DP.loc[df_idx[i], 'pc1_var_explained_all'] = pc_ratio

        pca_norm = pca.components_ / np.linalg.norm(pca.components_) # (just to double check)
        pca_proj = np.matmul(pca_norm, d)
        DP.loc[df_idx[i], 'pc1_proj_on_dec_all'] = pca_proj

        # Finally, compute some stimulus specific stats. i.e. For this combo of two 
        # stimuli, compute the first pc of each individually, and see how much it 
        # 1) overlaps with decoding axis and 2) overlaps with its mean evoked axis
        pca = PCA(n_components=1)
        pca.fit(X_st)
        pca_norm = pca.components_ / np.linalg.norm(pca.components_)
        DP.loc[df_idx[i], 'stim1_pc1_proj_on_dec'] = np.matmul(pca_norm, d)

        unorm1 = X_st.mean(axis=0)
        unorm1 /= np.linalg.norm(unorm1)
        DP.loc[df_idx[i], 'stim1_pc1_proj_on_u1'] = np.matmul(pca_norm, unorm1)

        pca = PCA(n_components=1)
        pca.fit(Y_st)
        pca_norm = pca.components_ / np.linalg.norm(pca.components_)
        DP.loc[df_idx[i], 'stim2_pc1_proj_on_dec'] = np.matmul(pca_norm, d)

        unorm2 = Y_st.mean(axis=0)
        unorm2 /= np.linalg.norm(unorm2)
        DP.loc[df_idx[i], 'stim2_pc1_proj_on_u2'] = np.matmul(pca_norm, unorm2)

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

        # the following two metrics are over the decoding data, not the 
        # actual data being discriminated!
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
        else:
            DP['category'] = 'sound_sound'

    if verbose:
        return DP, d
    else:
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


def compute_discrimination_from_dict(d, norm=True):
    """
            distance1 = euclidean_dist(st1[i, :], psth1)
            distance2 = euclidean_dist(st1[i, :], psth2)

            if distance1 < distance2:
                correct.append(True)
            else:
                correct.append(False)om_dicts` function.

    For each pair of stim combinations in the dictionary (a stim combo is an individual epoch/bin),
    compute the % of single trials correctly identified (based on their euc distance to each of the PSTHs)

    d should have keys epochs and values ndarray of trials x neuron time
    """

    r = d.copy()

    combos = get_epoch_combinations(r)

    df_idx = [x[0][0]+'_'+str(x[0][1])+'_'+x[1][0]+'_'+str(x[1][1]) for x in combos]

    # Compute performance for each comparison (pairs of stim_idxs) and save in df
    # Also compute raw distance between the PSTHs
    df = pd.DataFrame(index=df_idx, columns=['fraction_correct', 'raw_distance'])

    for i, combo in enumerate(combos):
        ep1 = combo[0][0]
        seg1 = combo[0][1]
        ep2 = combo[1][0]
        seg2 =  combo[1][1]

        st1 = r[ep1][:, :, seg1]
        st2 = r[ep2][:, :, seg2]
        psth1 = r[ep1][:, :, seg1].mean(axis=0)
        psth2 = r[ep2][:, :, seg2].mean(axis=0)

        total_trials = st1.shape[0] + st2.shape[0]
        correct = []
        for j in range(st1.shape[0]):
            distance1 = euclidean_dist(st1[j, :], psth1)
            distance2 = euclidean_dist(st1[j, :], psth2)

            if distance1 < distance2:
                correct.append(True)
            else:
                correct.append(False)

        for j in range(st2.shape[0]):
            distance1 = euclidean_dist(st2[j, :], psth1)
            distance2 = euclidean_dist(st2[j, :], psth2)

            if distance2 < distance1:
                correct.append(True)
            else:
                correct.append(False)

        fraction_correct = np.sum(correct) / total_trials
        raw_distance = euclidean_dist(psth1, psth2)

        df.loc[df_idx[i]]['fraction_correct'] = fraction_correct
        df.loc[df_idx[i]]['raw_distance'] = raw_distance

    return df

def compute_pairwise_euclidean_distances(d):
    """
    For each unique bin, compute the mean/std of within sound Euc. distances
        and the mean/std of across sound Euc. distances.
    """

    # get list of unique bins
    e = list(d.keys())[0]
    segs_per_epoch = d[e].shape[-1]

    epochs = list(d.keys())
    nEpochs = len(epochs)
    segments = np.arange(0, segs_per_epoch).tolist()
    repped_epochs = list(itertools.chain.from_iterable(itertools.repeat(x, segs_per_epoch) for x in epochs))
    repped_segments = segments * nEpochs
    stim_idx = [i for i in zip(repped_epochs, repped_segments)]
    df_idx = [x[0]+'_'+str(x[1]) for x in stim_idx]
    # create dataframe to hold results
    df = pd.DataFrame(index=df_idx, columns=['within_mean', 'within_std', 'within_reps', 'within_pairs',
                                'across_mean', 'across_std', 'across_reps', 'across_pairs'])

    for i, stim in enumerate(stim_idx):
        r = d[stim[0]][:, :, stim[1]]

        # compute within stim metrics
        nreps = r.shape[0]
        combos = list(itertools.combinations(range(0, nreps), 2))
        dist = []
        for c in combos:
            dist.append(euclidean_dist(r[c[0], :], r[c[1], :]))

        df.loc[df_idx[i]]['within_mean'] = np.mean(dist)
        df.loc[df_idx[i]]['within_std'] = np.std(dist)
        df.loc[df_idx[i]]['within_reps'] = nreps
        df.loc[df_idx[i]]['within_pairs'] = len(dist)

        # compute across stim metrics
        dist = []
        for stim2 in stim_idx:
            if (stim2[0] != stim[0]) | (stim2[1] != stim[1]):
                r2 = d[stim2[0]][:, :, stim2[1]]
                for rep in range(nreps):
                    nreps2 = r2.shape[0]
                    for rep2 in range(nreps2):
                        dist.append(euclidean_dist(r[rep, :], r2[rep2, :]))
        df.loc[df_idx[i]]['across_mean'] = np.mean(dist)
        df.loc[df_idx[i]]['across_std'] = np.std(dist)
        df.loc[df_idx[i]]['across_reps'] = nreps2
        df.loc[df_idx[i]]['across_pairs'] = len(dist)

    return df
