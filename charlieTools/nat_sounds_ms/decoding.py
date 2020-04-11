"""
decoding tools for natural sounds analysis. Most is based on 
analysis shown in Rumyantsev et al., 2020, Nature

CRH 04/10/2020
"""
import numpy as np
import pandas as pd
import itertools

import logging

log = logging.getLogger(__name__)

# ================================= Results Object to hold analysis results ===============================================
class DecodingResults():
    """
    Idea is to do a little preprocessing of jackknifed results and to make extracting of 
    results for a given recording site easier.
    """

    def __init__(self, df):
        """
        Init with the results data frame returned at the end of the dprime analysis
        Dprime analysis is pairwise, so each analysis returns results for many pairs of 
        stimuli and for many jackknifes and for many different dimensionalities (e.g. if
        doing PLS dim reduction, might have results in 2D space, 3D space etc.)
        """

        # set attributes with list of unique stimulus combinations / categories
        self.all_stimulus_pairs = df.combo.unique().tolist()
        self.spont_stimulus_pairs = df[df['category']=='spont_spont'].combos.unique().tolist()
        self.spont_evoked_stimulus_pairs = df[df['category']=='spont_evoked'].combos.unique().tolist()
        self.evoked_stimulus_pairs = df[df['category']=='evoked_evoked'].combos.unique().tolist()

        # set attribute for number of unique dimensionality reductions
        self.unique_subspace_dims = df.n_components.unique().tolist()

        # get df columns
        self.numeric_keys = df.select_dtypes(include=['float64']).columns
        self.object_keys = df.select_dtypes(include=['object']).columns

        # get number of jacknnifes
        self.n_jacks = df.jack_idx.max()

        
        # Collapse over jackknifes. Save mean / sd of all numeric params.
        log.info("Collapsing results over jackknifes")
        # returns a dataframe for numeric results
        self.numeric_results = self.collapse_numeric(df) 
        
        # objects = arrays (e.g. arrays of eigenvectors, evals, decoding vectors).
        # returns a dictionary of data frames
        self.array_results = self.collapse_objects(df)


    def collapse_numeric(self, df):
        """
        Collapse numeric results over jackknifes. Return a dataframe with
        mean / sd for each stimulus pair / subspace dimensionality
        """
        grouped = df.groupby(by=['jack_idx', 'combo', 'n_components'])
        num_cols = self.numeric_keys
        mean = grouped[num_cols].mean().groupby(by=['combo', 'n_components']).mean()
        sem = grouped[num_cols].mean().groupby(by=['combo', 'n_components']).sem()
        sem.columns = [c+'_sem' for c in sem.columns]
        return pd.concat([mean, sem], axis=1)


    def collapse_objects(self, df):
        categories = ['jack_idx', 'combo', 'n_components']
        obj_cols = self.object_keys
        unique_results = list(itertools.product(self.all_stimulus_pairs, self.unique_subspace_dims))
        # have to loop, because not straightforward to mean over categories in the 
        # case of objects
        results = {}
        index = pd.MultiIndex.from_tuples(unique_results, names=['combo', 'n_components'])
        for obj in obj_cols:
            _df = df[[obj]+categories]
            results[obj] = pd.DataFrame(index=index, columns=['mean', 'sem'])
            for idx in index:
                x = np.concatenate([np.expand_dims(arr, -1) for arr in _df[(_df.combo==idx[0]) & (_df.n_components==idx[1])][obj].values], -1)
                mean = x.mean(axis=-1)
                sem = x.std(axis=-1) / np.sqrt(x.shape[-1])
                results[obj].loc[idx]['mean'] = mean
                results[obj].loc[idx]['sem'] = sem

        return results

    def get_result(name, stim_pair, n_components):

        if name in self.num_cols:
            return [self.numeric_results.loc[pd.IndexSlice[stim_pair, n_components]][name],
                    self.numeric_results.loc[pd.IndexSlice[stim_pair, n_components]][name+'_sem']]
        
        elif name in self.obj_cols:
            return [self.array_results[name].loc[pd.IndexSlice[stim_pair, n_components]]['mean'], 
                    self.array_results[name].loc[pd.IndexSlice[stim_pair, n_components]]['sem']]

        else:
            raise ValueError("unknown result column") 







# =================================== Analysis functions to compute dprime =================================================

def compute_dprime(A, B, diag=False):
    """
    Compute discriminability between matrix A and matrix B
    where both are shape N neurons X N reps.

    Return:
        dprime ** 2 (with sign preserved... so that we can evaluate consitency of sign btwn est/val sets)
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
    #if dp2 < 0:
    #    dp2 = -dp2

    evals, evecs = np.linalg.eig(usig)

    return dp2, wopt, evals, evecs, u_vec


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
    
    #if denominator < 0:
    #    denominator = -denominator

    dp2 = numerator / denominator

    # best decoding axis ignoring correlations (reduces to direction of u_vec)
    wopt_diag = np.linalg.inv(usig_diag) @ u_vec.T

    return dp2, wopt_diag, evals, evecs, u_vec