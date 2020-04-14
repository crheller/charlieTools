"""
decoding tools for natural sounds analysis
"""
import numpy as np
import pandas as pd
import itertools
import pickle
import os

import logging

log = logging.getLogger()

# ================================= Results Object to hold analysis results ===============================================
class DecodingResults():
    """
    Idea is to do a little preprocessing of jackknifed results and to make extracting of 
    results for a given recording site easier.
    """

    def __init__(self, df=None):
        """
        Init with the results data frame returned at the end of the dprime analysis
        Dprime analysis is pairwise, so each analysis returns results for many pairs of 
        stimuli and for many jackknifes and for many different dimensionalities (e.g. if
        doing PLS dim reduction, might have results in 2D space, 3D space etc.)
        """

        if df is not None:
            # set attributes with list of unique stimulus combinations / categories
            self.all_stimulus_pairs = df.combo.unique().tolist()
            self.spont_stimulus_pairs = df[df['category']=='spont_spont'].combo.unique().tolist()
            self.spont_evoked_stimulus_pairs = df[df['category']=='spont_evoked'].combo.unique().tolist()
            self.evoked_stimulus_pairs = df[df['category']=='evoked_evoked'].combo.unique().tolist()

            # set attribute for number of unique dimensionality reductions
            self.unique_subspace_dims = df.n_components.unique().tolist()

            # get df columns
            self.numeric_keys = df.select_dtypes(include=['float64']).columns
            self.object_keys = df.select_dtypes(include=['object']).columns

            # get number of jacknnifes
            self.n_jacks = df.jack_idx.values.astype(np.float).max() + 1

            
            # Collapse over jackknifes. Save mean / sd of all numeric params.
            log.info("Collapsing results over jackknifes")
            # returns a dataframe for numeric results
            self.numeric_results = self._collapse_numeric(df) 
            
            # objects = arrays (e.g. arrays of eigenvectors, evals, decoding vectors).
            # returns a dictionary of data frames
            self.array_results = self._collapse_objects(df)

            # modify the above results by removing collapsing across spont repeats
            log.info("Consolidating results by combining all spont bins")
            self._consolidate_spont_results()


    def _collapse_numeric(self, df):
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


    def _collapse_objects(self, df):
        categories = ['jack_idx', 'combo', 'n_components']
        obj_cols = self.object_keys
        unique_results = list(itertools.product(self.all_stimulus_pairs, self.unique_subspace_dims))
        # have to loop, because not straightforward to mean over categories in the 
        # case of objects
        results = {}
        index = pd.MultiIndex.from_tuples(unique_results, names=['combo', 'n_components'])
        for obj in obj_cols:
            log.info("collapsing results for object: {}".format(obj))
            _df = df[[obj]+categories]
            results[obj] = pd.DataFrame(index=index, columns=['mean', 'sem'])
            for idx in index:
                try:
                    x = np.concatenate([np.expand_dims(arr, -1) for arr in _df[(_df.combo==idx[0]) & (_df.n_components==idx[1])][obj].values], -1)
                    mean = np.nanmean(x, axis=-1)
                    nanslice = [0] * (x.ndim - 1) + [None]
                    nanslice = tuple(nanslice)
                    sem = np.nanstd(x, axis=-1) / np.sqrt(np.isfinite(x[nanslice].squeeze()).sum())
                    results[obj].loc[idx]['mean'] = mean
                    results[obj].loc[idx]['sem'] = sem
                except ValueError:
                    log.info("Failed trying to concatenate object: {}".format(obj))

        return results


    def _consolidate_spont_results(self):
        """
        Idea is that there are many spont vs. spont comparisons, and for each sound
        there are many spont comparisons. In reality, there should we just one
        spont. vs. spont result and one spont vs. evoked result (per evoked sound)

        Perform the consolidation by modifying the already created numeric results df and
        object results dict / dfs. Propogate error using sem_new = np.sqrt(sem1**2 + sem2**2 + ...)
        """

        # SPONT vs. SPONT

        # 1) deal with numeric results for spont spont
        df = self.numeric_results.copy()
        mean_cols = [c for c in df.columns if '_sem' not in c]
        err_cols = [c for c in df.columns if '_sem' in c]

        spont_spont_mean = df.loc[pd.IndexSlice[self.spont_stimulus_pairs, :], :][mean_cols].groupby(by='n_components').mean()
        spont_spont_sem = df.loc[pd.IndexSlice[self.spont_stimulus_pairs, :], :][err_cols].groupby(by='n_components').apply(error_prop)
        spont_spont = pd.concat([spont_spont_mean, spont_spont_sem], axis=1)
        new_idx = pd.MultiIndex.from_tuples([pd.Categorical(('spont_spont', n_components)) 
                                for n_components in spont_spont.index], names=['combo', 'n_components'])
        spont_spont.set_index(new_idx, inplace=True)

        # drop individual spont_spont pairs from master df
        df = df[~df.index.get_level_values('combo').isin(self.spont_stimulus_pairs)]

        # add new spont results to df
        df = spont_spont.append(df)
        self.numeric_results = df


        # 2) deal with array results for spont_spont
        for obj in self.object_keys:
            df = self.array_results[obj].copy()
            sp_df = df.loc[pd.IndexSlice[self.spont_stimulus_pairs, :], :]

            if 'evecs' in obj:
                m = [np.nanmean(reflect_eigenvectors(x), axis=0) for x in [np.vstack([np.expand_dims(a, 0) for a in arr[1]['mean'].values]) for arr in sp_df.groupby('n_components')]]
                sem = [error_prop(reflect_eigenvectors(x), axis=0) for x in [np.vstack([np.expand_dims(a, 0) for a in arr[1]['sem'].values]) for arr in sp_df.groupby('n_components')]]
            else:
                m = [np.nanmean(x, axis=0) for x in [np.vstack([np.expand_dims(a, 0) for a in arr[1]['mean'].values]) for arr in sp_df.groupby('n_components')]]
                sem = [error_prop(x, axis=0) for x in [np.vstack([np.expand_dims(a, 0) for a in arr[1]['sem'].values]) for arr in sp_df.groupby('n_components')]]
            
            components = [arr[0] for arr in sp_df.groupby('n_components')]
            new_idx = pd.MultiIndex.from_tuples([pd.Categorical(('spont_spont', n_components)) 
                                for n_components in components], names=['combo', 'n_components'])
            new_df = pd.DataFrame(index=new_idx, columns=['mean', 'sem'])
            new_df['mean'] = m
            new_df['sem'] = sem

            df = df[~df.index.get_level_values('combo').isin(self.spont_stimulus_pairs)]
            df = new_df.append(df)
            self.array_results[obj] = df

        self.spont_stimulus_pairs = ['spont_spont']


        # SPONT vs. EVOKED
        df = self.numeric_results.copy()
        unique_evoked_bins = np.unique([[c.split('_')[0], c.split('_')[1]] for c in self.evoked_stimulus_pairs])

        # 1) deal with numeric results
        new_sp_ev_pairs = []
        for stim in unique_evoked_bins:
            # get all spont / evoked combos
            sp_ev = np.unique([c for c in self.spont_evoked_stimulus_pairs if stim in c])
            m = df.loc[pd.IndexSlice[sp_ev, :], :][mean_cols].groupby(by='n_components').mean()
            sem = df.loc[pd.IndexSlice[sp_ev, :], :][err_cols].groupby(by='n_components').apply(error_prop)
            sp_ev_df = pd.concat([m, sem], axis=1)
            new_idx = pd.MultiIndex.from_tuples([pd.Categorical(('spont_{}'.format(stim), n_components)) 
                                for n_components in sp_ev_df.index], names=['combo', 'n_components']) 
            sp_ev_df.set_index(new_idx, inplace=True)
            df = sp_ev_df.append(df)
            new_sp_ev_pairs.append('spont_{}'.format(stim))

        # remove inividual spont_evoked pairs 
        df = df[~df.index.get_level_values('combo').isin(self.spont_evoked_stimulus_pairs)] 

        # save updated dataframe for numeric results
        self.numeric_results = df

        # 2) deal with object results
        for obj in self.object_keys:
            df = self.array_results[obj]
            for stim in unique_evoked_bins:
                sp_ev = np.unique([c for c in self.spont_evoked_stimulus_pairs if stim in c])
                sp_df = df.loc[pd.IndexSlice[sp_ev, :], :]

                if 'evecs' in obj:
                    m = [np.nanmean(reflect_eigenvectors(x), axis=0) for x in [np.vstack([np.expand_dims(a, 0) for a in arr[1]['mean'].values]) for arr in sp_df.groupby('n_components')]]
                    sem = [error_prop(reflect_eigenvectors(x), axis=0) for x in [np.vstack([np.expand_dims(a, 0) for a in arr[1]['sem'].values]) for arr in sp_df.groupby('n_components')]]
                else:
                    m = [np.nanmean(x, axis=0) for x in [np.vstack([np.expand_dims(a, 0) for a in arr[1]['mean'].values]) for arr in sp_df.groupby('n_components')]]
                    sem = [error_prop(x, axis=0) for x in [np.vstack([np.expand_dims(a, 0) for a in arr[1]['sem'].values]) for arr in sp_df.groupby('n_components')]]
                components = [arr[0] for arr in sp_df.groupby('n_components')]
                new_idx = pd.MultiIndex.from_tuples([pd.Categorical(('spont_{}'.format(stim), n_components)) 
                                    for n_components in components], names=['combo', 'n_components'])
                new_df = pd.DataFrame(index=new_idx, columns=['mean', 'sem'])
                new_df['mean'] = m
                new_df['sem'] = sem

                df = df[~df.index.get_level_values('combo').isin(self.spont_stimulus_pairs)]
                df = new_df.append(df)
                self.array_results[obj] = df

        # update self.spont_evoked_stimulus_pairs
        self.spont_evoked_stimulus_pairs = new_sp_ev_pairs       

        # no need to return anything... just update object attributes


    def get_result(self, name, stim_pair, n_components):

        if name in self.numeric_keys:
            return [self.numeric_results.loc[pd.IndexSlice[stim_pair, n_components]][name],
                    self.numeric_results.loc[pd.IndexSlice[stim_pair, n_components]][name+'_sem']]

        elif name in self.object_keys:
            return [self.array_results[name].loc[pd.IndexSlice[stim_pair, n_components]]['mean'], 
                    self.array_results[name].loc[pd.IndexSlice[stim_pair, n_components]]['sem']]

        else:
            raise ValueError("unknown result column") 

    
    def save_pickle(self, fn):
        log.info("Saving pickle to {}".format(fn))
        with open(fn, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        log.info('Success!')

    def load_results(self, fn):
        if not os.path.isfile(fn):
            raise FileNotFoundError

        log.info("loading pickle from {}".format(fn))
        with open(fn, 'rb') as handle:
            data = pickle.load(handle)
        return data


def reflect_eigenvectors(x):
    """
    Need a special function for taking the mean of eigenvectors because we care about direction here.
    So, for example, the mean of [1, 0] and [-1, 0] should be [1, 0], not [0, 0].
    In order to do this, force cosines of all vectors in x to be postivie relative to a random
    vector in the space.

    First dim of x must be the number of vectors to be averaged. 
    Last dim of x is the indexer of eigenvectors. e.g. first eigenvector is x[0, :, 0]
    """
    # random reference vector
    rv = np.random.normal(0, 1, x.shape[-1]) 
    rv /= np.linalg.norm(rv)
    xnew = x.copy()
    for v in range(x.shape[-1]):
        for i in range(x.shape[0]):
            _x = x[0, :, v] / np.linalg.norm(x[0, :, v])
            cos = np.dot(_x, rv)
            if cos > 0:
                pass
            else:
                _x = np.negative(_x)
            
            xnew[i, :, v] = _x
  
    return xnew


def error_prop(x, axis=0):
    """
    Error propagation function.
    """
    nanslice = [0] * (x.ndim)
    nanslice[axis] = None
    nanslice = tuple(nanslice)
    if type(x) is not np.ndarray:
        return np.sqrt(x.pow(2).sum(axis=axis)) / np.isfinite(x.values[nanslice]).sum()
    else:
        return np.sqrt(np.nansum(x**2, axis=axis)) / np.isfinite(x[nanslice]).sum()

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
    usig = 0.5 * (np.cov((A - A.mean(axis=-1, keepdims=True))) + np.cov((B - B.mean(axis=-1, keepdims=True))))
    u_vec = (A.mean(axis=-1) - B.mean(axis=-1))[np.newaxis, :]

    try:
        wopt = np.matmul(np.linalg.inv(usig), u_vec.T)
    except:
        log.info('WARNING, Singular Covariance, dprime infinite, set to np.nan')
        wopt_nan = np.nan * np.ones((A.shape[0], 1))
        evals_nan = np.nan * np.ones((A.shape[0], ))
        evecs_nan = np.nan * np.ones((A.shape[0], A.shape[0]))
        u_vec_nan =  np.nan * np.ones((1, A.shape[0]))
        return np.nan, wopt_nan, evals_nan, evecs_nan, u_vec_nan


    dp2 = np.matmul(u_vec, wopt)[0][0]

    evals, evecs = np.linalg.eig(usig)

    return dp2, wopt, evals, evecs, u_vec


def _dprime_diag(A, B):
    """
    See Rumyantsev et. al 2020, Nature  and Averbeck 2006, JNP for nice derivations.
        Note typo in Rumyantsev though!
    """
    usig = 0.5 * (np.cov((A - A.mean(axis=-1, keepdims=True))) + np.cov((B - B.mean(axis=-1, keepdims=True))))
    u_vec = (A.mean(axis=-1) - B.mean(axis=-1))[np.newaxis, :]

    try:
        # get diagonal covariance matrix
        usig_diag = np.zeros(usig.shape)
        np.fill_diagonal(usig_diag, np.diagonal(usig))

        # compute numerator
        numerator = (u_vec @ np.linalg.inv(usig_diag) @ u_vec.T) ** 2

        # compute denominator
        denominator = u_vec @ np.linalg.inv(usig_diag) @ usig @ np.linalg.inv(usig_diag) @ u_vec.T
        denominator = denominator[0][0]
    except np.linalg.LinAlgError:
        log.info('WARNING, Singular Covariance, dprime infinite, set to np.nan')
        wopt_nan = np.nan * np.ones((A.shape[0], 1))
        evals_nan = np.nan * np.ones((A.shape[0], ))
        evecs_nan = np.nan * np.ones((A.shape[0], A.shape[0]))
        u_vec_nan =  np.nan * np.ones((1, A.shape[0]))
        return np.nan, wopt_nan, evals_nan, evecs_nan, u_vec_nan

    dp2 = numerator / denominator

    evals, evecs = np.linalg.eig(usig)

    # best decoding axis ignoring correlations (reduces to direction of u_vec)
    wopt_diag = np.linalg.inv(usig_diag) @ u_vec.T

    return dp2, wopt_diag, evals, evecs, u_vec

