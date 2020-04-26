"""
decoding tools for natural sounds analysis
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
import itertools
import pickle
import os

import nems_lbhb.baphy as nb

import charlieTools.nat_sounds_ms.preprocessing as nat_preproc
import charlieTools.nat_sounds_ms.dim_reduction as dr

import logging

log = logging.getLogger()

# ================================= Results Object to hold analysis results ===============================================
class DecodingResults():
    """
    Idea is to do a little preprocessing of jackknifed results and to make extracting of 
    results for a given recording site easier.
    """

    def __init__(self, df=None, pupil_range=None):
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

            # add dataframe detailing pupil results for each stim at this site
            self.pupil_range = pupil_range


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
                    x = np.concatenate([np.expand_dims(arr, 0) for arr in _df[(_df.combo==idx[0]) & (_df.n_components==idx[1])][obj].values], 0)
                    if 'evecs' in obj:
                        # reflect vectors so that there's no sign ambiguity when averaging
                        x = reflect_eigenvectors(x)
                    mean = np.nanmean(x, axis=0)
                    nanslice = [slice(0, x.shape[0], 1)] + [0] * (x.ndim - 1)
                    nanslice = tuple(nanslice)
                    sem = np.nanstd(x, axis=0) / np.sqrt(np.isfinite(x[nanslice].squeeze()).sum())
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
        self.numeric_results = df.copy()


        # 2) deal with array results for spont_spont
        for obj in self.object_keys:
            df = self.array_results[obj].copy()
            sp_df = df.loc[pd.IndexSlice[self.spont_stimulus_pairs, :], :]

            if 'evecs' in obj:
                m = [np.nanmean(reflect_eigenvectors(x), axis=0) for x in [np.stack([a for a in arr[1]['mean'].values]) for arr in sp_df.groupby('n_components')]]
                sem = [error_prop(x, axis=0) for x in [np.stack([a for a in arr[1]['sem'].values]) for arr in sp_df.groupby('n_components')]]
            else:
                m = [np.nanmean(x, axis=0) for x in [np.stack([a for a in arr[1]['mean'].values]) for arr in sp_df.groupby('n_components')]]
                sem = [error_prop(x, axis=0) for x in [np.stack([a for a in arr[1]['sem'].values]) for arr in sp_df.groupby('n_components')]]
            
            components = [arr[0] for arr in sp_df.groupby('n_components')]
            new_idx = pd.MultiIndex.from_tuples([pd.Categorical(('spont_spont', n_components)) 
                                for n_components in components], names=['combo', 'n_components'])
            new_df = pd.DataFrame(index=new_idx, columns=['mean', 'sem'])
            new_df['mean'] = m
            new_df['sem'] = sem

            df = df[~df.index.get_level_values('combo').isin(self.spont_stimulus_pairs)]
            df = new_df.append(df)
            
            self.array_results[obj] = df.copy()

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
        self.numeric_results = df.copy()

        # 2) deal with object results
        for obj in self.object_keys:
            df = self.array_results[obj].copy()
            for stim in unique_evoked_bins:
                sp_ev = np.unique([c for c in self.spont_evoked_stimulus_pairs if stim in c])
                sp_df = df.loc[pd.IndexSlice[sp_ev, :], :]

                if 'evecs' in obj:
                    m = [np.nanmean(reflect_eigenvectors(x), axis=0) for x in [np.stack([a for a in arr[1]['mean'].values]) for arr in sp_df.groupby('n_components')]]
                    sem = [error_prop(x, axis=0) for x in [np.stack([a for a in arr[1]['sem'].values]) for arr in sp_df.groupby('n_components')]]
                else:
                    m = [np.nanmean(x, axis=0) for x in [np.stack([a for a in arr[1]['mean'].values]) for arr in sp_df.groupby('n_components')]]
                    sem = [error_prop(x, axis=0) for x in [np.stack([a for a in arr[1]['sem'].values]) for arr in sp_df.groupby('n_components')]]
                components = [arr[0] for arr in sp_df.groupby('n_components')]
                new_idx = pd.MultiIndex.from_tuples([pd.Categorical(('spont_{}'.format(stim), n_components)) 
                                    for n_components in components], names=['combo', 'n_components'])
                new_df = pd.DataFrame(index=new_idx, columns=['mean', 'sem'])
                new_df['mean'] = m
                new_df['sem'] = sem

                df = df[~df.index.get_level_values('combo').isin(self.spont_evoked_stimulus_pairs)]
                df = new_df.append(df)
                self.array_results[obj] = df

        # update self.spont_evoked_stimulus_pairs
        self.spont_evoked_stimulus_pairs = new_sp_ev_pairs       

        # no need to return anything... just update object attributes


    def get_result(self, name, stim_pair, n_components):

        if name in self.numeric_keys:
            if stim_pair is None:
                return [self.numeric_results.loc[pd.IndexSlice[:, n_components], name],
                        self.numeric_results.loc[pd.IndexSlice[:, n_components], name+'_sem']]
            elif n_components is None:
                return [self.numeric_results.loc[pd.IndexSlice[stim_pair, :], name],
                        self.numeric_results.loc[pd.IndexSlice[stim_pair, :], name+'_sem']]
            else:
                return [self.numeric_results.loc[pd.IndexSlice[stim_pair, n_components], name],
                        self.numeric_results.loc[pd.IndexSlice[stim_pair, n_components], name+'_sem']]

        elif name in self.object_keys:
            if stim_pair is None:
                return [self.array_results[name].loc[pd.IndexSlice[:, n_components], 'mean'], 
                        self.array_results[name].loc[pd.IndexSlice[:, n_components], 'sem']]
            elif n_components is None:
                return [self.array_results[name].loc[pd.IndexSlice[stim_pair, :], 'mean'], 
                        self.array_results[name].loc[pd.IndexSlice[stim_pair, :], 'sem']]
            else:
                return [self.array_results[name].loc[pd.IndexSlice[stim_pair, n_components], 'mean'], 
                        self.array_results[name].loc[pd.IndexSlice[stim_pair, n_components], 'sem']]


        else:
            raise ValueError("unknown result column") 

    
    def slice_array_results(self, name, stim_pair, n_components, idx=None):
        """
        Specific fn to slice into a given index of the array result. For example,
        evecs_test might be a 2x2 array for each stim_pair, when n_components=2.
        If we only want the first eigenvector, we'd say:
            slice_array_results('evecs_test', None, 2, idx=[None, 0])
        """
        result = self.get_result(name, stim_pair, n_components)

        if type(result[0]) is np.ndarray:
            mean = self._slice_array(result[0], idx)
            sem = self._slice_array(result[1], idx)

        else:
            mean = self._slice_series(result[0], idx)
            sem = self._slice_series(result[1], idx)

        return [mean, sem]

    
    def _slice_array(self, x, idx=None):
        if idx is None:
            return x
        else:
            idx = np.array(idx)
            if np.any(idx == None):
                none_index = np.argwhere(idx == None)
                for ni in none_index[:, 0]:
                    idx[ni] = slice(0, x.shape[ni])
            idx = tuple(idx)
            return x[idx]

    def _slice_series(self, x, idx=None):
        if idx is None:
            return x

        else:
            newx = x.copy()
            newcol = [self._slice_array(_x, idx) for _x in x.values]
            newx[x.index] = pd.Series(newcol)
            return newx

    
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
    xnew = x.copy()
    for v in range(x.shape[-1]):
        cum_sum = x[0, :, v]
        cum_sum /= np.linalg.norm(cum_sum)
        for i in np.arange(1, x.shape[0]):   
            if np.any(np.isnan(x[i, :, v])):
                xnew[i, :, v] = x[i, :, v]
            else:
                cos = cum_sum.dot(x[i, :, v])
                if cos > 0:
                    cum_sum += x[i, :, v]
                    cum_sum /= np.linalg.norm(cum_sum)

                else:
                    cum_sum += np.negative(x[i, :, v])
                    cum_sum /= np.linalg.norm(cum_sum)
                    xnew[i, :, v] = np.negative(x[i, :, v])
  
    return xnew


def unit_vectors(x):
    """
    xdim = number components x nvectors
    return matrix of same shape where each vector in range xdim.shape[-1] 
    is norm 1
    """
    xnew = x.copy()
    for v in range(x.shape[-1]):
        xnew[:, v] = x[:, v] / np.linalg.norm(x[:, v])
    return xnew


def error_prop(x, axis=0):
    """
    Error propagation function.
    """
    nanslice = [0] * (x.ndim)
    nanslice[axis] = slice(0, x.shape[0], 1)
    nanslice = tuple(nanslice)
    if type(x) is not np.ndarray:
        return np.sqrt(x.pow(2).sum(axis=axis)) / np.isfinite(x.values[nanslice]).sum()
    else:
        return np.sqrt(np.nansum(x**2, axis=axis)) / np.isfinite(x[nanslice]).sum()


# =============================================== random helpers ==========================================================
# assortment of helper functions to clean up cache script.
def get_results_columns():
    columns = [
        'dp_opt_test', 
        'dp_diag_test', 
        'wopt_test', 
        'wdiag_test', 
        'var_explained_test',
        'evals_test', 
        'evecs_test', 
        'dU_test',
        'dp_opt_train', 
        'dp_diag_train', 
        'wopt_train', 
        'wdiag_train', 
        'var_explained_train',
        'evals_train', 
        'evecs_train', 
        'dU_train', 
        'dU_mag_test', 
        'dU_dot_evec_test', 
        'cos_dU_wopt_test', 
        'dU_dot_evec_sq_test', 
        'evec_snr_test', 
        'cos_dU_evec_test',
        'dU_mag_train', 
        'dU_dot_evec_train', 
        'cos_dU_wopt_train', 
        'dU_dot_evec_sq_train', 
        'evec_snr_train', 
        'cos_dU_evec_train',
        'jack_idx', 
        'n_components', 
        'combo', 
        'category', 
        'site'
    ]


    return columns


def do_pca_dprime_analysis(xtrain, xtest, nreps_train, nreps_test, ptrain_mask=None, ptest_mask=None):
        """
        perform pca on the trail averaged response, project into 2D space, compute dprime and associated metrics
        return results in a dictionary
        """
        
        # perform trial averaged PCA (bc pairwise comparison, can only do PCA dim = 2)
        ncells = xtrain.shape[0]
        xfold = nat_preproc.fold_X(xtrain, nreps=nreps_train, nstim=2, nbins=1)
        xtrain_trial_average = xfold.mean(axis=1)[:, np.newaxis, :, np.newaxis]
        xtrain_trial_average = xtrain_trial_average.reshape(ncells, -1)

        pca = PCA(n_components=2)
        pca.fit(xtrain_trial_average.T)
        pca_weights = pca.components_

        xtrain_pca = (xtrain.T @ pca_weights.T).T
        xtest_pca = (xtest.T @ pca_weights.T).T

        xtrain_pca = nat_preproc.fold_X(xtrain_pca, nreps=nreps_train, nstim=2, nbins=1).squeeze()
        xtest_pca = nat_preproc.fold_X(xtest_pca, nreps=nreps_test, nstim=2, nbins=1).squeeze()

        pca_train_var = np.var(xtrain_pca.T @ pca_weights)  / np.var(xtrain)
        pca_test_var = np.var(xtest_pca.T @ pca_weights)  / np.var(xtest)

        # compute dprime metrics raw 
        pca_dp_train, pca_wopt_train, pca_evals_train, pca_evecs_train, pca_dU_train = \
                                compute_dprime(xtrain_pca[:, :, 0], xtrain_pca[:, :, 1])
        pca_dp_test, pca_wopt_test, pca_evals_test, pca_evecs_test, pca_dU_test = \
                                compute_dprime(xtest_pca[:, :, 0], xtest_pca[:, :, 1], wopt=pca_wopt_train)

        # overwrite test decoder with training, since it's fixed
        pca_wopt_test = pca_wopt_train

        # compute dprime metrics diag decoder
        pca_dp_train_diag, pca_wopt_train_diag, _, _, x = \
                                compute_dprime(xtrain_pca[:, :, 0], xtrain_pca[:, :, 1], diag=True)
        pca_dp_test_diag, pca_wopt_test_diag, _, _, _ = \
                                compute_dprime(xtest_pca[:, :, 0], xtest_pca[:, :, 1], diag=True)

        # caculate additional metrics
        dU_mag_train         = np.linalg.norm(pca_dU_train)
        dU_dot_evec_train    = pca_dU_train.dot(pca_evecs_train)
        cos_dU_wopt_train    = abs(unit_vectors(pca_dU_train.T).T.dot(unit_vectors(pca_wopt_train)))
        dU_dot_evec_sq_train = pca_dU_train.dot(pca_evecs_train) ** 2
        evec_snr_train       = dU_dot_evec_sq_train / pca_evals_train
        cos_dU_evec_train    = abs(unit_vectors(pca_dU_train.T).T.dot(pca_evecs_train))

        dU_mag_test         = np.linalg.norm(pca_dU_test)
        dU_dot_evec_test    = pca_dU_test.dot(pca_evecs_test)
        cos_dU_wopt_test    = abs(unit_vectors(pca_dU_test.T).T.dot(unit_vectors(pca_wopt_test)))
        dU_dot_evec_sq_test = pca_dU_test.dot(pca_evecs_test) ** 2
        evec_snr_test       = dU_dot_evec_sq_test / pca_evals_test
        cos_dU_evec_test    = abs(unit_vectors(pca_dU_test.T).T.dot(pca_evecs_test))

        # pack results into dictionary to return
        results = {
            'dp_opt_test': pca_dp_test, 
            'dp_diag_test': pca_dp_test_diag,
            'wopt_test': pca_wopt_test,
            'wdiag_test': pca_wopt_test_diag,
            'var_explained_test': pca_test_var,
            'evals_test': pca_evals_test, 
            'evecs_test': pca_evecs_test, 
            'dU_test': pca_dU_test,
            'dp_opt_train': pca_dp_train, 
            'dp_diag_train': pca_dp_train_diag, 
            'wopt_train': pca_wopt_train, 
            'wdiag_train': pca_wopt_train_diag, 
            'var_explained_train': pca_train_var,
            'evals_train': pca_evals_train, 
            'evecs_train': pca_evecs_train, 
            'dU_train': pca_dU_train, 
            'dU_mag_test': dU_mag_test, 
            'dU_dot_evec_test': dU_dot_evec_test, 
            'cos_dU_wopt_test': cos_dU_wopt_test, 
            'dU_dot_evec_sq_test': dU_dot_evec_sq_test, 
            'evec_snr_test': evec_snr_test, 
            'cos_dU_evec_test': cos_dU_evec_test,
            'dU_mag_train': dU_mag_train, 
            'dU_dot_evec_train': dU_dot_evec_train, 
            'cos_dU_wopt_train': cos_dU_wopt_train, 
            'dU_dot_evec_sq_train': dU_dot_evec_sq_train, 
            'evec_snr_train': evec_snr_train, 
            'cos_dU_evec_train': cos_dU_evec_train
        }

        # deal with large / small pupil data
        if ptrain_mask is not None:
            # perform analysis for big / small pupil data too. Only on test set.
            A_bp = xtest_pca[:, ptest_mask[0, :, 0], 0]
            A_sp = xtest_pca[:, ~ptest_mask[0, :, 0], 0]
            B_bp = xtest_pca[:, ptest_mask[0, :, 1], 1]
            B_sp = xtest_pca[:, ~ptest_mask[0, :, 1], 1]

            # get dprime / dU
            bp_dprime, _, _, _, bp_dU = compute_dprime(A_bp, B_bp, wopt=pca_wopt_train)
            sp_dprime, _, _, _, sp_dU = compute_dprime(A_sp, B_sp, wopt=pca_wopt_train)

            # get pupil-dependent variance along the prinicple noise axes (analogous to lambda)
            # point is to compare the variance along these PCs between large / small pupil
            big = np.concatenate([A_bp, B_bp], axis=-1)
            small = np.concatenate([A_sp, B_sp], axis=-1)
            bp_lambda = np.var(big.T.dot(pca_evecs_test), axis=0)
            sp_lambda = np.var(small.T.dot(pca_evecs_test), axis=0)

            # compute additional metrics
            bp_dU_mag         = np.linalg.norm(bp_dU)
            bp_dU_dot_evec    = bp_dU.dot(pca_evecs_test)
            bp_cos_dU_wopt    = abs(unit_vectors(bp_dU.T).T.dot(unit_vectors(pca_wopt_train)))
            bp_dU_dot_evec_sq = bp_dU.dot(pca_evecs_test) ** 2
            bp_evec_snr       = bp_dU_dot_evec_sq / bp_lambda
            bp_cos_dU_evec    = abs(unit_vectors(bp_dU.T).T.dot(pca_evecs_test))

            sp_dU_mag         = np.linalg.norm(sp_dU)
            sp_dU_dot_evec    = sp_dU.dot(pca_evecs_test)
            sp_cos_dU_wopt    = abs(unit_vectors(sp_dU.T).T.dot(unit_vectors(pca_wopt_train)))
            sp_dU_dot_evec_sq = sp_dU.dot(pca_evecs_test) ** 2
            sp_evec_snr       = sp_dU_dot_evec_sq / sp_lambda
            sp_cos_dU_evec    = abs(unit_vectors(sp_dU.T).T.dot(pca_evecs_test))

            results.update({
                'bp_dp': bp_dprime,
                'bp_evals': bp_lambda,
                'bp_dU_mag': bp_dU_mag,
                'bp_dU_dot_evec': bp_dU_dot_evec,
                'bp_cos_dU_wopt': bp_cos_dU_wopt,
                'bp_dU_dot_evec_sq': bp_dU_dot_evec_sq,
                'bp_evec_snr': bp_evec_snr,
                'bp_cos_dU_evec': bp_cos_dU_evec,
                'sp_dp': sp_dprime,
                'sp_evals': sp_lambda,
                'sp_dU_mag': sp_dU_mag,
                'sp_dU_dot_evec': sp_dU_dot_evec,
                'sp_cos_dU_wopt': sp_cos_dU_wopt,
                'sp_dU_dot_evec_sq': sp_dU_dot_evec_sq,
                'sp_evec_snr': sp_evec_snr,
                'sp_cos_dU_evec': sp_cos_dU_evec
            })

        return results


def do_pls_dprime_analysis(xtrain, xtest, nreps_train, nreps_test, ptrain_mask=None, ptest_mask=None, n_components=2):
        """
        perform pca on the trail averaged response, project into 2D space, compute dprime and associated metrics
        return results in a dictionary
        """
        try:
            Y = dr.get_one_hot_matrix(ncategories=2, nreps=nreps_train)
            pls = PLSRegression(n_components=n_components, max_iter=500, tol=1e-7)
            pls.fit(xtrain.T, Y.T)
            pls_weights = pls.x_weights_
            nan_pad = False
        except np.linalg.LinAlgError:
            # deflated matrix on this iteration of NIPALS was ~0. e.g. the overall matrix rank may have been 6, but
            # by the time it gets to this iteration, the last couple of indpendent dims are so small, that matrix
            # is essentially 0 and PLS can't converge.
            log.info("PLS can't converge. No more dimensions in the deflated matrix. Pad with nan and continue. \n"
                            "jack_idx: {0} \n"
                            "n_components: {1} \n "
                            "stim category: {2} \n "
                            "stim combo: {3}".format(ev_set, n_components, category, combo))
            nan_pad = True

        if not nan_pad:
            xtrain_pls = (xtrain.T @ pls_weights).T
            xtest_pls = (xtest.T @ pls_weights).T

            if np.linalg.matrix_rank(xtrain_pls) < n_components:
                # add one more check - this will cause singular matrix. dprime fn handles this,
                # but to prevent a barrage of log messages, check here to prevent even attempting dprime calc.
                # what this means is that the last dim(s) of x_weights are 0. i.e. there is no more explainable 
                # information about Y in X.
                nan_pad = True

            else:
                nan_pad = False
                xtrain_pls = nat_preproc.fold_X(xtrain_pls, nreps=nreps_train, nstim=2, nbins=1).squeeze()
                xtest_pls = nat_preproc.fold_X(xtest_pls, nreps=nreps_test, nstim=2, nbins=1).squeeze()
    
                pls_train_var = np.var(xtrain_pls.T @ pls_weights.T)  / np.var(xtrain)
                pls_test_var = np.var(xtest_pls.T @ pls_weights.T)  / np.var(xtest)

                # compute dprime metrics raw 
                pls_dp_train, pls_wopt_train, pls_evals_train, pls_evecs_train, pls_dU_train = \
                                        compute_dprime(xtrain_pls[:, :, 0], xtrain_pls[:, :, 1])
                pls_dp_test, pls_wopt_test, pls_evals_test, pls_evecs_test, pls_dU_test = \
                                        compute_dprime(xtest_pls[:, :, 0], xtest_pls[:, :, 1], wopt=pls_wopt_train)

                # override test decoder as train decoder
                pls_wopt_test = pls_wopt_train

                # compute dprime metrics diag decoder
                pls_dp_train_diag, pls_wopt_train_diag, _, _, _ = \
                                        compute_dprime(xtrain_pls[:, :, 0], xtrain_pls[:, :, 1], diag=True)
                pls_dp_test_diag, pls_wopt_test_diag, _, _, _ = \
                                        compute_dprime(xtest_pls[:, :, 0], xtest_pls[:, :, 1], diag=True)

                # caculate additional metrics
                dU_mag_train         = np.linalg.norm(pls_dU_train)
                dU_dot_evec_train    = pls_dU_train.dot(pls_evecs_train)
                cos_dU_wopt_train    = abs(unit_vectors(pls_dU_train.T).T.dot(unit_vectors(pls_wopt_train)))
                dU_dot_evec_sq_train = pls_dU_train.dot(pls_evecs_train) ** 2
                evec_snr_train       = dU_dot_evec_sq_train / pls_evals_train
                cos_dU_evec_train    = abs(unit_vectors(pls_dU_train.T).T.dot(pls_evecs_train))

                dU_mag_test         = np.linalg.norm(pls_dU_test)
                dU_dot_evec_test    = pls_dU_test.dot(pls_evecs_test)
                cos_dU_wopt_test    = abs(unit_vectors(pls_dU_test.T).T.dot(unit_vectors(pls_wopt_test)))
                dU_dot_evec_sq_test = pls_dU_test.dot(pls_evecs_test) ** 2
                evec_snr_test       = dU_dot_evec_sq_test / pls_evals_test
                cos_dU_evec_test    = abs(unit_vectors(pls_dU_test.T).T.dot(pls_evecs_test))

                # pack results into dictionary to return
                results = {
                    'dp_opt_test': pls_dp_test, 
                    'dp_diag_test': pls_dp_test_diag,
                    'wopt_test': pls_wopt_test,
                    'wdiag_test': pls_wopt_test_diag,
                    'var_explained_test': pls_test_var,
                    'evals_test': pls_evals_test, 
                    'evecs_test': pls_evecs_test, 
                    'dU_test': pls_dU_test,
                    'dp_opt_train': pls_dp_train, 
                    'dp_diag_train': pls_dp_train_diag, 
                    'wopt_train': pls_wopt_train, 
                    'wdiag_train': pls_wopt_train_diag, 
                    'var_explained_train': pls_train_var,
                    'evals_train': pls_evals_train, 
                    'evecs_train': pls_evecs_train, 
                    'dU_train': pls_dU_train, 
                    'dU_mag_test': dU_mag_test, 
                    'dU_dot_evec_test': dU_dot_evec_test, 
                    'cos_dU_wopt_test': cos_dU_wopt_test, 
                    'dU_dot_evec_sq_test': dU_dot_evec_sq_test, 
                    'evec_snr_test': evec_snr_test, 
                    'cos_dU_evec_test': cos_dU_evec_test,
                    'dU_mag_train': dU_mag_train, 
                    'dU_dot_evec_train': dU_dot_evec_train, 
                    'cos_dU_wopt_train': cos_dU_wopt_train, 
                    'dU_dot_evec_sq_train': dU_dot_evec_sq_train, 
                    'evec_snr_train': evec_snr_train, 
                    'cos_dU_evec_train': cos_dU_evec_train
                }


        if nan_pad:
            evec_nan = np.nan * np.ones((n_components, n_components))
            eval_nan = np.nan * np.ones(n_components)
            dU_nan = np.nan * np.ones((1, n_components))
            wopt_nan = np.nan * np.ones((n_components, 1)) 

            # pack results into dictionary to return
            results = {
                'dp_opt_test': np.nan, 
                'dp_diag_test': np.nan,
                'wopt_test': wopt_nan,
                'wdiag_test': wopt_nan,
                'var_explained_test': np.nan,
                'evals_test': eval_nan, 
                'evecs_test': evec_nan, 
                'dU_test': dU_nan,
                'dp_opt_train': np.nan, 
                'dp_diag_train': np.nan, 
                'wopt_train': wopt_nan, 
                'wdiag_train': wopt_nan, 
                'var_explained_train': np.nan,
                'evals_train': eval_nan, 
                'evecs_train': evec_nan, 
                'dU_train': dU_nan, 
                'dU_mag_test': np.nan, 
                'dU_dot_evec_test': dU_nan, 
                'cos_dU_wopt_test': np.nan, 
                'dU_dot_evec_sq_test': dU_nan, 
                'evec_snr_test': dU_nan, 
                'cos_dU_evec_test': dU_nan,
                'dU_mag_train': np.nan, 
                'dU_dot_evec_train': dU_nan, 
                'cos_dU_wopt_train': np.nan, 
                'dU_dot_evec_sq_train': dU_nan, 
                'evec_snr_train': dU_nan, 
                'cos_dU_evec_train': dU_nan
            }

        if (ptrain_mask is not None) & (nan_pad == False):
            # perform analysis for big / small pupil data too. Only on test set.
            A_bp = xtest_pls[:, ptest_mask[0, :, 0], 0]
            A_sp = xtest_pls[:, ~ptest_mask[0, :, 0], 0]
            B_bp = xtest_pls[:, ptest_mask[0, :, 1], 1]
            B_sp = xtest_pls[:, ~ptest_mask[0, :, 1], 1]

            # get dprime / dU
            bp_dprime, _, _, _, bp_dU = compute_dprime(A_bp, B_bp, wopt=pls_wopt_train)
            sp_dprime, _, _, _, sp_dU = compute_dprime(A_sp, B_sp, wopt=pls_wopt_train)

            # get pupil-dependent variance along the prinicple noise axes (analogous to lambda)
            # point is to compare the variance along these PCs between large / small pupil
            big = np.concatenate([A_bp, B_bp], axis=-1)
            small = np.concatenate([A_sp, B_sp], axis=-1)
            bp_lambda = np.var(big.T.dot(pls_evecs_test), axis=0)
            sp_lambda = np.var(small.T.dot(pls_evecs_test), axis=0)

            # compute additional metrics
            bp_dU_mag         = np.linalg.norm(bp_dU)
            bp_dU_dot_evec    = bp_dU.dot(pls_evecs_test)
            bp_cos_dU_wopt    = abs(unit_vectors(bp_dU.T).T.dot(unit_vectors(pls_wopt_train)))
            bp_dU_dot_evec_sq = bp_dU.dot(pls_evecs_test) ** 2
            bp_evec_snr       = bp_dU_dot_evec_sq / bp_lambda
            bp_cos_dU_evec    = abs(unit_vectors(bp_dU.T).T.dot(pls_evecs_test))

            sp_dU_mag         = np.linalg.norm(sp_dU)
            sp_dU_dot_evec    = sp_dU.dot(pls_evecs_test)
            sp_cos_dU_wopt    = abs(unit_vectors(sp_dU.T).T.dot(unit_vectors(pls_wopt_train)))
            sp_dU_dot_evec_sq = sp_dU.dot(pls_evecs_test) ** 2
            sp_evec_snr       = sp_dU_dot_evec_sq / sp_lambda
            sp_cos_dU_evec    = abs(unit_vectors(sp_dU.T).T.dot(pls_evecs_test))

            results.update({
                'bp_dp': bp_dprime,
                'bp_evals': bp_lambda,
                'bp_dU_mag': bp_dU_mag,
                'bp_dU_dot_evec': bp_dU_dot_evec,
                'bp_cos_dU_wopt': bp_cos_dU_wopt,
                'bp_dU_dot_evec_sq': bp_dU_dot_evec_sq,
                'bp_evec_snr': bp_evec_snr,
                'bp_cos_dU_evec': bp_cos_dU_evec,
                'sp_dp': sp_dprime,
                'sp_evals': sp_lambda,
                'sp_dU_mag': sp_dU_mag,
                'sp_dU_dot_evec': sp_dU_dot_evec,
                'sp_cos_dU_wopt': sp_cos_dU_wopt,
                'sp_dU_dot_evec_sq': sp_dU_dot_evec_sq,
                'sp_evec_snr': sp_evec_snr,
                'sp_cos_dU_evec': sp_cos_dU_evec
            })
        
        elif (ptrain_mask is not None) & (nan_pad == True):
            results.update({
                'bp_dp': np.nan,
                'bp_evals': eval_nan,
                'bp_dU_mag': np.nan,
                'bp_dU_dot_evec': dU_nan,
                'bp_cos_dU_wopt': np.nan,
                'bp_dU_dot_evec_sq': dU_nan,
                'bp_evec_snr': dU_nan,
                'bp_cos_dU_evec': dU_nan,
                'sp_dp': np.nan,
                'sp_evals': eval_nan,
                'sp_dU_mag': np.nan,
                'sp_dU_dot_evec': dU_nan,
                'sp_cos_dU_wopt': np.nan,
                'sp_dU_dot_evec_sq': dU_nan,
                'sp_evec_snr': dU_nan,
                'sp_cos_dU_evec': dU_nan
            })
        else:
            pass

        return results


def cast_dtypes(df):
    dtypes = {'dp_opt_test': 'float64',
              'dp_diag_test': 'float64',
              'wopt_test': 'object',
              'wdiag_test': 'object',
              'var_explained_test': 'float64',
              'evals_test': 'object',
              'evecs_test': 'object',
              'dU_test': 'object',
              'dp_opt_train': 'float64',
              'dp_diag_train': 'float64',
              'wopt_train': 'object',
              'wdiag_train': 'object',
              'var_explained_train': 'float64',
              'evals_train': 'object',
              'evecs_train': 'object',
              'dU_train': 'object',
              'dU_mag_test': 'float64',
              'dU_dot_evec_test': 'object',
              'cos_dU_wopt_test': 'float64',
              'dU_dot_evec_sq_test': 'object',
              'evec_snr_test': 'object', 
              'cos_dU_evec_test': 'object',
              'dU_mag_train': 'float64',
              'dU_dot_evec_train': 'object',
              'cos_dU_wopt_train': 'float64',
              'dU_dot_evec_sq_train': 'object',
              'evec_snr_train': 'object', 
              'cos_dU_evec_train': 'object',
              'bp_dp': 'float64',
              'bp_evals': 'object',
              'bp_dU_mag': 'float64',
              'bp_dU_dot_evec': 'object',
              'bp_cos_dU_wopt': 'float64',
              'bp_dU_dot_evec_sq': 'object',
              'bp_evec_snr': 'object',
              'bp_cos_dU_evec': 'object',
              'sp_dp': 'float64',
              'sp_evals': 'object',
              'sp_dU_mag': 'float64',
              'sp_dU_dot_evec': 'object',
              'sp_cos_dU_wopt': 'float64',
              'sp_dU_dot_evec_sq': 'object',
              'sp_evec_snr': 'object',
              'sp_cos_dU_evec': 'object',
              'category': 'category',
              'jack_idx': 'category',
              'n_components': 'category',
              'combo': 'category',
              'site': 'category'}
    dtypes_new = {k: v for k, v in dtypes.items() if k in df.columns}
    df = df.astype(dtypes_new)
    return df


# ================================== Analysis functions to compute dprime =================================================

def compute_dprime(A, B, diag=False, wopt=None):
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
    if (A.shape[0] > A.shape[1]) & (wopt is None):
        raise ValueError("Number of dimensions greater than number of observations. Unstable")

    if A.shape[0] != B.shape[0]:
        raise ValueError("Number of dimensions do not match between conditions")

    if diag:
        dprime, wopt, evals, evecs, dU = _dprime_diag(A, B)

    else:
        dprime, wopt, evals, evecs, dU = _dprime(A, B, wopt=wopt)

    return dprime, wopt, evals, evecs, dU 


def _dprime(A, B, wopt=None):
    """
    See Rumyantsev et. al 2020, Nature for nice derivation
    """

    usig = 0.5 * (np.cov((A - A.mean(axis=-1, keepdims=True))) + np.cov((B - B.mean(axis=-1, keepdims=True))))
    u_vec = (A.mean(axis=-1) - B.mean(axis=-1))[np.newaxis, :]    

    if wopt is not None:
        wopt_train = wopt / np.linalg.norm(wopt)
        A = A.T.dot(wopt_train).T
        B = B.T.dot(wopt_train).T

        usig_ = 0.5 * (np.cov((A - A.mean(axis=-1, keepdims=True))) + np.cov((B - B.mean(axis=-1, keepdims=True))))
        u_vec_ = (A.mean(axis=-1) - B.mean(axis=-1))[np.newaxis, :]
    
    try:
        if wopt is not None:
            wopt = (1 / usig_) * u_vec_
            dp2 = np.matmul(u_vec_, wopt)[0][0]
            try:
                # if wopt is passed, could still compute dpirme but can't compute 
                # evecs/ evals
                evals, evecs = np.linalg.eig(usig)
                # make sure evals / evecs are sorted
                idx_sort = np.argsort(evals)[::-1]
                evals = evals[idx_sort]
                evecs = evecs[:, idx_sort]
            except:
                wopt = np.nan * np.ones((A.shape[0], 1))
                evals = np.nan * np.ones((A.shape[0], ))
                evecs = np.nan * np.ones((A.shape[0], A.shape[0]))

        else:
            inv = np.linalg.inv(usig)
            wopt = inv @ u_vec.T
            dp2 = np.matmul(u_vec, wopt)[0][0]

            evals, evecs = np.linalg.eig(usig)
            # make sure evals / evecs are sorted
            idx_sort = np.argsort(evals)[::-1]
            evals = evals[idx_sort]
            evecs = evecs[:, idx_sort]

    except:
        log.info('WARNING, Singular Covariance, dprime infinite, set to np.nan')
        wopt_nan = np.nan * np.ones((A.shape[0], 1))
        evals_nan = np.nan * np.ones((A.shape[0], ))
        evecs_nan = np.nan * np.ones((A.shape[0], A.shape[0]))
        u_vec_nan =  np.nan * np.ones((1, A.shape[0]))
        return np.nan, wopt_nan, evals_nan, evecs_nan, u_vec_nan

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

    dp2 = (numerator / denominator).squeeze()

    evals, evecs = np.linalg.eig(usig)
    # make sure evals / evecs are sorted
    idx_sort = np.argsort(evals)[::-1]
    evals = evals[idx_sort]
    evecs = evecs[:, idx_sort]

    # best decoding axis ignoring correlations (reduces to direction of u_vec)
    wopt_diag = np.linalg.inv(usig_diag) @ u_vec.T

    return dp2, wopt_diag, evals, evecs, u_vec


# ================================= Data Loading Utils ========================================
def load_site(site, batch):
    """
    Loads recording and does some standard preprocessing for nat sounds decoding analysis
        e.g. masks validation set and removes post stim silence.
    
    Returns full spike count matrix, X, and a spike count matrix with just spont data
    """
    options = {'cellid': site, 'rasterfs': 4, 'batch': batch, 'pupil': True, 'stim': False}
    if batch == 294:
        options['runclass'] = 'VOC'
    rec = nb.baphy_load_recording_file(**options)
    rec['resp'] = rec['resp'].rasterize()
    if 'cells_to_extract' in rec.meta.keys():
        if rec.meta['cells_to_extract'] is not None:
            log.info("Extracting cellids: {0}".format(rec.meta['cells_to_extract']))
            rec['resp'] = rec['resp'].extract_channels(rec.meta['cells_to_extract'])

    # remove post stim silence (keep prestim so that can get a baseline dprime on each sound)
    rec = rec.and_mask(['PostStimSilence'], invert=True)
    if batch == 294:
        epochs = [epoch for epoch in rec.epochs.name.unique() if 'STIM_' in epoch]
    else:
        epochs = [epoch for epoch in rec.epochs.name.unique() if 'STIM_00' in epoch]
    rec = rec.and_mask(epochs)
    resp_dict = rec['resp'].extract_epochs(epochs, mask=rec['mask'], allow_incomplete=True)
    spont_signal = rec['resp'].epoch_to_signal('PreStimSilence')
    sp_dict = spont_signal.extract_epochs(epochs, mask=rec['mask'], allow_incomplete=True)
    pup_dict = rec['pupil'].extract_epochs(epochs, mask=rec['mask'], allow_incomplete=True)

    # create response matrix, X
    X = nat_preproc.dict_to_X(resp_dict)
    X_sp = nat_preproc.dict_to_X(sp_dict)
    X_pup = nat_preproc.dict_to_X(pup_dict)

    return X, X_sp, X_pup

# ================================= Plotting Utilities =========================================
def plot_pair(est, val, nreps_train, nreps_test, train_pmask=None, test_pmask=None):
    
    # unfold data
    if train_pmask is None:
        train = nat_preproc.fold_X(est, nreps=nreps_train, nstim=2, nbins=1).squeeze()
        test = nat_preproc.fold_X(val, nreps=nreps_test, nstim=2, nbins=1).squeeze()
    else:
        train = nat_preproc.fold_X(est, nreps=nreps_train, nstim=2, nbins=1).squeeze()
        test = nat_preproc.fold_X(val, nreps=nreps_test, nstim=2, nbins=1).squeeze()
        Atrain = train[:, :, 0]
        Atest = test[:, :, 0]
        Btrain = train[:, :, 1]
        Btest = test[:, :, 1]

        Atrain_bp = Atrain[:, train_pmask[0, :, 0]]
        Atrain_sp = Atrain[:, ~train_pmask[0, :, 0]]
        Atest_bp = Atest[:, test_pmask[0, :, 0]]
        Atest_sp = Atest[:, ~test_pmask[0, :, 0]]

        Btrain_bp = Btrain[:, train_pmask[0, :, 1]]
        Btrain_sp = Btrain[:, ~train_pmask[0, :, 1]]
        Btest_bp = Btest[:, test_pmask[0, :, 1]]
        Btest_sp = Btest[:, ~test_pmask[0, :, 1]]

    # get stats
    dp_train, wopt_train, evals_train, evecs_train, dU_train = \
                    compute_dprime(train[:, :, 0], train[:, :, 1])
    dp_test, wopt_test, evals_test, evecs_test, dU_test = \
                    compute_dprime(test[:, :, 0], test[:, :, 1], wopt=wopt_train)

    wopt_unit = wopt_train / np.linalg.norm(wopt_train) * np.std(train)

    f, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    
    # plot train data
    ax[0].set_title(r"Train, $d'={}$".format(round(dp_train**(1/2), 3)))
    if train_pmask is None:
        ax[0].scatter(train[0, :, 0], train[1, :, 0], edgecolor='white', color='r', label='A')
        ax[0].scatter(train[0, :, 1], train[1, :, 1], edgecolor='white', color='b', label='B')
    else:
        ax[0].scatter(Atrain_bp[0, :], Atrain_bp[1, :], color='r', label='A, big pup')
        ax[0].scatter(Btrain_bp[0, :], Btrain_bp[1, :], color='b', label='B, big pup')
        ax[0].scatter(Atrain_sp[0, :], Atrain_sp[1, :], facecolor='none', color='r', label='A, sm. pup')
        ax[0].scatter(Btrain_sp[0, :], Btrain_sp[1, :], facecolor='none', color='b', label='B, sm. pup')
    
    # plot vectors
    ax[0].plot([0, wopt_unit[0, 0]], [0, wopt_unit[1, 0]], 'k-', lw=2, label=r'$w_{opt}$')

    ax[0].set_xlabel('PLS 1')
    ax[0].set_ylabel('PLS 2')
    ax[0].axis('square')

    ax[0].legend(frameon=False)

    # plot test data
    ax[1].set_title(r"Test, $d'={}$".format(round(dp_test**(1/2), 3)))
    if test_pmask is None:
        ax[1].scatter(test[0, :, 0], test[1, :, 0], edgecolor='white', color='r', label='A')
        ax[1].scatter(test[0, :, 1], test[1, :, 1], edgecolor='white', color='b', label='B')
    else:
        ax[1].scatter(Atest_bp[0, :], Atest_bp[1, :], color='r', label='A, big pup')
        ax[1].scatter(Btest_bp[0, :], Btest_bp[1, :], color='b', label='B, big pup')
        ax[1].scatter(Atest_sp[0, :], Atest_sp[1, :], facecolor='none', color='r', label='A, sm. pup')
        ax[1].scatter(Btest_sp[0, :], Btest_sp[1, :], facecolor='none', color='b', label='B, sm. pup')
        
    ax[1].set_xlabel('PLS 1')
    ax[1].set_ylabel('PLS 2')
    ax[1].axis('square')

    return f