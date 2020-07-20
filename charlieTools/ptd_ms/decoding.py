import charlieTools.ptd_ms.dim_reduction as dr

import pickle
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
                m = self.numeric_results.index.get_level_values('n_components') == n_components
                return [self.numeric_results.loc[m, name],
                        self.numeric_results.loc[m, name+'_sem']]
            elif n_components is None:
                if type(stim_pair) is str:
                    m = self.numeric_results.index.get_level_values('combo') == stim_pair
                else:
                    m = self.numeric_results.index.get_level_values('combo').isin(stim_pair)
                return [self.numeric_results.loc[m, name],
                        self.numeric_results.loc[m, name+'_sem']]
            else:
                m1 = self.numeric_results.index.get_level_values('n_components') == n_components
                if type(stim_pair) is str:
                    m2 = self.numeric_results.index.get_level_values('combo') == stim_pair
                else:
                    m2 = self.numeric_results.index.get_level_values('combo').isin(stim_pair)
                m = m1 & m2
                return [self.numeric_results.loc[m, name],
                        self.numeric_results.loc[m, name+'_sem']]

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

    def save_json(self, fn):
        log.info("json serializing DecodingResults object to {}".format(fn))
        js_string = jsonpickle.encode(self)
        with open(fn, 'w') as handle:
            json.dump(js_string, handle)
    
    def load_json(self, fn):
        if not os.path.isfile(fn):
            raise FileNotFoundError
        log.info("loading json string from {}".format(fn))
        with open(fn, 'r') as handle:
            js_string = json.load(handle)
        return jsonpickle.decode(js_string)
        

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


# HELPER FUNCTIONS
def get_est_val_sets(d, njacks=10, masks=None, min_reps=10):
    """
    Pretty specialized function. The idea is, for a given dictionary of spike counts 
    over all state conditions, you want to make sure that each est set contains balanced (within 
    the specified tolerance) reps of each state that you're interested in studying. So, you 
    need to pass the state masks (masks argument) is order to make sure this happens. 

    tolerance is a percentage. e.g. if tolerance = 5, can have 5% more data in the True
    than the False half. 

    If masks is not None, there is an additional balancing step here to make sure that within
    dnew, each state is equally represented. This is done with resampling (after the above
    tolerance criteria are met)

    If masks is None, this function is much more simple. Just does 50-50 split on the spike
    dict for njacks unique sets. tolerance isn't used in this case.
    """

    # make sure each mask constains the same keys as the raw data.
    if masks is not None:
        if type(masks) is not list:
            masks = [masks]
        for m in masks:
            if (m is not None) and (m.keys() != d.keys()):
                raise ValueError

    # set up new dicts to hold masked data
    mdicts = []
    if masks is not None:
        for mask in masks:
            mdict = dict.fromkeys(d.keys())
            for k in mdict.keys():
                mdict[k] = {}
                mdict[k]['est'] = []
                mdict[k]['val'] = []
            mdicts.append(mdict)

    dnew = dict.fromkeys(d.keys())
    for k in d.keys():
        data = d[k]
        
        nreps = data.shape[0]
        n_test_reps = int(nreps / 2)
        all_idx = np.arange(0, nreps)

        test_idx = []
        dnew[k] = {}
        dnew[k]['est'] = []
        dnew[k]['val'] = []
        count = 0
        max_iterations = 1000
        max_iter = 0
        if n_test_reps >= min_reps:
            while (count <= njacks) & (max_iter <= max_iterations):
                ti = list(np.random.choice(all_idx, n_test_reps, replace=False))
                # if masks is not None, determine if ti leads to sufficient data in est / val for each mask
                if masks is not None:
                    for mask in masks:
                        md = mask[k][ti, 0, 0]  # masks are def. per epoch, so can just take first el
                        mdv = md.sum()
                        mde = (~md).sum()
                        if (mdv > (min_reps/2)) & (mde > (min_reps/2)):
                            # this meets tolerance condition, continue going through masks
                            ti_is_balanced = True
                            pass
                        else:
                            ti_is_balanced = False 
                            break

                if (ti not in test_idx) & ti_is_balanced:
                    try_balance = 0
                    tri = list(set(all_idx) - set(ti))
                    test_idx.append(ti)

                    # save masked data too
                    if masks is not None:
                        min_occurences = 1e5
                        for i, mask in enumerate(masks):
                            mdicts[i][k]['est'].append(data[ti, :, :][mask[k][ti, 0, 0]])
                            mdicts[i][k]['val'].append(data[tri, :, :][mask[k][tri, 0, 0]])
                            if mdicts[i][k]['est'][-1].shape[0] < min_occurences:
                                min_occurences = mdicts[i][k]['est'][-1].shape[0]

                        # need to do some extra balancing of the "all data" category based on
                        # the min represented state. We do this so that estimates using "all data"
                        # are not biased towards one state.
                        # This is done randomly, so extra jackknifes help
                        # make sure all the data is sampled
                        keep_idxs = []
                        for mask in masks:
                            # keep min_occurences of true mask indexes in ti
                            idx = [i for i in np.argwhere(mask[k][:,0,0]).flatten() if i in ti]
                            keep_idxs.extend(np.random.choice(idx, min_occurences, replace=False))
                        keep_idx = np.unique(keep_idxs)

                        # for the "all data" only keep this new balanced set. That way it can be used to 
                        # calculated decoding axes in an unbiased way
                        dnew[k]['est'].append(data[keep_idx, :, :])
                        # val set doesn't need to be balanced
                        dnew[k]['val'].append(data[tri, :, :])

                    else:
                        dnew[k]['est'].append(data[ti, :, :])
                        dnew[k]['val'].append(data[tri, :, :])


                    count += 1
                
                else:
                    max_iter +=1
    
            if max_iter >= max_iterations:
                log.info("couldn't balance reps across masks for epoch: {}".format(k))

        elif n_test_reps < min_reps:
            log.info("couldn't preform cross-validation for epoch: {}. Didn't meet min rep requirements".format(k))


    return (dnew,)+tuple(mdicts)


def squeeze_time_dim(d, fs, twin):
    """
    Squish time bins down to single bin according the time window specified by twin
    """

    idx = np.arange(int(fs * twin[0]), int(fs * twin[1]))
    if twin[0] == twin[1]:
        idx = [int(twin[0] * fs)]

    for k in d.keys():
        if type(d[k]) is dict:
            # do for est / val set
            for i in range(0, len(d[k]['est'])):
                d[k]['est'][i] = d[k]['est'][i][:, :, idx].mean(axis=-1, keepdims=True)
                d[k]['val'][i] = d[k]['val'][i][:, :, idx].mean(axis=-1, keepdims=True)
        else:   
            # no est / val data
            d[k] = d[k][:, :, idx].mean(axis=-1)

    return d


# wrapper function(s) to perform dim reduction, compute dprime, save associated statistics 
def do_tdr_dprime_analysis(train_data, test_data, decoding_data=None, tdr2_axis=None):

    if decoding_data is None:
        decoding_data = train_data

    # do dim reduction on decoding data
    Add = decoding_data[0]
    Bdd = decoding_data[1]
    tdr = dr.TDR(tdr2_init=tdr2_axis)   
    tdr.fit(Add, Bdd)
    tdr_weights = tdr.weights

    # calculate dprime on decoding data
    ddtrainA = (Add @ tdr_weights.T).T
    ddtrainB = (Bdd @ tdr_weights.T).T

    _dp_train, _wopt_train, _evals_train, _evecs_train, _evec_sim_train, _dU_train = \
                        compute_dprime(ddtrainA, ddtrainB)


    # apply to train / test data 
    # train data
    Add = train_data[0]
    Bdd = train_data[1]
    # calculate dprime on train data
    edtrainA = (Add @ tdr_weights.T).T
    edtrainB = (Bdd @ tdr_weights.T).T
    dp_train, wopt_train, evals_train, evecs_train, evec_sim_train, dU_train = \
                    compute_dprime(edtrainA, edtrainB, wopt=_wopt_train)

    # test data
    Add = test_data[0]
    Bdd = test_data[1]
    # calculate dprime on test data
    edtestA = (Add @ tdr_weights.T).T
    edtestB = (Bdd @ tdr_weights.T).T
    dp_test, wopt_test, evals_test, evecs_test, evec_sim_test, dU_test = \
                    compute_dprime(edtestA, edtestB, wopt=_wopt_train)

    # save decoding results
    results = {
        'dp_opt_test': dp_test, 
        'wopt_test': wopt_test,
        'evals_test': evals_test, 
        'evecs_test': evecs_test, 
        'evec_sim_test': evec_sim_test,
        'dU_test': dU_test,
        'dp_opt_train': dp_train, 
        'wopt_train': wopt_train,
        'evals_train': evals_train, 
        'evecs_train': evecs_train, 
        'evec_sim_train': evec_sim_train,
        'dU_train': dU_train,
    }

    # udpate decoding results with the metrics from the decoding data
    results.update({
        'dp_opt_dd': _dp_train, 
        'wopt_dd': _wopt_train,
        'evals_dd': _evals_train, 
        'evecs_dd': _evecs_train, 
        'evec_sim_dd': _evec_sim_train,
        'dU_dd': _dU_train,
    })

    # caculate addtional metrics and append to results

    return results


def cast_dtypes(df):
    dtypes = {'dp_opt_dd': 'float64', 
              'wopt_dd': 'object',
              'evals_dd': 'object', 
              'evecs_dd': 'object', 
              'evec_sim_dd': 'float64',
              'dU_dd': 'object',
              'dp_opt_test': 'float64',
              'dp_diag_test': 'float64',
              'wopt_test': 'object',
              'wdiag_test': 'object',
              'var_explained_test': 'float64',
              'evals_test': 'object',
              'evecs_test': 'object',
              'evec_sim_test': 'float64',
              'dU_test': 'object',
              'dp_opt_train': 'float64',
              'dp_diag_train': 'float64',
              'wopt_train': 'object',
              'wdiag_train': 'object',
              'var_explained_train': 'float64',
              'evals_train': 'object',
              'evecs_train': 'object',
              'evec_sim_train': 'float64',
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
              'beta1_dot_dU': 'float64',
              'beta2_dot_dU': 'float64',
              'beta1_dot_tdr2': 'float64',
              'beta2_dot_tdr2': 'float64',
              'beta1_dot_wopt': 'float64',
              'beta2_dot_wopt': 'float64',
              'beta1_lambda': 'float64',
              'beta2_lambda': 'float64',
              'beta1_mag': 'float64',
              'beta2_mag': 'float64',
              'dU_dot_beta1_sq': 'float64',
              'dU_dot_beta2_sq':'float64',
              'beta1_snr':'float64',
              'beta2_snr':  'float64',
              'cos_dU_beta1': 'float64',
              'cos_dU_beta2': 'float64',
              'dU_all': 'object',
              'wopt_all': 'object',
              'evecs_all': 'object',
              'dU_all_test': 'object',
              'evecs_all_test': 'object',
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
              'mean_pupil_range': 'float64',
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
        evec_sim: similarity of first eigenvector between the two stimuli, A and B
        dU: <A> - <B>
    """
    if (A.shape[0] > A.shape[1]) & (wopt is None):
        raise ValueError("Number of dimensions greater than number of observations. Unstable")

    if A.shape[0] != B.shape[0]:
        raise ValueError("Number of dimensions do not match between conditions")

    if diag:
        dprime, wopt, evals, evecs, evec_sim, dU = _dprime_diag(A, B)

    else:
        dprime, wopt, evals, evecs, evec_sim, dU = _dprime(A, B, wopt=wopt)

    return dprime, wopt, evals, evecs, evec_sim, dU 


def _dprime(A, B, wopt=None):
    """
    See Rumyantsev et. al 2020, Nature for nice derivation
    """

    sigA = np.cov((A - A.mean(axis=-1, keepdims=True)))
    sigB = np.cov((B - B.mean(axis=-1, keepdims=True)))

    usig = 0.5 * (sigA + sigB)
    u_vec = (A.mean(axis=-1) - B.mean(axis=-1))[np.newaxis, :]    

    try:
        valA, vecA = np.linalg.eig(sigA)
        valB, vecB = np.linalg.eig(sigB)
        evec_sim = abs(vecB[:, np.argsort(valB)[::-1][0]].dot(vecA[:, np.argsort(valA)[::-1][0]]))
    except:
        evec_sim = np.nan

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
        return np.nan, wopt_nan, evals_nan, evecs_nan, np.nan, u_vec_nan

    return dp2, wopt, evals, evecs, evec_sim, u_vec


def _dprime_diag(A, B):
    """
    See Rumyantsev et. al 2020, Nature  and Averbeck 2006, JNP for nice derivations.
        Note typo in Rumyantsev though!
    """

    sigA = np.cov((A - A.mean(axis=-1, keepdims=True)))
    sigB = np.cov((B - B.mean(axis=-1, keepdims=True)))

    usig = 0.5 * (sigA + sigB)
    u_vec = (A.mean(axis=-1) - B.mean(axis=-1))[np.newaxis, :]    

    try:
        valA, vecA = np.linalg.eig(sigA)
        valB, vecB = np.linalg.eig(sigB)
        evec_sim = abs(vecB[:, np.argsort(valB)[::-1][0]].dot(vecA[:, np.argsort(valA)[::-1][0]]))
    except:
        evec_sim = np.nan

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
        return np.nan, wopt_nan, evals_nan, evecs_nan, np.nan, u_vec_nan

    dp2 = (numerator / denominator).squeeze()

    evals, evecs = np.linalg.eig(usig)
    # make sure evals / evecs are sorted
    idx_sort = np.argsort(evals)[::-1]
    evals = evals[idx_sort]
    evecs = evecs[:, idx_sort]

    # best decoding axis ignoring correlations (reduces to direction of u_vec)
    wopt_diag = np.linalg.inv(usig_diag) @ u_vec.T

    return dp2, wopt_diag, evals, evecs, evec_sim, u_vec