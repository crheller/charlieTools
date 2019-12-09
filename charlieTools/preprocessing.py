import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import nems.xforms as xforms
import nems.db as nd
import logging
from nems.recording import Recording
from nems_lbhb.preprocessing import create_pupil_mask
import sys
import os
import scipy.fftpack as fp
import scipy.signal as ss
import copy
from nems_lbhb.preprocessing import create_pupil_mask

log = logging.getLogger(__name__)


def generate_state_corrected_psth(batch=None, modelname=None, cellids=None, cache_path=None):
    """
    Modifies the exisiting recording so that psth signal is the prediction specified
    by the modelname. Designed with stategain models in mind. CRH.

    If the model doesn't exist already in /auto/users/hellerc/results/, this
    will go ahead and fit the model and save it in /auto/users/hellerc/results.

    If the fit dir (from xforms) exists, simply reload the result and call this psth.
    """
    if cache_path is not None:
        fn = cache_path + cellids[0][:7] + '_{}.tgz'.format(modelname.split('.')[1])
        if os.path.isfile(fn):
            rec = Recording.load(fn)
            return rec
        else:
            # do the rest of the code
            pass

    if batch is None or modelname is None:
        raise ValueError('Must specify batch and modelname!')
    results_table = nd.get_results_file(batch, modelnames=[modelname])
    preds = []
    for cell in cellids:
        log.info(cell)
        try:
            p = results_table[results_table['cellid']==cell]['modelpath'].values[0]
            if os.path.isdir(p):
                xfspec, ctx = xforms.load_analysis(p)
                preds.append(ctx['val'])
            else:
                sys.exit('Fit for {0} does not exist'.format(cell))
        except:
            log.info("WARNING: fit doesn't exist for cell {0}".format(cell))

    # Need to add a check to make sure that the preds are the same length (if
    # multiple cellids). This could be violated if one cell for example existed
    # in a prepassive run but the other didn't and so they were fit differently
    file_epochs = []

    for pr in preds:
        file_epochs += [ep for ep in pr.epochs.name if 'FILE' in ep]

    unique_files = np.unique(file_epochs)
    shared_files = []
    for f in unique_files:
        if np.sum([1 for file in file_epochs if file == f]) == len(preds):
            shared_files.append(str(f))
        else:
            # this rawid didn't span all cells at the requested site
            pass

    # mask all file epochs for all preds with the shared file epochs
    # and adjust epochs
    if (int(batch) == 307) | (int(batch) == 294):
        for i, p in enumerate(preds):
            preds[i] = p.and_mask(shared_files)
            preds[i] = preds[i].apply_mask(reset_epochs=True)

    sigs = {}
    for i, p in enumerate(preds):
        if i == 0:
            new_psth = p['pred']
            new_resp = p['resp'].rasterize()
        else:
            new_psth = new_psth.concatenate_channels([new_psth, p['pred']])
            new_resp = new_resp.concatenate_channels([new_resp, p['resp'].rasterize()])

    new_pup = preds[0]['pupil']
    sigs['pupil'] = new_pup

    if 'pupil_raw' in preds[0].signals.keys():
        sigs['pupil_raw'] = preds[0]['pupil_raw']

    if 'mask' in preds[0].signals:
        new_mask = preds[0]['mask']
        sigs['mask'] = new_mask
    else:
        mask_rec = preds[0].create_mask(True)
        new_mask = mask_rec['mask']
        sigs['mask'] = new_mask

    if 'rem' in preds[0].signals.keys():
        rem = preds[0]['rem']
        sigs['rem'] = rem

    if 'pupil_eyespeed' in preds[0].signals.keys():
        new_eyespeed = preds[0]['pupil_eyespeed']
        sigs['pupil_eyespeed'] = new_eyespeed

    new_psth.name = 'psth'
    new_resp.name = 'resp'
    sigs['psth'] = new_psth
    sigs['resp'] = new_resp

    new_rec = Recording(sigs, meta=preds[0].meta)

    if cache_path is not None:
        log.info('caching {}'.format(fn))
        new_rec.save_targz(fn)

    return new_rec


def regress_pupil(rec):
    """
    ** pupil specific function - used for batch 289**
    ** `regress_state` is an equivalent function, but allows you to fit multiple to regression
    to > 1 state variabel and choose which to regress out **

    Remove first-order effect on the trial-trial variability for each neuron

    1) Compute compute psth in each bin by averaging over the whole recording
    2) At each time bin, model residuals as linear function of pupil
    3) Subtract off prediction
    4) Add the corrected residuals back to the overall psth
    """
    r = copy.deepcopy(rec)
    ep = np.unique([ep for ep in r.epochs.name if 'STIM_00' in ep]).tolist()

    r_st = r['resp'].extract_epochs(ep)
    p_st = r['pupil'].extract_epochs(ep)
    r_psth = r_st.copy()
    r_new = r_st.copy()
    for e in ep:
        m = r_st[e].mean(axis=0)
        r_psth[e] = np.tile(m, [r_st[e].shape[0], 1, 1])
        # for each stim bin
        for b in range(r_st[e].shape[-1]):
            # for each neuron, regress out pupil effects
            for n in range(r_psth[e].shape[1]):
                X = p_st[e][:, :, b].squeeze()
                y = r_st[e][:, n, b] - r_psth[e][:, n, b]
                reg = LinearRegression()
                reg.fit(X[:, np.newaxis], y[:, np.newaxis])
                y_pred = reg.predict(X[:, np.newaxis])
                y_new_residual = y - y_pred.squeeze()
                r_new[e][:, n, b] = r_psth[e][:, n, b] + y_new_residual

    r['resp'] = r['resp'].replace_epochs(r_new)

    return r

def regress_state(rec, state_sigs=['behavior', 'pupil'], regress=['pupil']):
    """
    Remove first order effects of given state variable(s). Idea is to model all state
    variables, for example, pupil and behavior, then just choose to remove the first order
    effects of one or the other, or both.

    1) Compute compute psth in each bin by averaging over the whole recording
    2) At each time bin, model residuals as linear function of state variable(s)
    3) Subtract off prediction computed using the coef(s) for the "regress" states
    4) Add the corrected residuals back to the overall psth
    """
    r = copy.deepcopy(rec)
    ep = np.unique([ep for ep in r.epochs.name if ('STIM' in ep) | ('TARGET' in ep)]).tolist()

    r_st = r['resp'].extract_epochs(ep)
    state_signals = dict.fromkeys(state_sigs)
    for s in state_sigs:
        if s == 'pupil':
            state_signals[s] = r['pupil'].extract_epochs(ep)
        elif s == 'behavior':
            r_beh_mask = r.create_mask(True)
            r_beh_mask = r_beh_mask.and_mask(['ACTIVE_EXPERIMENT'])
            state_signals[s] = r_beh_mask['mask'].extract_epochs(ep)
        elif s == 'lv':
            state_signals[s] = r['lv'].extract_epochs(ep)
        else:
            raise ValueError("No case set up for {}".format(s))

    r_psth = r_st.copy()
    r_new = r_st.copy()
    for e in ep:
        m = r_st[e].mean(axis=0)
        r_psth[e] = np.tile(m, [r_st[e].shape[0], 1, 1])
        # for each stim bin
        for b in range(r_st[e].shape[-1]):
            # for each neuron, regress out state effects
            for n in range(r_psth[e].shape[1]):
                for i, s in enumerate(state_sigs):
                    if i == 0:
                        X = state_signals[s][e][:, :, b]
                    else:
                        X = np.concatenate((X, state_signals[s][e][:, :, b]), axis=-1)

                y = r_st[e][:, n, b] - r_psth[e][:, n, b]
                reg = LinearRegression()

                X = X - X.mean(axis=0)
                X = X / X.std(axis=0)

                reg.fit(X, y[:, np.newaxis])

                # figure out regression coefficients
                args = [True if r in regress else False for r in state_sigs]
                model_coefs = reg.coef_[:, args]
                y_pred = np.matmul(X[:, args], model_coefs.T) + reg.intercept_
                y_new_residual = y - y_pred.squeeze()
                r_new[e][:, n, b] = r_psth[e][:, n, b] + y_new_residual

    r['resp'] = r['resp'].replace_epochs(r_new)

    return r


def regress_state2(rec, state_sigs=['behavior', 'pupil'], regress=['pupil']):
    """
    Same as the above function (and will likely replace it), except that now, instead of fitting
    for each stimulus independently, fit the residuals of all the data. I think this is important
    in order to fully sample state.
    """

    r = rec.copy()
    r = generate_psth(r)

    # get state signals
    state_signals = dict.fromkeys(state_sigs)
    for s in state_sigs:
        if s == 'pupil':
            state_signals[s] = r['pupil']._data
        elif s == 'behavior':
            r_beh_mask = r.create_mask(True)
            r_beh_mask = r_beh_mask.and_mask(['ACTIVE_EXPERIMENT'])
            state_signals[s] = r_beh_mask['mask']._data
        elif s == 'lv':
            state_signals[s] = r['lv']._data
        else:
            raise ValueError("No case set up for {}".format(s))

    # build state matrix
    for i, s in enumerate(state_sigs):
        if i == 0:
            X = state_signals[s]
        else:
            X = np.concatenate((X, state_signals[s]), axis=0)


    r_new = r['resp']._data.copy()
    for n in range(r_new.shape[0]):
        # fit regression model for each neuron
        y = r['resp']._data[n, :] - r['psth']._data[n, :]
        reg = LinearRegression()

        X = X - X.mean(axis=-1)
        X = X / X.std(axis=-1)

        reg.fit(X.T, y[:, np.newaxis])

        # compute predictions (of residuals)
        pred = reg.predict(X.T)

        # compute new residual (regressing out state)
        new_residual = y - pred.T

        # add new residual (w/o state modulation) back to psth
        r_new[n, :] = r['psth']._data[n, :] + new_residual.squeeze()

    r['resp'] = r['resp']._modified_copy(r_new)

    return r


def bandpass_filter_resp(rec, low_c, high_c):
    '''
    Bandpass filter resp. Return new recording with filtered resp.
    '''

    if low_c is None:
        low_c = 0
    if high_c is None:
        high_c = rec['resp'].fs

    newrec = rec.copy()
    newrec = newrec.apply_mask(reset_epochs=True)
    fs = rec['resp'].fs
    resp = rec['resp'].rasterize()._data
    resp_filt = resp.copy()
    for n in range(resp.shape[0]):
        s = resp[n, :]
        resp_fft = fp.fft(s)
        w = fp.fftfreq(s.size, 1 / fs)
        inds = np.argwhere((w >= low_c) & (w <= high_c))
        inds2 = np.argwhere((w <= -low_c) & (w >= -high_c))
        m = np.zeros(w.shape)
        alpha = 0.1
        m[inds] = ss.tukey(len(inds), alpha)[:, np.newaxis]
        m[inds2] = ss.tukey(len(inds2), alpha)[:, np.newaxis]
        resp_cut = resp_fft * m
        resp_filt[n, :] = fp.ifft(resp_cut)

    newrec['resp'] = newrec['resp']._modified_copy(resp_filt)

    return newrec

def highpass_filter_response(rec, cutoff):
    '''
    will apply mask and filter all data in resp. cutoff in Hz.
    '''
    newrec = rec.copy()
    newrec = newrec.apply_mask(reset_epochs=True)
    resp = newrec['resp'].extract_epoch('REFERENCE')
    fs = rec['resp'].fs
    resp_filt = resp.copy()
    # filter each neuron/trial independently
    for rep in range(resp.shape[0]):
        for n in range(resp.shape[1]):
            s = resp[rep, n, :]
            resp_fft = fp.fft(s)
            w = fp.fftfreq(s.size, 1 / fs)
            inds = np.argwhere((w >= cutoff))
            inds2 = np.argwhere((w <= -cutoff))
            m = np.zeros(w.shape)
            alpha = 0.4
            m[inds] = ss.tukey(len(inds), alpha)[:, np.newaxis]
            m[inds2] = ss.tukey(len(inds2), alpha)[:, np.newaxis]
            resp_cut = resp_fft * m
            resp_filt[rep, n, :] = fp.ifft(resp_cut)

    newrec['resp'] = newrec['resp'].replace_epochs({'REFERENCE': resp_filt})

    return newrec


def zscore_data(rec):
    """
    Z-score data and return new recording with each neuron having mean 0, std 1
    """
    newrec = rec.copy()
    resp = newrec['resp']._data

    u = np.mean(resp, axis=-1)
    resp -= u

    std = np.std(resp, axis=-1)
    resp /= std

    newrec['resp'] = newrec['resp']._modified_copy(resp)

    return newrec

def get_pupil_balanced_epochs(rec, rec_sp=None, rec_bp=None):
    """
    Given big/small pupil recordings return list of
    epochs that are balanced between the two.
    """
    all_epochs = np.unique([str(ep) for ep in rec.epochs.name if 'STIM_00' in ep]).tolist()

    if (rec_sp is None) | (rec_bp is None):
        pup_ops = {'state': 'big', 'epoch': ['REFERENCE'], 'collapse': True}
        rec_bp = create_pupil_mask(rec.copy(), **pup_ops)
        pup_ops['state']='small'
        rec_sp = create_pupil_mask(rec.copy(), **pup_ops)

        rec_bp = rec_bp.apply_mask(reset_epochs=True)
        rec_sp = rec_sp.apply_mask(reset_epochs=True)

    # get rid of pre/post stim silence
    #rec_bp = rec_bp.and_mask(['PostStimSilence'], invert=True)
    #rec_sp = rec_sp.and_mask(['PostStimSilence'], invert=True)
    #rec = rec.and_mask(['PostStimSilence'], invert=True)
    #rec_bp = rec_bp.apply_mask(reset_epochs=True)
    #rec_sp = rec_sp.apply_mask(reset_epochs=True)
    #rec = rec.apply_mask(reset_epochs=True)

    # find pupil matched epochs
    balanced_eps = []
    for ep in all_epochs:
        sp = rec_sp['resp'].extract_epoch(ep).shape[0]
        bp = rec_bp['resp'].extract_epoch(ep).shape[0]
        if len(all_epochs)==3:
            if abs(sp - bp) < 3:
                balanced_eps.append(ep)
        else:
            if sp==bp:
                balanced_eps.append(ep)

    if len(balanced_eps)==0:
        log.info("no balanced epochs at site {}".format(rec.name))

    else:
        log.info("found {0} balanced epochs:".format(len(balanced_eps)))

    return balanced_eps

def get_pupil_balanced_torcs(rec, mask1, mask2, balanced=True, present=False):
    """
    Used to get the torcs that are present/balanced between both masks
    """
    all_epochs = np.unique([str(ep) for ep in rec.epochs.name if 'TORC' in ep]).tolist()

    # find pupil matched epochs
    balanced_eps = []
    for ep in all_epochs:
        try:
            sp = rec['resp'].extract_epoch(ep, mask=mask1).shape[0]
            bp = rec['resp'].extract_epoch(ep, mask=mask2).shape[0]
            if balanced == True:
                if (abs(sp - bp) < 2) & (bp > 0) & (sp > 0):
                    balanced_eps.append(ep)
            elif present == True:
                # just make sure epoch exists in both
                if ((bp > 0) & (sp > 0)):
                    balanced_eps.append(ep)
        except:
            # epoch doesn't exist inside one of the masks.
            pass

    if len(balanced_eps)==0:
        log.info("no balanced epochs at site {}".format(rec.name))

    else:
        log.info("found {0} balanced epochs:".format(len(balanced_eps)))

    return balanced_eps


def generate_psth(rec):
    """
    Generate a psth contained in the mask and tile it across the whole passed recording, r,
    using replace epochs
    """
    r = rec.copy()
    r_mask = r.apply_mask(reset_epochs=True).copy()

    # get epochs inside the masked data
    ep = np.unique([e for e in r_mask.epochs.name if 'STIM_00' in e]).tolist()

    # for each epoch, compute psth for the masked data, rep it for the number of trials in all the data
    r_psth = dict.fromkeys(ep)
    for e in ep:
        mask_psth = r_mask['resp'].extract_epoch(e).mean(axis=0)[np.newaxis, :, :]
        nreps = r['resp'].extract_epoch(e).shape[0]
        mask_psth = np.tile(mask_psth, [nreps, 1, 1])
        r_psth[e] = mask_psth

    # finally, replace epochs in the orignal recording with the masked psth
    r['psth'] = r['resp'].replace_epochs(r_psth)

    return r

def create_ptd_masks(rec):
    """
    Create active behavior mask, passive mask, passive big pupil mask, and passive small pupil mask.
    return new recording with the four masks as signals.
    """

    r = rec.copy()
    r = r.create_mask(True)

    act_mask = r.and_mask(['HIT_TRIAL'])['mask']
    pass_mask = r.and_mask(['PASSIVE_EXPERIMENT'])['mask']
    miss_mask = r.and_mask(['MISS_TRIAL'])['mask']

    options = {'state': 'big', 'method': 'median', 'collapse': True, 'epoch': ['REFERENCE', 'TARGET']}
    pass_big_mask = create_pupil_mask(r.and_mask(['PASSIVE_EXPERIMENT']), **options)['mask']
    options['state'] = 'small'
    pass_small_mask = create_pupil_mask(r.and_mask(['PASSIVE_EXPERIMENT']), **options)['mask']

    r['a_mask'] = r['mask']._modified_copy(act_mask._data)
    r['p_mask'] = r['mask']._modified_copy(pass_mask._data)
    r['miss_mask'] = r['mask']._modified_copy(miss_mask._data)
    r['pb_mask'] = r['mask']._modified_copy(pass_big_mask._data)
    r['ps_mask'] = r['mask']._modified_copy(pass_small_mask._data)

    return r


def zscore_dict(d, d_norm=None):
    # d_norm is optional - used to compute the mean/std if passed (for example if
    # d is a subset of d_norm and want to normalize to ALL data)

    if d_norm is None:
        d_norm = d

    epochs = d_norm.keys()
    # get mean per neuron
    # get std per neuron
    for i, e in enumerate(epochs):
        if i == 0:
            r = d_norm[e]
        else:
            r = np.concatenate((r, d_norm[e]), axis=0)
    m = r.transpose([0, 2, 1]).reshape(-1, r.shape[1]).mean()
    std = r.transpose([0, 2, 1]).reshape(-1, r.shape[1]).std()

    # normalize dict
    epochs = d.keys()
    for i, e in enumerate(epochs):
        reps = d[e].shape[0]
        bins = d[e].shape[-1]
        cells = d[e].shape[1]
        d[e] = (d[e].transpose([0, 2, 1]).reshape(-1, r.shape[1]) - m).reshape(reps, bins, cells).transpose([0, 2, 1])
        d[e] = (d[e].transpose([0, 2, 1]).reshape(-1, r.shape[1]) / std).reshape(reps, bins, cells).transpose([0, 2, 1])

    return d


def zscore_per_stim(d1, d2):
    d = d1.copy()
    d_norm = d2.copy()
    if d_norm is None:
        d_norm = d

    for k in d_norm.keys():
        reps = d[k].shape[0]
        m = np.tile(d_norm[k].mean(axis=0), [reps, 1, 1])
        std = np.tile(d_norm[k].std(axis=0), [reps, 1, 1])

        d[k] = d[k] - m
        d[k] = d[k] / std

    return d

def pca_reduce_dimensionality(dict_proj, npcs=1, dict_fit=None):
    """
    Compute pcs on psth of d_fit and then project all data from d dict onto the first npcs.
    Return the new dict with reduced dimensionality and the variance explained by npcs
    """
    d = dict_proj.copy()
    if dict_fit is None:
        d_fit = d
    else:
        d_fit = dict_fit.copy()

    epochs = d_fit.keys()
    for i, e in enumerate(epochs):
        if i == 0:
            r = d_fit[e].mean(axis=0)
        else:
            r = np.concatenate((r, d_fit[e].mean(axis=0)), axis=-1)

    # perform PCA
    pca = PCA(n_components=npcs)
    pca = pca.fit(r.T)

    # normalize variance of components
    # pca.components_ = (pca.components_.T / pca.explained_variance_).T

    epochs = d.keys()
    for e in epochs:
        reps = d[e].shape[0]
        bins = d[e].shape[-1]
        cells = d[e].shape[1]
        d[e] = np.matmul(d[e].transpose([-1, 0, 1]).reshape(reps*bins, cells), pca.components_[:npcs, :].T).reshape(bins, reps, npcs).transpose([1, -1, 0])

    return d, np.sum(pca.explained_variance_ratio_[:npcs])