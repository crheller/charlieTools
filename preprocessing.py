import numpy as np
from sklearn.linear_model import LinearRegression
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

log = logging.getLogger(__name__)


def generate_state_corrected_psth(batch=None, modelname=None, cellids=None):
    """
    Modifies the exisiting recording so that psth signal is the prediction specified
    by the modelname. Designed with stategain models in mind. CRH.

    If the model doesn't exist already in /auto/users/hellerc/results/, this
    will go ahead and fit the model and save it in /auto/users/hellerc/results.

    If the fit dir (from xforms) exists, simply reload the result and call this psth.
    """

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

    return new_rec


def regress_pupil(rec):
    """
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
    rec_bp = rec_bp.and_mask(['PostStimSilence'], invert=True)
    rec_sp = rec_sp.and_mask(['PostStimSilence'], invert=True)
    rec = rec.and_mask(['PostStimSilence'], invert=True)
    rec_bp = rec_bp.apply_mask(reset_epochs=True)
    rec_sp = rec_sp.apply_mask(reset_epochs=True)
    rec = rec.apply_mask(reset_epochs=True)

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
        log.info("no balanced epochs at site {}".format(site))

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
