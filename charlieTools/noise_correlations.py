import numpy as np
import sys
import scipy.stats as ss
import scipy.signal as ssig
import pandas as pd
from itertools import permutations, combinations
import scipy
import nems.epoch as ep
import scipy.fftpack as sfp
import charlieTools.preprocessing as preproc

import logging

log = logging.getLogger(__name__)

def z_score(rec, sig):
    r = rec.copy()
    epochs = np.unique([e for e in r.epochs.name if 'STIM_00' in e]).tolist()
    zscore_dict = dict.fromkeys(epochs)
    for k in epochs:
        resp = r[sig].extract_epoch(k)
        m = resp.mean(axis=0)
        std = resp.std(axis=0)
        zr = (resp - m) / std
        zr = np.nan_to_num(zr)
        zscore_dict[k] = zr

    r[sig] = r['resp'].replace_epochs(zscore_dict)

    return r


def compute_noise_correlations(r, **options):
    """
    Function to compute and return the noise correlation matrix for all
    channels in the given recording, r. By default, will also return the mean
    and standard error of this matrix, as well as a matrix of pvals for each
    pair of cells in the matrix.

    options dictionary
    ==========================================================================
        normalize: the signal to subtract in order to get residual responses
            default - psth
        evoked: If true, only compute noise correlations during evoked periods
            default - False
        spont: If true, only compute noise correlations during spont periods
            default - False
        epoch: List of epochs to compute noise correlations for.
            default - ['REFERENCE']
        fs: sampling rate for noise correlations. By default, set to r['resp'].fs
            default - r['resp'].fs
        fft: Use fourier transform to filter data into different frequency bands
            default = False
        band: which frequency band to filter into (for fft):
            default = (0, fs / 2)
        box: to use sqaure cutoff in freq domain for fft
            default: False
        collapse: If true, collapse all stim epochs to a single point
            default - False
        plotting: If true, return the intermediate step - i.e. the collapsed
                    residuals, for plotting purposes
            default - False
        correlogram: If true, compute the cross-correlogram over the specified time lag (window)
            default - False
        window: The time lag length to compute the cross_correlogram over, and integrate over. (in seconds)
            default - None
        verbose: Whether or not to return the residual response matrix
            default - False
    """
    raise DeprecationWarning("Use new, simpler function: noise_correlations.compute_rsc")
    # parse options dictionary
    normalize = options.get('normalize', 'psth')
    evoked = options.get('evoked', False)
    spont = options.get('spont', False)
    stimulus = options.get('epoch', ['REFERENCE'])
    fs = options.get('fs', r['resp'].fs)
    collapse = options.get('collapse', False)
    plotting = options.get('plotting', False)
    correlogram = options.get('correlogram', False)
    fft = options.get('fft', False)
    box = options.get('fft_square_window', False)
    band = options.get('band', (0, r['resp'].fs / 2))
    window = options.get('window', None)
    verbose = options.get('verbose', False)
    zscore = options.get('zscore', False)

    if evoked & spont:
        # can't set evoked and spont to True
        raise ValueError
    if fs > r['resp'].fs:
        # can't upsample the recording. fs must be <= rec['resp'].fs
        raise ValueError
    if correlogram & collapse:
        # Correlogram can only be computed within trial. So, can't collapse over trials because we need continuous time.
        # However, pupil must be masked accordingly! In other words, can't divide up trials. Pupil must be categorized
        # as big or small over the whole stim presentation. So, pupil collapse must be true, but collapse here must not
        # be true. The pupil collapse is taken care of elsewhere.
        raise ValueError
    if fft & (len(stimulus) > 1):
        # can't do fft with anything more than just REF. Need all stims to be same length
        raise ValueError

    rec = r.copy()
    trec = rec.copy()
    trec = trec.create_mask(True)
    trec = trec.and_mask(['REFERENCE']) # just for stimulus timing purposes
    trec = trec.apply_mask(reset_epochs=True)
    trec = trec.create_mask(True)

    # normalize the resp and create residual signal
    residual_data = rec['resp']._data - rec[normalize]._data
    residual = rec['resp']._modified_copy(residual_data)
    residual.name = 'residual'
    rec.add_signal(residual)
    if zscore:
        log.info('z-scoring responses')
        rec = z_score(rec, 'residual')

    # ============= fold the data over all stimuli ===========================
    folded_dict = {}
    folded_mask = {}
    masks = {}
    for s in stimulus:
        folded_dict[s] = rec['residual'].extract_epoch(s)
        folded_mask[s] = rec['mask'].extract_epoch(s)

        # make unique masks for each unique STIM (to be used for sig test...
        # need to shuffle while preserving stim ID). Assuming that if len(stimulus)
        # == 1 then you've asked for all REF
        if len(stimulus) == 1:
            # if there is no data withint the mask, no point in running significant test,
            # so skip this code.
            if rec['mask']._data.sum() > 0:
                un_epochs = np.unique([ep for ep in rec.apply_mask(reset_epochs=True).epochs.name if 'STIM' in ep]).tolist()
                masks = dict.fromkeys(un_epochs)
                for e in un_epochs:
                    temp_rec = rec.copy()
                    temp_rec = temp_rec.apply_mask(reset_epochs=True)
                    temp_rec = temp_rec.and_mask(e)
                    masks[e] = temp_rec['mask'].extract_epoch('REFERENCE')

        # get rid of Nans (needed to do this for batch 314 where we mask either all TOR or NAT for model fitting
        keep = np.argwhere(~np.isnan(folded_dict[s]))
        reps = np.sort(np.unique(keep[:, 0]))
        times = np.sort(np.unique(keep[:, 2]))

        folded_dict[s] = folded_dict[s][reps, :, :]
        folded_dict[s] = folded_dict[s][:, :, times]

        folded_mask[s] = folded_mask[s][reps, :, :]
        folded_mask[s] = folded_mask[s][:, :, times]

    # ==========  extract only evoked or spont period if necessary ===========
    if evoked:
        log.info("extracting only evoked period")
        for s in stimulus:
             # note that we use original r here because the masked r may no longer have pre/post stim epochs
             # depending on how the mask was set (for example, because of pupil size)
             onset = trec['resp'].extract_epoch('PreStimSilence').shape[-1]
             offset = trec['resp'].extract_epoch(s).shape[-1] - \
                     trec['resp'].extract_epoch('PostStimSilence').shape[-1]

             folded_dict[s] = folded_dict[s][:, :, onset:offset]
             folded_mask[s] = folded_mask[s][:, :, onset:offset]

    elif spont:
        log.info("extracting only spont period")
        for s in stimulus:
             onset = 0
             offset = rec['residual'].extract_epoch(s).shape[-1] - \
                     rec['residual'].extract_epoch('PostStimSilence').shape[-1]
             folded_dict[s] = folded_dict[s][:, :, onset:offset]
             folded_mask[s] = folded_mask[s][:, :, onset:offset]

    # ========= if fs is not the same as rec['resp'].fs, resample ===========
    if (fs != rec['resp'].fs) & (collapse == False):
        log.info("resampling each rep of epochs {0} to fs: {1}".format(stimulus, fs))
        for s in stimulus:
            num = int(round(folded_dict[s].shape[-1] / (rec['resp'].fs / fs)))
            folded_dict[s] = ssig.resample(folded_dict[s], num, axis=-1)
            folded_mask[s] = ssig.resample(folded_mask[s], num, axis=-1)

    elif collapse == True:
        log.info("collapsing over entire epochs for {0}".format(stimulus))
        for s in stimulus:
            folded_dict[s] = np.nanmean(folded_dict[s], axis=-1, keepdims=True)
            folded_mask[s] = np.nanmean(folded_mask[s], axis=-1, keepdims=True)
    else:
        log.info("Leave sampling rate at {0} and do not collapse".format(fs))
        pass

    if len(stimulus) == 1:
        trial_len = folded_dict[stimulus[0]].shape[-1]

    # === reshape/concatenate dictionaries to compute noise correlations =====
    for i, s in enumerate(stimulus):
        nreps = folded_dict[s].shape[0]
        nbins = folded_dict[s].shape[-1]
        nchans = folded_dict[s].shape[1]
        rr = np.transpose(folded_dict[s], [1, 0, 2])
        mask = folded_mask[s].squeeze()

        # make sure to only use data contained in the mask
        rr = rr.reshape(nchans, nreps*nbins)
        log.info('reshape and cast mask to bool type')
        mask = mask.reshape(nreps*nbins).astype(np.bool)
        rr = rr[:, mask]

        if i == 0:
            res_resp = rr
        else:
            res_resp = np.concatenate((res_resp, rr), axis=-1)

    if plotting:
        # return the residuals
        return res_resp

    # ================ compute noise correlations ==========================
    if correlogram:
        rsc_matrix = np.zeros((res_resp.shape[0], res_resp.shape[0]))
        combos = list(permutations(range(0, res_resp.shape[0]), 2))
        for i, j in combos:
            if i == j:
                continue
            else:
                # compute the cross-correlogram
                cc = cross_correlation(res_resp[i, :], res_resp[j, :], fs=fs, window=window)
                # integrate the cross-correlogram under the specified window
                rsc_matrix[i, j] = np.trapz(cc)
    if fft:
        # need to reshape res_resp back into neurons X rep X time
        res_resp = res_resp.reshape(res_resp.shape[0], -1, folded_mask[stimulus[0]].shape[-1])
        # band pass filter the response on a per trial basis
        log.info("band pass filtering residual response betweeen {0} and {1} Hz on a trial by trial basis".format(band[0],
                                                                                                               band[1]))
        for rep in range(0, res_resp.shape[1]):
            for n in range(0, res_resp.shape[0]):
                s = res_resp[n, rep, :]
                s_fft = sfp.fft(s)
                w = sfp.fftfreq(s.size, 1 / fs)
                s_cut = s_fft.copy()
                inds = np.argwhere((w >= band[0]) & (w <= band[1]))
                inds2 = np.argwhere((w <= -band[0]) & (w >= -band[1]))
                m = np.zeros(w.shape)

                if box == False:
                    # use tukey window for soft edges
                    if len(inds) <= 5:
                        alpha = 0.8
                    else:
                        alpha = 0.5

                    m[inds] = ssig.tukey(len(inds), alpha)[:, np.newaxis]
                    m[inds2] = ssig.tukey(len(inds2), alpha)[:, np.newaxis]

                    if (band[0] == 0) & (band[1] == fs / 2):
                        # take all the data
                        pass

                    else:
                        s_cut = s_cut * m
                else:
                    # use square window
                    scut_new = np.zeros(s_cut.shape)
                    scut_new[inds] = s_cut[inds]
                    scut_new[inds2] = s_cut[inds2]
                    s_cut = scut_new.copy()

                s_ifft = sfp.ifft(s_cut)
                res_resp[n, rep, :] = s_ifft

        # now that we've bandpass filtered the signal, we unravel it back into neurons x rep*time
        res_resp = res_resp.reshape(res_resp.shape[0], -1)
        # compute correlations
        rsc_matrix = np.corrcoef(res_resp)
        np.fill_diagonal(rsc_matrix, 0)
    else:
        rsc_matrix = np.corrcoef(res_resp)
        np.fill_diagonal(rsc_matrix, 0)

    # ======================= get pvals ====================================
    if correlogram:
        sig_mat = np.nan * np.ones(rsc_matrix.shape)

    elif 0:
        log.info("getting rsc pvals...")
        if np.any(np.isnan(rsc_matrix)):
            sig_mat = np.nan*np.ones(rsc_matrix.shape)
        else:
            sig_mat = get_rsc_pvalues(res_resp, rsc_matrix, masks, trial_len)
    else:
        log.info('skip pvalue calculation')
        sig_mat = np.nan*np.ones(rsc_matrix.shape)

    # =============== format matrix and compute means/error ================
    lower = np.triu(rsc_matrix)
    mean_rsc = np.mean(lower[lower != 0])
    error_rsc = np.std(lower[lower != 0]) / np.sqrt(len(lower[lower != 0]))

    # ============ pack results into an output dictionary ==================
    output = {
        'rsc_matrix': rsc_matrix,
        'mean': mean_rsc,
        'error': error_rsc,
        'pvalues': sig_mat
    }

    if verbose:
        return output, res_resp
    else:
        return output


def get_rsc_pvalues(folded_resp, rsc_matrix, ep_mask=None, trial_len=None):
    #lower = np.tril(rsc_matrix)
    counts = np.zeros(rsc_matrix.shape)

    if ep_mask==None:
        raise ValueError("need epoch masks to compute sig correctly")
    nbins = trial_len
    folded_resp = folded_resp.reshape(folded_resp.shape[0], -1, nbins)

    njacks = 5000
    sl = []
    for i in range(0, njacks):
        # shuffle epochs
        shuf_mat = np.zeros(np.shape(folded_resp))
        for c in np.arange(0, shuf_mat.shape[0]):
            for j, ep in enumerate(ep_mask.keys()):
                shuf_rep_idx = np.argwhere(ep_mask[ep][:, 0, 0]).squeeze()
                if shuf_rep_idx.size != 1:
                    rep_idx = shuf_rep_idx.copy()
                    np.random.shuffle(shuf_rep_idx)
                    shuf_mat[c, rep_idx, :] = folded_resp[c, shuf_rep_idx, :]
                else:
                    if i == 0:
                        log.info("Can't shuffle epoch: {}, only one rep".format(ep))
                    shuf_mat[c, shuf_rep_idx, :] = folded_resp[c, shuf_rep_idx, :]

        r_mat = np.corrcoef(shuf_mat.reshape(folded_resp.shape[0], -1))
        np.fill_diagonal(r_mat, 0)
        #l = np.tril(r_mat)
        #bool_diff = (abs(l) > abs(lower)).astype(np.float)
        bool_diff = (abs(r_mat) > abs(rsc_matrix)).astype(np.float)
        counts = counts + bool_diff
    pvals = counts/njacks

    return pvals


def compute_rsc(d, chans=None):
    """
    Very simple, low level function to compute  noise correlations for a given 
    spike count dictionary. If you want to do any preprocessing (i.e. remove first
    order pupil effects, filter the data etc., this happens before building the 
    spike count dictionary)

    z-score responses in dictionary, and compute the noise correlation matrix. 
    Return a dataframe with index: neuron pair, column: ['rsc', 'pval']
    chans is list of unit names. If none, will just label the index with neuron indices
    """
    resp_dict = d.copy()
    log.info("Compute z-scores of responses for noise correlation calculation")
    resp_dict = preproc.zscore_per_stim(resp_dict, d2=resp_dict)

    log.info("Concatenate responses to all stimuli")
    eps = list(resp_dict.keys())
    nCells = resp_dict[eps[0]].shape[1]
    for i, k in enumerate(resp_dict.keys()):
        if i == 0:
            resp_matrix = np.transpose(resp_dict[k], [1, 0, -1]).reshape(nCells, -1)
        else:
            resp_matrix = np.concatenate((resp_matrix, np.transpose(resp_dict[k], [1, 0, -1]).reshape(nCells, -1)), axis=-1)
    # Note, there will be Nan bins for some neurons
    # (where there were no spikes, std is zero so zscore is nan)
    # these will be excluded in the noise corr. calculation
    combos = list(combinations(np.arange(0, nCells), 2))

    log.info("Computing pairwise rsc values / pvals and saving in df")
    if chans is None:
        df_idx = ["{0}_{1}".format(i, j) for (i, j) in combos]
    else:
        df_idx = ["{0}_{1}".format(chans[i], chans[j]) for (i, j) in combos]
    cols = ['rsc', 'pval']
    df = pd.DataFrame(columns=cols, index=df_idx)
    for i, pair in enumerate(combos):
        n1 = pair[0]
        n2 = pair[1]
        idx = df_idx[i]

        rr = np.isfinite(resp_matrix[n1, :] + resp_matrix[n2, :])
        if rr.sum() >= 2:
            cc, pval = ss.pearsonr(resp_matrix[n1, rr], resp_matrix[n2, rr])
            df.loc[idx, cols] = [cc, pval]
        else:
            df.loc[idx, cols] = [np.nan, np.nan]

    return df