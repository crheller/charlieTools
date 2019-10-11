import numpy as np

def regress_pupil(r, recache=False):
    """
    Remove first-order effect on the trial-trial variability for each neuron

    1) Compute compute psth in each bin by averaging over the whole recording
    2) At each time bin, model residuals as linear function of pupil
    3) Subtract off prediction
    4) Add the corrected residuals back to the overall psth
    """
    rec = r.copy()
    ep = np.unique([ep for ep in rec['resp'].epochs.name if 'STIM_00' in ep]).tolist()
    r_st = rec['resp'].extract_epochs(ep)
    p_st = rec['pupil'].extract_epochs(ep)
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
                y_residual = y - y_pred.squeeze()
                r_new[e][:, n, b] = r_psth[e][:, n, b] + y_residual

    rec['psth'] =  rec['resp'].replace_epochs(r_psth)
    rec['resp'] = rec['resp'].replace_epochs(r_new)

    return rec


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

    spont_bins = rec['resp'].extract_epoch('PreStimSilence').shape[-1]

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
