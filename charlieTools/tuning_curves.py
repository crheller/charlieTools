import pandas as pd
import numpy as np
import copy

def get_tuning_curves(rec):
    """
    Created with BVT stimuli (narrow band noise) in mind. For each ref epoch, compute mean evoked response.
    Return pandas df with columns - center freq, index - cellids
    """
    r = rec.copy()
    r = r.and_mask(['PreStimSilence', 'PostStimSilence'], invert=True)
    r = r.apply_mask(reset_epochs=True)
    ref_stims = [s for s in r.epochs.name.unique() if 'STIM' in s]
    freqs = [int(s.split('_')[1]) for s in ref_stims]
    idx = np.argsort(freqs)
    ref_stims = np.array(ref_stims)[idx].tolist()
    freqs = np.array(freqs)[idx]

    cellids = r['resp'].chans

    dfr = pd.DataFrame(index=cellids, columns=freqs)
    dfsem = pd.DataFrame(index=cellids, columns=freqs)
    df = pd.concat([dfr, dfsem], keys=['r', 'sem'])

    islice = pd.IndexSlice
    for f, ref in zip(freqs, ref_stims):
        reps = r['resp'].extract_epoch(ref).shape[0]
        resp = np.nanmean(r['resp'].extract_epoch(ref), axis=-1).mean(axis=0)
        sem = np.nanmean(r['resp'].extract_epoch(ref), axis=-1).std(axis=0) / np.sqrt(reps)

        df.loc[islice['r', f]] = resp
        df.loc[islice['sem', f]] = sem

    return df


def get_bf(rec):
    """
    Return df with index = cellid, columns = [BF, sig]
    signficance calculated based on spont mean / sem. If BF mean +/- sem does not overlap
    with spont mean / sem, then significant
    """
    r = copy.deepcopy(rec)
    ftc = get_tuning_curves(r)

    spont_r = r['resp'].extract_epoch('PostStimSilence')
    spont = np.nanmean(spont_r, axis=(0, -1))
    spont_sem = np.nanstd(spont_r, axis=(0, -1)) / np.sqrt((~np.isnan(spont_r[:, 0, :])).sum())
    
    bf = ftc.columns[np.argmax(ftc.loc['r'].to_numpy(np.float), axis=1)]
    bf_r = ftc.loc['r'].max(axis=1)
    bf_sem = ftc.loc['sem'].max(axis=1)

    df = pd.DataFrame(index=ftc.loc['r'].index, columns=['BF', 'sig'])
    sig = abs(bf_r.values - spont) > (bf_sem.values + spont_sem)

    df['BF'] = bf
    df['sig'] = sig

    return df