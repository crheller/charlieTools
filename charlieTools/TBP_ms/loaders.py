"""
data loading utilities for TBP data
"""
from nems_lbhb.baphy_experiment import BAPHYExperiment
import nems_lbhb.tin_helpers as thelp

def load_tbp_for_decoding(site, batch, mask, fs=10, wins=0.1, wine=0.4, collapse=True, recache=False):
    """
    mask is list of epoch categories (e.g. HIT_TRIAL) to include in the returned data

    return: 
        X - neuron x rep x time bin (spike counts dictionary for each epoch)
        Xp - 1 x rep x time bin (pupil size dict)
    """
    options = {'resp': True, 'pupil': True, 'rasterfs': fs, 'stim': False}
    manager = BAPHYExperiment(batch=batch, cellid=site, rawid=None)
    rec = manager.get_recording(recache=recache, **options)
    rec['resp'] = rec['resp'].rasterize()
    rec = rec.create_mask(True)
    rec = rec.and_mask(mask)
    # get tbp epochs and create spike matrix using mask
    _, _, all_stim = thelp.get_sound_labels(rec)

    bs = int(wins * fs)
    be = int(wine * fs)
    r = rec["resp"].extract_epochs(all_stim, mask=rec["mask"])
    p = rec["pupil"].extract_epochs(all_stim, mask=rec["mask"])
    if collapse:
        r = {k: v[:, :, bs:be].mean(axis=-1, keepdims=True) for k, v in r.items()}
        p = {k: v[:, :, bs:be].mean(axis=-1, keepdims=True) for k, v in p.items()}
    else:
        r = {k: v[:, :, bs:be] for k, v in r.items()}
        p = {k: v[:, :, bs:be] for k, v in p.items()}
    r = {k: v.transpose([1, 0, -1]) for k, v in r.items()}
    p = {k: v.transpose([1, 0, -1]) for k, v in p.items()}
    return r, p 
