import nems_lbhb.baphy as nb
from nems0.recording import Recording
import os
import pandas as pd
import numpy as np
from nems_lbhb.baphy_experiment import BAPHYExperiment

def load_site(site, fs=20, return_baphy_manager=False, recache=False):
    """
    Load data for all active/passive files at this site with the largest number of stable cellids
        i.e. if the site has a prepassive file where there were only 3 cells but there were 4 other a/p 
        files with 20 cells, will not load the prepassive.
    """
    rawid = which_rawids(site)
    ops = {'batch': 307, 'pupil': 1, 'rasterfs': fs, 'cellid': site, 'stim': 0,
        'rawid': rawid, 'resp': True, 'recache': recache}
    #rec = nb.baphy_load_recording_file(**ops)
    manager = BAPHYExperiment(batch=307, siteid=site, rawid=rawid)
    rec = manager.get_recording(**ops)
    rec['resp'] = rec['resp'].rasterize()

    if return_baphy_manager:
        return rec, manager
    else:
        return rec

def which_rawids(site):
    if site == 'TAR010c':
        rawid = [123675, 123676, 123677, 123681]                                 # TAR010c
    if site == 'AMT018a':
        rawid = [134965, 134966, 134967]                                         # AMT018a
    if site == 'AMT020a':
        rawid = [135002, 135003, 135004]                                         # AMT020a
    if site == 'AMT022c':
        rawid = [135055, 135056, 135057, 135058, 135059]                         # AMT022c
    if site == 'AMT026a':
        rawid = [135176, 135178, 135179]                                         # AMT026a
    if site == 'BRT026c':
        rawid = [129368, 129369, 129371, 129372]                                 # BRT026c
    if site == 'BRT033b':
        rawid = [129703, 129705, 129706]                                         # BRT033b
    if site == 'BRT034f':
        rawid = [129788, 129791, 129792, 129797, 129799, 129800, 129801]         # BRT034f
    if site == 'BRT036b':
        rawid = [131947, 131948, 131949, 131950, 131951, 131952, 131953, 131954] # BRT036b
    if site == 'BRT037b':
        rawid = [131988, 131989, 131990]                                         # BRT037b
    if site == 'BRT039c':
        rawid = [132094, 132097, 132098, 132099, 132100, 132101]                 # BRT039c
    if site == 'bbl102d':
        rawid = [130649, 130650, 130657, 130661]                                 # bbl102d

    return tuple(rawid)


def load_noise_correlations(modelname, path=None):
    """
    Load data frame of noise correlations for all sites
    """

    if path is None:
        path = '/auto/users/hellerc/results/ptd_ms/noise_correlation'

    files = os.listdir(path)
    files = [f for f in files if modelname in f]

    dfs = []
    for f in files:
        fil = os.path.join(path, f)
        df = pd.read_csv(fil, index_col=0)
        df['site'] = f.split('_')[-1].split('.')[0]
        dfs.append(df)

    df = pd.concat(dfs) 

    return df