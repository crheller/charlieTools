import nems_lbhb.baphy as nb
from nems.recording import Recording

def load_site(site, fs=20):
    """
    Load data for all active/passive files at this site with the largest number of stable cellids
        i.e. if the site has a prepassive file where there were only 3 cells but there were 4 other a/p 
        files with 20 cells, will not load the prepassive.
    """
    rawid = which_rawids(site)
    ops = {'batch': 307, 'pupil': 1, 'rasterfs': fs, 'siteid': site, 'stim': 0,
        'rawid': rawid}
    uri = nb.baphy_load_recording_uri(**ops)
    rec = Recording.load(uri)
    rec['resp'] = rec['resp'].rasterize()

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

    return rawid