"""
data loading utilities for TBP data
"""
from nems_lbhb.baphy_experiment import BAPHYExperiment
import nems_lbhb.tin_helpers as thelp
import numpy as np
import pickle

def load_tbp_for_decoding(site, batch, mask, fs=10, wins=0.1, wine=0.4, collapse=True, recache=False, balance=False):
    """
    mask is list of epoch categories (e.g. HIT_TRIAL) to include in the returned data

    balance: If true, on a per stimulus basis, make sure there are equal number of 
        active and passive trials (using random subsampling of larger category)

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

    if (len(mask) > 1) & ("PASSIVE_EXPERIMENT" in mask) & balance:
        np.random.seed(123)
        # balance active vs. passive trials
        rnew = rec.copy()
        rnew = rnew.create_mask(True)
        rnew = rnew.and_mask(["PASSIVE_EXPERIMENT"])
        for s in all_stim:
            pmask = rnew["mask"].extract_epoch(s, mask=rec["mask"])
            nptrials = pmask[:, 0, 0].sum() 
            natrials = (pmask[:, 0, 0] == False).sum() 
            nktrials = np.min([nptrials, natrials])
            pchoose = np.random.choice(np.argwhere(pmask[:, 0, 0]).squeeze(), nktrials, replace=False)
            achoose = np.random.choice(np.argwhere(pmask[:, 0, 0]==False).squeeze(), nktrials, replace=False)
            choose = np.sort(np.concatenate((achoose, pchoose)))
            r[s] = r[s][:, choose, :]
            p[s] = p[s][:, choose, :]
    return r, p 


def load_FA_model(site, batch, psth, state, sim=1, rr=None, fa_model="FA", nreps=2000):
    """
    pretty specialized code to load the results of factor analysis model
    and generate data based on this. Since only one psth, if you want to manipulate first
    order (e.g. swap psth for active / passive) that has to happen outside this function.

    psth should be a dictionary with entries of len nCells
    return a dictionary with entries nCells x nreps (simulated)

    generate nreps per stimulus

    state = active or passive

    sim:
        0 = no change (null) model
        1 = change in gain only
        2 = change in indep only (fixing absolute covariance)
        3 = change in indep only (fixing relative covariance - so off-diagonals change)
        4 = change in everything (full FA simulation)
        # extras:
        5 = set off-diag to zero, only change single neuron var.
        6 = set off-diag to zero, fix single neuorn var
        7 = no change (and no correlations at all)
    """
    np.random.seed(123)
    # load the model results. This hardcoding is a bit kludgy
    path = f"/auto/users/hellerc/results/TBP-ms/factor_analysis/{batch}/{site}/"
    filename = f"{fa_model}.pickle"
    with open(path + filename, 'rb') as handle:
        results = pickle.load(handle)

    # if reduced rank, then compute the new, reduced rank shared matrix here (doesn't apply for diag)
    def sigma_shared(components):
        return (components.T @ components)
    if rr is not None:
        active_factors_unique = results["final_fit"]["fa_active.components_"][:rr, :]
        passive_factors_unique = results["final_fit"]["fa_passive.components_"][:rr, :]
        # then, share the rest (big for both)
        factors_shared = results["final_fit"]["fa_active.components_"][rr:, :]
        if factors_shared.shape[0]>0:
            results["final_fit"]["fa_active.sigma_shared"] = sigma_shared(np.concatenate((active_factors_unique, factors_shared), axis=0))
            results["final_fit"]["fa_passive.sigma_shared"] = sigma_shared(np.concatenate((passive_factors_unique, factors_shared), axis=0))
        else:
            results["final_fit"]["fa_active.sigma_shared"] = sigma_shared(active_factors_unique)
            results["final_fit"]["fa_passive.sigma_shared"] = sigma_shared(passive_factors_unique)

    Xsim = dict.fromkeys(psth.keys())
    if sim==0:
        cov_active = results["final_fit"]["fa_active.sigma_ind"] + results["final_fit"]["fa_active.sigma_shared"]
        cov_passive = results["final_fit"]["fa_active.sigma_ind"] + results["final_fit"]["fa_active.sigma_shared"]
    if sim==1:
        cov_active = results["final_fit"]["fa_active.sigma_ind"] + results["final_fit"]["fa_active.sigma_shared"]
        cov_passive = results["final_fit"]["fa_active.sigma_ind"] + results["final_fit"]["fa_active.sigma_shared"]
    elif sim==2:
        # absolute covariance fixed, but fraction shared variance can change
        cov_active = results["final_fit"]["fa_active.sigma_ind"] + results["final_fit"]["fa_active.sigma_shared"]
        cov_passive = results["final_fit"]["fa_passive.sigma_ind"] + results["final_fit"]["fa_active.sigma_shared"]
    elif sim==3:
        # relative covariance fixed, i.e. fraction shared variance can stays the same but absolute covariance can change
        cov_active = results["final_fit"]["fa_active.sigma_ind"] + results["final_fit"]["fa_active.sigma_shared"]
        cov_passive = results["final_fit"]["fa_passive.sigma_ind"] + results["final_fit"]["fa_active.sigma_shared"]
        # force small to have same corr. coef. as cov_active
        norm = np.sqrt(np.diag(cov_active)[:, np.newaxis] @ np.diag(cov_active)[np.newaxis, :])
        corr_active = cov_active / norm # normalize covariance
        var = np.diag(cov_passive) # variance of small pupil          
        rootV = np.sqrt(var[:, np.newaxis] @ var[np.newaxis, :])
        cov_passive = corr_active * rootV # cov small has same (normalized) correlations as cov_active, but variance like cov_passive
    elif sim==4:
        cov_active = results["final_fit"]["fa_active.sigma_ind"] + results["final_fit"]["fa_active.sigma_shared"]
        cov_passive = results["final_fit"]["fa_passive.sigma_ind"] + results["final_fit"]["fa_passive.sigma_shared"]
    elif sim==5:
        # diag matrix, entries change between large and small
        cov_active = results["final_fit"]["fa_active.sigma_ind"]
        cov_passive = results["final_fit"]["fa_passive.sigma_ind"]
    elif sim==6:
        # diag matrix, entries fixed to big pupil between states
        cov_active = results["final_fit"]["fa_active.sigma_ind"]
        cov_passive = results["final_fit"]["fa_active.sigma_ind"]
    elif sim==7:
        cov_active = results["final_fit"]["fa_active.sigma_ind"]
        cov_passive = results["final_fit"]["fa_active.sigma_ind"]

    for s, k in enumerate(psth.keys()):
        _ca = cov_active.copy()            
        _cp = cov_passive.copy()
        if state=="active":
            cov_to_use = _ca
        elif state=="passive":
            cov_to_use = _cp
        Xsim[k] = np.random.multivariate_normal(psth[k], cov=cov_to_use, size=nreps).T

    return Xsim

def load_FA_model_perstim(site, batch, psth, state, sim=1, fa_model="FA_perstim", nreps=2000):
    """
    Only distinct from the above function in that this loads / uses per-stimulus FA results

    pretty specialized code to load the results of factor analysis model
    and generate data based on this. Since only one psth, if you want to manipulate first
    order (e.g. swap psth for active / passive) that has to happen outside this function.

    psth should be a dictionary with entries of len nCells
    return a dictionary with entries nCells x nreps (simulated)

    generate nreps per stimulus

    state = active or passive

    sim:
        0 = no change (null) model
        1 = change in gain only
        2 = change in indep only (fixing absolute covariance)
        3 = change in indep only (fixing relative covariance - so off-diagonals change)
        4 = change in everything (full FA simulation)
        # extras:
        5 = set off-diag to zero, only change single neuron var.
        6 = set off-diag to zero, fix single neuorn var
        7 = no change (and no correlations at all)
    """
    np.random.seed(123)
    # load the model results. This hardcoding is a bit kludgy
    path = f"/auto/users/hellerc/results/TBP-ms/factor_analysis/{batch}/{site}/"
    filename = f"{fa_model}.pickle"
    with open(path + filename, 'rb') as handle:
        results = pickle.load(handle)

    cov_active = dict.fromkeys(psth.keys())
    cov_passive = dict.fromkeys(psth.keys())
    keep_keys = []
    for k in psth.keys():
        try:
            if sim==0:
                cov_active[k] = results["active"][k]["sigma_ind"] + results["active"][k]["sigma_shared"]
                cov_passive[k] = results["active"][k]["sigma_ind"] + results["active"][k]["sigma_shared"]
            if sim==1:
                cov_active[k] = results["active"][k]["sigma_ind"] + results["active"][k]["sigma_shared"]
                cov_passive[k] = results["active"][k]["sigma_ind"] + results["active"][k]["sigma_shared"]
            elif sim==2:
                # absolute covariance fixed, but fraction shared variance can change
                cov_active[k] = results["active"][k]["sigma_ind"] + results["active"][k]["sigma_shared"]
                cov_passive[k] = results["passive"][k]["sigma_ind"] + results["active"][k]["sigma_shared"]
            elif sim==3:
                # relative covariance fixed, i.e. fraction shared variance can stays the same but absolute covariance can change
                cov_active[k] = results["active"][k]["sigma_ind"] + results["active"][k]["sigma_shared"]
                cov_passive[k] = results["passive"][k]["sigma_ind"] + results["active"][k]["sigma_shared"]
                # force small to have same corr. coef. as cov_active
                norm = np.sqrt(np.diag(cov_active[k])[:, np.newaxis] @ np.diag(cov_active[k])[np.newaxis, :])
                corr_active = cov_active[k] / norm # normalize covariance
                var = np.diag(cov_passive[k]) # variance of small pupil          
                rootV = np.sqrt(var[:, np.newaxis] @ var[np.newaxis, :])
                cov_passive[k] = corr_active * rootV # cov small has same (normalized) correlations as cov_active, but variance like cov_passive
            elif sim==4:
                cov_active[k] = results["active"][k]["sigma_ind"] + results["active"][k]["sigma_shared"]
                cov_passive[k] = results["passive"][k]["sigma_ind"] + results["passive"][k]["sigma_shared"]
            elif sim==5:
                # diag matrix, entries change between large and small
                cov_active[k] = results["active"][k]["sigma_ind"]
                cov_passive[k] = results["passive"][k]["sigma_ind"]
            elif sim==6:
                # diag matrix, entries fixed to big pupil between states
                cov_active[k] = results["active"][k]["sigma_ind"]
                cov_passive[k] = results["passive"][k]["sigma_ind"]
            elif sim==7:
                cov_active[k] = results["active"][k]["sigma_ind"]
                cov_passive[k] = results["active"][k]["sigma_ind"]

            keep_keys.append(k)
        except KeyError:
            print(f"Missing key {k}. Skip simulation for this epoch")

    Xsim = dict.fromkeys(keep_keys)
    for k in keep_keys:
        _ca = cov_active[k].copy()            
        _cp = cov_passive[k].copy()
        if state=="active":
            cov_to_use = _ca
        elif state=="passive":
            cov_to_use = _cp
        Xsim[k] = np.random.multivariate_normal(psth[k], cov=cov_to_use, size=nreps).T

    return Xsim