import numpy as np

def get_rewarded_targets(rec, manager):
    targets = [e for e in rec.epochs.name.unique() if 'TAR_' in e]
    int_tar = [int(t.replace('TAR_', '')) for t in targets]
    targets = np.array(targets)[np.argsort(int_tar)]
    exptparams = manager.get_baphy_exptparams()[1]
    pump_dur = np.array(exptparams['BehaveObject'][1]['PumpDuration'])
    rew_idx = [True if i>0 else False for i in pump_dur]
    return [str(x) for x in np.array(targets)[rew_idx]]

def get_nonrewarded_targets(rec, manager):
    targets = [e for e in rec.epochs.name.unique() if 'TAR_' in e]
    int_tar = [int(t.replace('TAR_', '')) for t in targets]
    targets = np.array(targets)[np.argsort(int_tar)]
    exptparams = manager.get_baphy_exptparams()[1]
    pump_dur = np.array(exptparams['BehaveObject'][1]['PumpDuration'])
    rew_idx = [True if i>0 else False for i in pump_dur]
    return [str(x) for x in np.array(targets)[~np.array(rew_idx)]]