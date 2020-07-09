import numpy as np
import nems.db as nd
import datetime as dt
import pandas as pd

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

def get_training_files(animal, runclass, earliest_date, latest_date=None, min_trials=50):

    an_regex = "%" + animal + "%"

    if latest_date is None:
        latest_date = earliest_date

    # get list of all training parmfiles
    sql = "SELECT parmfile, resppath FROM gDataRaw WHERE runclass=%s and resppath like %s and training = 1 and bad=0 and trials>%s"
    parmfiles = nd.pd_query(sql, (runclass, an_regex, min_trials))

    try:
        parmfiles['date'] = [dt.datetime.strptime('-'.join(x.split('_')[1:-2]), '%Y-%m-%d') for x in parmfiles.parmfile]
        ed = dt.datetime.strptime(earliest_date, '%Y_%m_%d')
        ld = dt.datetime.strptime(latest_date, '%Y_%m_%d')
        parmfiles = parmfiles[(parmfiles.date >= ed) & (parmfiles.date <= ld)]
        return parmfiles
    
    except:
        raise ValueError("No files found")