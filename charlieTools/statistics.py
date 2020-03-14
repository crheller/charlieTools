import numpy as np
from collections import namedtuple

def _do_jackknife_ttest(x, x0):
    '''
    Perform jackknifed ttest. 
    Null hypothesis is that mean(x) == x0

    x are single trial observations (np.ndarray) 
    x0 is null 
    
    create jackknifed sets of x to estimate mean and standard error,
    
    return alpha level of significance (i.e. 0.001, 0.01, 0.05, or n.s.)
    '''
    x = x.copy()
    x0 = x0
    idx = np.arange(0, len(x))
    jack_sets = _get_jack_sets(idx)

    x -= x0  # data will be centered at 0 if null hypothesis is true

    # generate jackknifed distribution of sample statistic
    obs = np.zeros(len(jack_sets))
    for i, j in enumerate(jack_sets):
        obs[i] = x[j].mean() 

    # calculate jackknife estimates of mean and standard error
    u = np.mean(obs)
    se = (((len(x) - 1) / len(x)) * np.sum((obs - u)**2)) ** (1 / 2) 

    if se > 0:
        z = u / se
    elif u == 0:
        z = 0
    else:
        # if se = 0 and u != 0, then should be significant
        z = 10

    result = namedtuple('jackknifed_ttest_result', 'statistic pvalue')
    # based on z-table
    if abs(z) > 3.3:
        return result(statistic=z, pvalue=0.001)
    elif abs(z) > 2.58:
        return result(statistic=z, pvalue=0.01)
    elif abs(z) > 1.96:
        return result(statistic=z, pvalue=0.05)
    else:
        return result(statistic=z, pvalue=np.nan)


def _get_jack_sets(idx):
    '''
    Given a set of indices, generate njacks random sets of unique observations
    '''

    jack_sets = []
    set_size = len(idx) - 1
    for j in range(len(idx)):
        roll_idx = np.roll(idx, j)
        jack_sets.append(roll_idx[:set_size])

    return jack_sets
