import pandas as pd
import numpy as np
import charlieTools.statistics as stats

def get_psth_pvals(R, spont, cellids=None):
    '''
    Determine if significant response for each cell, for each time bin,
         using jackknifed t-test.

    R: folded response matrix (reps X cells X time)
    R0: folded spont matrix (reps X cells X time)

    cellids: list of cellids corresponding to R.shape[1]
    '''

    if cellids is None:
        index = np.arange(0, R.shape[1])
    else:
        index = cellids

    bins = np.arange(0, R.shape[-1])
    sig_df = pd.DataFrame(index=index, columns=[bins])

    for i, idx in enumerate(index):
        x0 = spont[:, i, :].mean()
        for t in bins:
            x = R[:, i, t]
            res = stats._do_jackknife_ttest(x, x0)
            sig_df.loc[idx, t] = res.pvalue
    
    return sig_df
