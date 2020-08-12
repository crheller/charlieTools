"""
Simulated nested data, see how hierarchical methods handle it (if you sample evenly, or not...)
"""
from charlieTools import statistics as stats
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from itertools import combinations

import statsmodels.api as sm
import statsmodels.formula.api as smf

np.random.seed(123)
# values taken from computing mean / sd over all pairs of cells 
trueDiff = 0.009   # True mean difference in noise corr. for a pair of cells
sd = 0.08          # Variability of this value

# under null hypothesis, do we get false positives with either method?
trueDiff = 0
sd = 0.1
plot = False

nSimulations = 1000
fp_even = []      # count number of false positives
fp_weighted = []  # count number of false positives
fp_wilcox = []
fp_wilcoxg = []

pval_even = []       # heir bootstrap even sampling
pval_weighted = []   # heir bootstrap weighted sampling
pval_wilcox = []     # signed-rank test on raw values, ignore site ID
pval_wilcoxg = []    # signed-rank test on grouped raw values over sites

# option for plotting
if plot:
    f, axes = plt.subplots(5, 2, figsize=(16, 12))

else:
    axes = np.ones(nSimulations)

for sim, ax in zip(range(nSimulations), axes.flatten()):
    print("simulation {0} / {1}".format(sim, nSimulations))
    nSites = 25
    nPairs = np.random.randint(len(list(combinations(range(10), 2))), 
                            len(list(combinations(range(60), 2))), 25)

    # generate simulated data
    d = dict()
    all_vals = []
    siteID = []
    for s in range(nSites):
        # should each site have a different true mean?
        vals = np.random.normal(trueDiff, sd, nPairs[s])
        d[s] = vals
        siteID.extend(s*np.ones(nPairs[s]).tolist())
        all_vals.extend(vals)

    grouped_vals = [d[s].mean() for s in range(nSites)]

    # generate bootstrapped samples
    bs_even = stats.get_bootstrapped_sample(d, even_sample=True, nboot=1000)
    bs_weighted = stats.get_bootstrapped_sample(d, even_sample=False, nboot=1000)

    # get pvalue for each method (and for standard Wilcoxon over the population, ignoring sites, one sided)
    p_even = round(1 - stats.get_direct_prob(np.zeros(len(bs_even)), bs_even)[0], 5)
    p_weighted = round(1 - stats.get_direct_prob(np.zeros(len(bs_even)), bs_weighted)[0], 5)
    p_wilcox = round(ss.wilcoxon(all_vals, np.zeros(len(all_vals)), alternative='greater').pvalue, 5)
    p_wilcoxg = round(ss.wilcoxon(grouped_vals, np.zeros(len(grouped_vals)), alternative='greater').pvalue, 5)

    # plot results

    if plot:
        bins = np.linspace(-0.005, 0.02, 100)
        bins = np.linspace(min(np.append(bs_even, bs_weighted)), max(np.append(bs_even, bs_weighted)), 50)
        ax.hist(bs_even, bins=bins, alpha=0.5, label='Even re-sampling, pvalue: {}'.format(p_even))
        ax.hist(bs_weighted, bins=bins, alpha=0.5, label='Weighted re-sampling, pvalue: {}'.format(p_weighted))
        ax.axvline(np.mean(all_vals), color='k', linestyle='--', lw=1, label='Pop. mean, ignoring site ID" {}'.format(p_wilcox))
        ax.axvline(np.mean(grouped_vals), color='grey', linestyle='--', lw=1, label='Pop. mean, after grouping by site ID: {}'.format(pval_wilcoxg))
        ax.axvline(trueDiff, color='red', linestyle='-', lw=2, label='True population mean')

        ax.legend(frameon=False, fontsize=10, loc='upper left')
        ax.set_xlabel(r"Mean $\Delta$noise correlation", fontsize=12)
        ax.set_ylabel(r"$n$ bootstraps", fontsize=12)
        ax.set_title("Iteration {}".format(sim))
    
    else:
        pval_even.append(p_even)
        pval_weighted.append(p_weighted)
        pval_wilcox.append(p_wilcox)
        pval_wilcoxg.append(p_wilcoxg)
        # just save the false positive rate for each method
        if (p_even <= 0.025) | (p_even >= 0.975):
            fp_even.append(1)
        if (p_weighted <= 0.025) | (p_weighted >= 0.975): 
            fp_weighted.append(1)
        if (p_wilcox <= 0.025) | (p_wilcox >= 0.975):
            fp_wilcox.append(1)
        if (p_wilcoxg <= 0.025) | (p_wilcoxg >= 0.975):
            fp_wilcoxg.append(1)        
if plot:
    f.tight_layout()

    plt.show()

else:
    # print number of false positives for each bootstrap method

    print("Even resampling false positive rate: {0}".format(sum(fp_even) / nSimulations))
    print("\n")
    print("Weighted resampling false positive rate: {0}".format(sum(fp_weighted) / nSimulations))
    print("\n")
    print("Wilcoxon, ignore siteID, false positive rate: {0}".format(sum(fp_wilcox) / nSimulations))
    print("\n")
    print("Wilcoxon, group by siteID, false positive rate: {0}".format(sum(fp_wilcoxg) / nSimulations))

    # plot the pvalue distributions for each method

    f, ax = plt.subplots(1, 1, figsize=(5, 5))

    bins = np.linspace(min(pval_even+pval_weighted+pval_wilcox+pval_wilcoxg), 
                                max(pval_even+pval_weighted+pval_wilcox+pval_wilcoxg), 50)

    ax.hist(pval_even, bins=bins, histtype='step', label='even')
    ax.hist(pval_weighted, bins=bins, histtype='step', label='weighted')
    ax.hist(pval_wilcox, bins=bins, histtype='step', label='Wilcoxon')
    ax.hist(pval_wilcoxg, bins=bins, histtype='step', label='Wilcoxon, grouped by site')
    ax.set_xlabel('p-value (probability greater than 0)')
    ax.set_ylabel(r"$n$ bootstraps")
    ax.legend(frameon=False)

    ax.set_title("False postive rates \n Even sampling: {0}"
                    " \n Weighted sampling: {1}"
                    " \n Wilcoxon (independent samples): {2}"
                    " \n Wilcoxon (grouped by site) {3}".format(sum(fp_even) / nSimulations, 
                                                                sum(fp_weighted) / nSimulations,
                                                                sum(fp_wilcox) / nSimulations,
                                                                sum(fp_wilcoxg) / nSimulations))

    f.tight_layout()

    f.savefig('/auto/users/hellerc/heir_bootstrap_false_positive_rate_{0}sims.png'.format(nSimulations))

    plt.show()