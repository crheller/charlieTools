"""
Similar-ish plotting functions to baphy remote (rasters / tuning curves)
"""
import numpy as np
import matplotlib.pyplot as plt

def raster_psth(r, mask=None, epochs=None):
    """
    r is nems recording
    mask is mask to apply prior to extracting spikes
    epochs is list of epochs to plot
    """

def psth(r, chan, epochs=None, mask=None, ep_dur=None, cmap=None, prestim=None, supp_legend=False, ax=None):
    """
    r is nems recording
    chan is resp channel
    epochs is epochs to plot (list / array of strings)
    ep_dur: specify number of bins in epoch presentation

    only include completed trials (e.g. something with nan in range of dur, excluded)
        FA trials for example, will get tossed
    """

    if epochs is None:
        epochs = [n for n in r['resp'].epochs.name.unique() if 'STIM_' in n]

    d = r['resp'].extract_channels([chan]).extract_epochs(epochs, mask=mask)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    if cmap is None:
        cmap = 'tab10'
    colors = plt.cm.get_cmap(cmap, len(epochs))
    for i, e in enumerate(epochs):
        resp = d[e]
        if (ep_dur is not None) & (prestim is not None):
            resp = resp[:, :, :int(ep_dur+prestim+prestim)]
        nanidx = np.isnan(resp.sum(axis=(1, 2)))
        resp = resp[~nanidx, 0, :]

        t = np.linspace(0, resp.shape[-1] / r['resp'].fs, resp.shape[-1])
        ax.plot(t, resp.mean(axis=0) * r['resp'].fs, label=e, color=colors(i))
    
    if prestim is not None:
        ax.axvline(prestim  / r['resp'].fs, linestyle='--', color='k')
        ax.axvline((prestim + ep_dur)  / r['resp'].fs, linestyle='--', color='k')
    ax.set_xlabel('Time (s)', fontsize=6)
    ax.set_ylabel('Spk / sec', fontsize=6)
    ax.set_title(chan, fontsize=6)
    if not supp_legend:
        ax.legend(frameon=False, fontsize=6)

    return ax