import matplotlib.pyplot as plt
import numpy as np
import charlieTools.preprocessing as preproc
import copy

def compute_ellipse(x, y):
    inds = np.isfinite(x) & np.isfinite(y)
    x= x[inds]
    y = y[inds]
    data = np.vstack((x, y))
    mu = np.mean(data, 1)
    data = data.T - mu

    D, V = np.linalg.eig(np.divide(np.matmul(data.T, data), data.shape[0] - 1))
    # order = np.argsort(D)[::-1]
    # D = D[order]
    # V = abs(V[:, order])

    t = np.linspace(0, 2 * np.pi, 100)
    e = np.vstack((np.sin(t), np.cos(t)))  # unit circle
    VV = np.multiply(V, np.sqrt(D))  # scale eigenvectors
    e = np.matmul(VV, e).T + mu  # project circle back to orig space
    e = e.T

    return e

def get_square_asp(ax):
    asp = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
    return asp


def plot_raster_psth(rec, epochs, psth_fs=20, ax=None, ylim=None, raster=True):
    """
    Apply mask, extract epochs (all stims, ref only or tar only),
    make raster plot and psth.
    """
    r = copy.deepcopy(rec)
    if 'mask' in r.signals.keys():
        r = r.apply_mask(reset_epochs=True)

    if ax is None:
        f, ax = plt.subplots(1, 1)

    if epochs is None:
        ep = [e for e in r.epochs.name.unique() if 'STIM' in e]
    else:
        ep = epochs

    # compute psth for each epoch and plot
    r = preproc.generate_psth(r)
    max_psth = 0
    colors = []
    for i, e in enumerate(ep):
        try:
            psth = r['psth'].extract_epoch(e)[0, 0, :].squeeze()
            sem = r['psth_sem'].extract_epoch(e)[0, 0, :].squeeze()
            bins = len(psth)
            time = bins / r['resp'].fs
            newbins = time * psth_fs
            if (newbins % 1) != 0:
                raise ValueError("Cannot bin trials of length {0} s at {1} Hz".format(time, psth_fs))
            else:
                newbins = int(newbins)
            psth = preproc.downsample_raster(psth, newbins)
            sem = preproc.downsample_error(sem, newbins)
            if (psth+sem).max()  > max_psth:
                max_psth = (psth+sem).max()

            time = np.arange(0, len(psth) / psth_fs, 1 / psth_fs)
            ax.plot(time, psth, label=e)
            colors.append(ax.get_lines()[-1].get_color())
            ax.fill_between(time, psth-sem, psth+sem, alpha=0.5)    

        except:
            # No epochs matching e. Plot nans just to keep colors straight between subplots
            ax.plot(np.nan, np.nan)
            colors.append(ax.get_lines()[-1].get_color())

    # Now, plot raster for each epoch
    if ylim is None:
        ymin = max_psth + 2
    else:
        ymin = ylim

    if raster:
        epoch_offset = 0
        on_off = False
        for i, e in enumerate(ep):
            try:
                resp = r['resp'].extract_epoch(e).squeeze()
                tr, ti = np.where(resp)
                try:
                    ti = ti / r['resp'].fs
                    tr = (tr / 10) + ymin + epoch_offset
                    ax.plot(ti, tr, '|', markersize=1, color=colors[i])
                except:
                    # no spikes
                    pass
                epoch_offset = (resp.shape[0] / 10) + epoch_offset

                if on_off == False:
                    # plot sound onset / offset
                    on_off = True
                    onset = r['resp'].extract_epoch('PreStimSilence').shape[-1] / r['resp'].fs
                    offset = (resp.shape[-1] - r['resp'].extract_epoch('PostStimSilence').shape[-1]) / r['resp'].fs
                    ax.axvline(onset, color='lightgrey', linestyle='--')
                    ax.axvline(offset, color='lightgrey', linestyle='--')
            except:
                # No epochs matching e. Pass
                pass      

    ax.legend(fontsize=6, loc='upper right', frameon=False)
    ax.set_ylabel('Spk count', fontsize=6)
    ax.set_xlabel('Time (s)', fontsize=6)
        


def plot_raster_psth_perfile(rec, cellid, epochs=None, psth_fs=20):
    """
    Wrapper around plot_raster_psth. Plots raster/psth in separate
    axis for each file in the recording.
    """
    r = rec.copy()
    r['resp'] = r['resp'].extract_channels([cellid])
    if 'mask' in r.signals.keys():
        r = r.apply_mask(reset_epochs=True)
    files = [f for f in r.epochs.name.unique() if 'FILE' in f]

    fig, ax = plt.subplots(1, len(files), sharey=True, figsize=(10, 3))
    
    # first, figure out axis limits
    ylim = get_ylim(r, fs=psth_fs, epochs=epochs)
    
    # Now, plot
    for i, f in enumerate(files):
        rt = copy.deepcopy(r)
        rt = rt.and_mask([f])
        rt = rt.apply_mask(reset_epochs=True)
        plot_raster_psth(rt, epochs, psth_fs, ax=ax[i], ylim=ylim)
        ax[i].set_title(f, fontsize=8)

    fig.canvas.set_window_title(cellid)
    fig.tight_layout()

    return fig


def get_ylim(rec, fs=20, epochs=None):
    r = rec.copy()
    if 'mask' in r.signals.keys():
        r = r.apply_mask(reset_epochs=True)
    files = [f for f in r.epochs.name.unique() if 'FILE' in f]
    
    ylim = 0
    for f in files:
        rt = copy.deepcopy(r)
        rt = rt.and_mask([f])
        rt = rt.apply_mask(reset_epochs=True)
        if epochs is None:
            ep = [e for e in rt.epochs.name.unique() if 'STIM' in e]
        else:
            ep = epochs

        # compute psth for each epoch in order to find ylim
        rt = preproc.generate_psth(rt)
        for e in ep:
            try:
                psth = rt['psth'].extract_epoch(e)[0, 0, :].squeeze()
                sem = rt['psth_sem'].extract_epoch(e)[0, 0, :].squeeze()
                bins = len(psth)
                time = bins / rt['resp'].fs
                newbins = time * fs
                if (newbins % 1) != 0:
                    raise ValueError("Cannot bin trials of length {0} s at {1} Hz".format(time, fs))
                else:
                    newbins = int(newbins)
                psth = preproc.downsample_raster(psth, newbins)
                sem = preproc.downsample_error(sem, newbins)
                if (psth+sem).max() > ylim:
                    ylim = (psth+sem).max()
            except:
                # No epochs matching e. Pass
                pass

    return ylim