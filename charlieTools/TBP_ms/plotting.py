import charlieTools.TBP_ms.loaders as loaders
import charlieTools.TBP_ms.decoding as decoding
import charlieTools.plotting as cplt
import nems_lbhb.tin_helpers as thelp
import numpy as np
import matplotlib.pyplot as plt

def dump_ellipse_plot(site, batch, filename, mask=["HIT_TRIAL", "MISS_TRIAL", "CORRECT_REJECT_TRIAL", "PASSIVE_EXPERIMENT"]):
    amask = [m for m in mask if m!="PASSIVE_EXPERIMENT"]
    pmask = ["PASSIVE_EXPERIMENT"]
    Xa, _ = loaders.load_tbp_for_decoding(site=site, 
                                    batch=batch,
                                    wins = 0.1,
                                    wine = 0.4,
                                    collapse=True,
                                    mask=amask,
                                    recache=False)
    Xp, _ = loaders.load_tbp_for_decoding(site=site, 
                                    batch=batch,
                                    wins = 0.1,
                                    wine = 0.4,
                                    collapse=True,
                                    mask=pmask,
                                    recache=False)

    targets = [t for t in Xa.keys() if t.startswith("TAR")]
    catch = [c for c in Xa.keys() if c.startswith("CAT")]

    tar_mat = []
    for t in targets:
        tar_mat.append(Xa[t])
        tar_mat.append(Xp[t])
    tar_mat = np.concatenate(tar_mat, axis=1)
    cat_mat = []
    for c in catch:
        cat_mat.append(Xa[c])
        cat_mat.append(Xp[c])
    cat_mat = np.concatenate(cat_mat, axis=1)
    if len(cat_mat.shape)==2:
        cat_mat = cat_mat[:, :, np.newaxis]

    Xcat = {"TARGET": tar_mat, "CATCH": cat_mat}

    try:
        # make a new concatenated dict with only "TARGET" and "CATCH" category to 
        # define decoding space, then project Xa / Xp into this space
        decoding_space = decoding.get_decoding_space(Xcat, [("TARGET", "CATCH")], 
                                                method="dDR", 
                                                ndims=2)

        # dim reduction axes
        axes = decoding_space[0]

        # set up figure and plot data
        f, ax = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

        for t in targets:
            # plot active data
            proj = Xa[t][:, :, 0].T.dot(axes.T)
            ax[0].plot(proj[:, 0], proj[:, 1], "o", alpha=0.5, label=t)
            el = cplt.compute_ellipse(proj[:, 0], proj[:, 1])
            ax[0].plot(el[0], el[1], color=ax[0].get_lines()[-1].get_color())
            
            # plot passive data
            proj = Xp[t][:, :, 0].T.dot(axes.T)
            ax[1].plot(proj[:, 0], proj[:, 1], "o", alpha=0.5, label=t)
            el = cplt.compute_ellipse(proj[:, 0], proj[:, 1])
            ax[1].plot(el[0], el[1], color=ax[0].get_lines()[-1].get_color())
        
        for c in catch:
            # plot active data
            proj = Xa[c][:, :, 0].T.dot(axes.T)
            ax[0].plot(proj[:, 0], proj[:, 1], "o", color="k", alpha=0.5, label=c)
            el = cplt.compute_ellipse(proj[:, 0], proj[:, 1])
            ax[0].plot(el[0], el[1], color=ax[0].get_lines()[-1].get_color())
            
            # plot passive data
            proj = Xp[c][:, :, 0].T.dot(axes.T)
            ax[1].plot(proj[:, 0], proj[:, 1], "o", color="k", alpha=0.5, label=c)
            el = cplt.compute_ellipse(proj[:, 0], proj[:, 1])
            ax[1].plot(el[0], el[1], color=ax[0].get_lines()[-1].get_color())
        
        ax[1].set_title("Passive")
        ax[0].set_title("Active")
        ax[1].legend(bbox_to_anchor=(1, 1), loc="upper left", frameon=False)
        ax[0].legend(bbox_to_anchor=(1, 1), loc="upper left", frameon=False)

        f.tight_layout()
    except:
        # fi plotting error, plot empthy fig
        f, ax = plt.subplots(1,1,figsize=(5,5))

    f.savefig(filename)