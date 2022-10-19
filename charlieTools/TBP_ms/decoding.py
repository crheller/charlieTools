from dDR.dDR import dDR
from dDR.PCA import PCA
from dDR.utils.decoding import compute_dprime
import numpy as np

def get_noise_axis(X, stims):
    """
    center X then compute first PC
    """
    nc = X[stims[0]].shape[0]
    Xa = (X[stims[0]] - X[stims[0]].mean(axis=1, keepdims=True)).reshape(nc, -1)
    for s in stims[1:]:
        center = X[s] - X[s].mean(axis=1, keepdims=True)
        center = center.reshape(nc, -1)
        Xa = np.concatenate((Xa, center), axis=1)

    pca = PCA(n_components=1)
    pca.fit(Xa.T)
    
    return pca.components_

def get_decoding_space(X, pairs,
                          method="dDR", 
                          noise_space="global",
                          ndims=2,
                          common_space=False):
    """
    Compute dDR space loading vectors for all stimulus pairs
    according to the specified options.
    Return dictionary of transformation matrices for each
    stimulus pair
    """
    # for each stimulus pair, compute the decoding axis
    loading = []
    if method == "dDR":
        additional_axes = ndims - 2
        if additional_axes == 0:
            additional_axes = None
        if noise_space=="global":
            perstim = False
            noise_axis = get_noise_axis(X, stims=list(X.keys()))
        elif noise_space=="targets":
            perstim = False
            noise_axis = get_noise_axis(X, stims=[s for s in X.keys() if ("TAR_" in s) | ("CAT_" in s)])
        elif noise_space=="perstim":
            perstim = True
        else:
            raise ValueError(f"method for noise calculation: {noise_space} is not implemented yet") 

        if common_space:
            # very specialized bit of code. Idea is that we use Target vs. Catch space for all pairs
            targets = [t for t in X.keys() if "TAR_" in t]
            catch = [c for c in X.keys() if "CAT_" in c]
            tar_resp = []
            for t in targets:
                tar_resp.append(X[t])
            cat_resp = []
            for c in catch:
                cat_resp.append(X[c])
            X["TARGET"] = np.concatenate(tar_resp, axis=1)
            X["CATCH"] = np.concatenate(cat_resp, axis=1)
            pairs = [("TARGET", "CATCH")]*len(pairs)

        for p in pairs:
            if perstim==False:
                ddr = dDR(ddr2_init=noise_axis, n_additional_axes=additional_axes)
            else:
                ddr = dDR(n_additional_axes=additional_axes)
            
            ddr.fit(X[p[0]][:, :, 0].T, X[p[1]][:, :, 0].T)
            loading.append(ddr.components_)
    else:
        raise ValueError(f"method: {method} is not implemented yet")
    
    return loading

def get_category(e1, e2):
    """
    Assign category to decoding result based on the 
    stim names
    """
    if ("TAR_" in e1) & ("TAR_" in e2):
        return "tar_tar"
    elif ("CAT_" in e1) & ("CAT_" in e2):
        return "cat_cat"
    elif ("STIM_" in e1) & ("STIM_" in e2):
        return "ref_ref"
    elif (("STIM_" in e1) & ("CAT_" in e2)) | (("CAT_" in e1) & ("STIM_" in e2)):
        return "ref_cat"
    elif (("TAR_" in e1) & ("CAT_" in e2)) | (("CAT_" in e1) & ("TAR_" in e2)):
        return "tar_cat"
    elif (("TAR_" in e1) & ("STIM_" in e2)) | (("STIM_" in e1) & ("TAR_" in e2)):
        return "tar_ref"
    elif ((e1=="TARGET") & (e2=="CATCH")) | ((e2=="TARGET") & (e1=="CATCH")):
        return "aTAR_aCAT"
    else:
        raise ValueError(f"Could not find a category match for {e1} / {e2}")


def do_decoding(x1, x2, axes, wopt=None):
    """
    x1/x2 are response to a stimulus (neuron x reps)
    axes are the loading vectors to project on before computing dprime
    """
    x1 = x1.T.dot(axes.T)
    x2 = x2.T.dot(axes.T)
    output = compute_dprime(x1.T, x2.T, diag=False, wopt=wopt)
    return output


