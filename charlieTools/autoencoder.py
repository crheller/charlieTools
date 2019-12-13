import numpy as np
import charlieTools.preprocessing as preproc
import scipy.optimize as opt
from sklearn.decomposition import PCA

def objective(w, big, small, b1=1, b2=1):
    '''
    projects big/small onto 1-D axis, w,
    splits into big/small pupil, computes noise correlations,
    computes the mean difference in noise corr. between big/small
    '''
    w = w / np.linalg.norm(w)
    w = w[:, np.newaxis]
    big = (big.T - big.mean(axis=-1)).T
    small = (small.T - small.mean(axis=-1)).T
    big_proj = np.matmul(big.T, w)
    small_proj = np.matmul(small.T, w)

    # minimization function, but we want the dimension that MAXIMIZES difference
    # in variance so we take the negative of the abs value
    diff = -abs(np.var(big_proj) - np.var(small_proj))

    # add some constraint for w to be shared across neurons? Try to minimize MSE between
    # projection and true data. Without the above pupil constraint, should just give first PC?
    dat = np.concatenate((big, small), axis=-1)
    #proj_out = np.matmul(np.matmul(dat.T, w), w.T).T
    #mse = np.mean((dat - proj_out)**2)

    cost = (b1 * diff) #+ (b2 * mse)

    return cost


def compute_pupil_dep_lv(rec):

    rec = preproc.generate_psth(rec)

    pup_ops = {'state': 'big', 'epoch': ['REFERENCE'], 'collapse': True}
    rec_bp = preproc.create_pupil_mask(rec.copy(), **pup_ops)
    rec_bp = rec_bp.apply_mask(reset_epochs=True)
    pup_ops['state'] = 'small'
    rec_sp = preproc.create_pupil_mask(rec.copy(), **pup_ops)
    rec_sp = rec_sp.apply_mask(reset_epochs=True)

    big = rec_bp['resp']._data - rec_bp['psth']._data
    small = rec_sp['resp']._data - rec_sp['psth']._data
    all = rec['resp']._data - rec['psth']._data
    big = big.T - all.mean(axis=-1)
    big = (big / all.std(axis=-1)).T
    small = small.T - all.mean(axis=-1)
    small = (small / all.std(axis=-1)).T

    full_mat = all

    pca = PCA(n_components=1)
    pca_mat = (full_mat.T - full_mat.mean(axis=-1))
    pca.fit(pca_mat)
    x0 = pca.components_.squeeze()

    lv = opt.minimize(objective, x0, (big, small, 1, 4), options={'gtol':1e-6, 'disp': True})

    weights = lv.x / np.linalg.norm(lv.x)

    return weights
