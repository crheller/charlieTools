import numpy as np

def generate_uncorrelated_trials(rec, epochs=None, N=500):
    """
    For the data contained in the epochs specified, and rec['mask'],
    compute the mean/std of all reps of each unique stimulus bin. Using
    these params, generate many (500) pseudo trials. Because this is too
    many trials to shove back into the recording, we have to
    return a dict of numpy arrays (trials x neuron x time) where the
    keys are each unique epoch.

    Purpose is to generate pseudo data with zero correlations
    between neurons.

    CRH -- 09/17/2019
    """

    if type(rec) is not dict:
        if 'mask' not in rec.signals:
            rec = rec.create_mask(True)

        if epochs is not None:
            rec = rec.and_mask(epochs)
        else:
            epochs = np.unique([ep for ep in rec['resp'].epochs.name if 'STIM' in ep]).tolist()

        r = rec.apply_mask(reset_epochs=True)
        r = r.and_mask(['PostStimSilence'], invert=True)
        r = r.apply_mask(reset_epochs=True)

        data = r['resp'].extract_epochs(epochs)
    else:
        data = rec

    pseudo_data = dict.fromkeys(data.keys())
    n_units = data[list(data.keys())[0]].shape[1]
    for ep in data.keys():
        nsegs = data[ep].shape[-1]
        pseudo_data[ep] = np.zeros((N, n_units, nsegs))
        for s in range(nsegs):
            for n in range(0, n_units):
                u = data[ep][:, n, s].mean()
                sd = data[ep][:, n, s].std()
                pseudo_data[ep][:, n, s] = np.random.normal(u, sd, N)

    return pseudo_data


def generated_correlated_trials(r_state, r_all, N=500):
    """
    Generate simulated data with single neuron variance / mean calculated from
    all data (r_all) and covariance specified by r_state.
    For example, r_state could be dictionary of all big pupil trials and r_all
    is dictionary including all trials. Compute single neuron stats on r_all,
    compute covariance matrix on r_state, generate new data.

    r_state / r_all: dictionaries. Keys = epochs. Each value is np.array of
        shape: trial x neuron x time.
        neuron / time dimensions must match between the two dictionaries
    """

    if len(list(r_state.keys())) != len(list(r_all.keys())):
        raise ValueError("mismatched epochs between r_state and r_all")
    else:
        epochs = list(r_state.keys())

    if r_state[epochs[0]].shape[1] != r_all[epochs[0]].shape[1]:
        raise ValueError("mismatched neurons between r_state and r_all")

    if r_state[epochs[0]].shape[2] != r_all[epochs[0]].shape[2]:
        raise ValueError("mismatched trial length between r_state and r_all")

    new_data = dict.fromkeys(r_state.keys())
    for ep in epochs:
        for b in range(0, m_all.shape[-1]):
            # get mean of each neuron across all states
            u_all = r_all[ep][:, :, b].mean(axis=0)
            # get variance of each single neuron across all states
            cov_all = np.diagonal(np.cov(r_all[ep][:, :, b].T))
            # get covariance matrix for the current state (big/small pupil)
            cov_state = np.cov(r_state[ep][:, :, b].T)
            # replace state-dependent single neuron covariance with single
            # neuron variance across states
            np.fill_diagonal(cov_state, cov_all)
            # generate data
            new_data[ep][:, :, b] = np.random.multivariate_normal(u_all, cov_state, (N))

    return new_data
