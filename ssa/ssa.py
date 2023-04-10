import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
def as_vector(x):
    """Convert to vector.
    Examples
    --------
    x = nk.as_vector(x=range(3))
    y = nk.as_vector(x=[0, 1, 2])
    z = nk.as_vector(x=np.array([0, 1, 2]))
    z #doctest: +SKIP

    x = nk.as_vector(x=0)
    x #doctest: +SKIP
    x = nk.as_vector(x=pd.Series([0, 1, 2]))
    y = nk.as_vector(x=pd.DataFrame([0, 1, 2]))
    y #doctest: +SKIP
    """
    if isinstance(x, (pd.Series, pd.DataFrame)):
        out = x.values
    elif isinstance(x, (str, float, int, np.intc, np.int8, np.int16, np.int32, np.int64)):
        out = np.array([x])
    else:
        out = np.array(x)

    if isinstance(out, np.ndarray):
        shape = out.shape
        if len(shape) == 1:
            pass
        elif len(shape) != 1 and len(shape) == 2 and shape[1] == 1:
            out = out[:, 0]
        else:
            raise ValueError(
                "NeuroKit error: we expect the user to provide a "
                "vector, i.e., a one-dimensional array (such as a "
                "list of values). Current input of shape: " + str(shape)
            )
    return out

# =============================================================================
# Singular spectrum analysis (SSA)
# =============================================================================
class SSA():
    def __init__(self, n_components):
        self.trajectory_matrix = None
        self.n_components = n_components
    def ssa_initialisation(self, signal):
        """Singular spectrum analysis (SSA)-based signal separation method.

        SSA decomposes a time series into a set of summable components that are grouped together and
        interpreted as trend, periodicity and noise.

        References
        ----------
        - https://www.kaggle.com/jdarcy/introducing-ssa-for-time-series-decomposition

        """
        # sanitize input
        signal = as_vector(signal)
        # Parameters
        # The window length.
        if self.n_components is None:
            L = 50 if len(signal) >= 100 else int(len(signal) / 2)
        else:
            L = self.n_components
        N = len(signal) # Length.
        if not 2 <= L <= N / 2:
            raise ValueError("`n_components` must be in the interval [2, len(signal)/2].")
        K = N - L + 1 # The number of columns in the trajectory matrix.

        # Embed the time series in a trajectory matrix by pulling the relevant subseries of F,
        # and stacking them as columns.
        X = np.array([signal[i : L + i] for i in range(0, K)]).T

        # Get n components
        d = np.linalg.matrix_rank(X)

        # Decompose the trajectory matrix
        u, sigma, vt = np.linalg.svd(X, full_matrices=False)

        # Initialize components matrix
        components = np.zeros((N, d))
        # Reconstruct the elementary matrices without storing them
        for i in range(d):
            X_elem = sigma[i] * np.outer(u[:, i], vt[i, :])
            X_rev = X_elem[::-1]
            components[:, i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0] + 1, X_rev.shape[1])]

        # Return the components
        return components.T
    def ssa_update(self, new_data):
        new_data = as_vector(new_data)
        N = len(new_data)
        L = self.n_components
        K = N - L + 1
        X = np.array([new_data[i: L + i] for i in range(0, K)]).T

        self.trajectory_matrix = np.concatenate((self.trajectory_matrix, X))
        u, sigma, vt = np.linalg.svd(self.trajectory_matrix, full_matrices=False)


folder = '../data3/*.npz'
for i in glob.glob(folder):
    data = np.load(i, allow_pickle=True)
    name = i[-19:-12]
    train_ts, train_dl, test_ts_1gal, test_dl_1gal, label = data['train_ts'], data['train_dl'], data['test_ts_2gal'], \
                                                            data['test_dl_2gal'], data['label'].item()
    dl = np.concatenate((train_dl, test_dl_1gal))
    test_dl_1gal = test_dl_1gal[~np.isnan(test_dl_1gal).any(axis=1)]
    test_ts_1gal = test_ts_1gal[~np.isnan(test_ts_1gal).any(axis=1)]

    ts = test_dl_1gal[:, 0]
    cps = label['test_2gal']

    train_var_dl = train_dl[:, 1]
    train_ht_dl = train_dl[:, 2]
    test_var_dl = test_dl_1gal[:, 1]
    test_ht_dl = test_dl_1gal[:, 2]
    # multi_test = np.stack((test_var_dl, test_ht_dl), axis=1)
    test_var_dl = np.reshape(test_var_dl, (test_var_dl.shape[0], 1))
    # initialisation
    ssa = SSA(n_components=2)
    X_new = ssa.ssa_initialisation(test_var_dl[:300, :])
    j = 300
    while j < test_var_dl.shape[0]:
        test_var_dl[j:j + 10, :]
        X_new = np.concatenate((X_new, k), axis=1)
        j += 10

    test_var_dl = np.reshape(test_var_dl, (test_var_dl.shape[1]))
    e = X_new[0, :].reshape(-1,1)
    e1 = X_new[1, :].reshape(-1, 1)

    fig = plt.figure()
    fig, ax = plt.subplots(3, figsize=[18, 16], sharex=True)
    ax[0].plot(test_var_dl)
    ax[1].plot(e)
    ax[2].plot(e1)
    plt.show()