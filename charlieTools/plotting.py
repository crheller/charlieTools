import matplotlib.pyplot as plt
import numpy as np

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

