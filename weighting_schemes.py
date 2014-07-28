"""Weighting function."""

import numpy as np
import matplotlib.pylab as plt

def weight(lags, kay1, kay2):
    """Weighting function.

    The source is Ghysels, Santa-Clara, Valkanov (2005, JFE)
    Inputs
        d : lags
        k1, k2 : fixed parameters
    Returns
        vector of weights of the same size as d
    """
    return np.exp(kay1 * lags + kay2 * lags**2) \
        / np.exp(kay1 * lags + kay2 * lags**2).sum()

def test():
    """Test weighting function with both parameters."""
    lags = np.linspace(10, 250, 50)
    grid = 5
    kay1 = np.linspace(0, 1, grid) / 100
    kay2 = np.linspace(-2, 2, grid)[::-1] / 10000
    axes = plt.subplots(nrows=grid, ncols=grid,
                        figsize=(8, 8), sharex=True, sharey=True)[1]
    for i in range(grid):
        for j in range(grid):
            wvec = weight(lags, kay1[i], kay2[j])
            axes[j, i].plot(lags, wvec, lw=2)
            axes[-1, i].set_xlabel('k1 = %.3f' % (kay1[i] * 100))
            axes[j, 0].set_ylabel('k2 = %.3f' % (kay2[j] * 100))

    plt.tight_layout()
    plt.show()

def test2():
    """Test weighting function with one parameter."""
    lags = np.linspace(30, 250, 5)
    grid = 5
    kay2 = np.linspace(-1, 1, grid)[::-1] / 10000
    axes = plt.subplots(nrows=grid, ncols=1, figsize=(8, 8), sharex=True)[1]
    for i in range(grid):
        wvec = weight(lags, 0, kay2[i])
        axes[i].plot(lags, wvec, lw=2)
        axes[i].set_title('k2 = %.3f' % (kay2[i] * 10000))

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    test2()
