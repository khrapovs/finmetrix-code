import numpy as np
import matplotlib.pylab as plt

def weight(d, k1, k2):
    """Weighting function.
    
    The source is Ghysels, Santa-Clara, Valkanov (2005, JFE)
    Inputs
        d : lags
        k1, k2 : fixed parameters
    Returns
        vector of weights of the same size as d
    """
    return np.exp(k1 * d + k2 * d**2) / np.exp(k1 * d + k2 * d**2).sum()

def test():
    d = np.linspace(10,250,50)
    grid = 5
    K1 = np.linspace(0, 1, grid) / 100
    K2 = np.linspace(-2, 2, grid)[::-1] / 10000
    fig, ax = plt.subplots(nrows = grid, ncols = grid,
                           figsize = (8,8), sharex = True, sharey = True)
    for i in range(grid):
        for j in range(grid):
            w = weight(d, K1[i], K2[j])
            ax[j,i].plot(d, w, lw = 2)
            ax[-1,i].set_xlabel('k1 = %.3f' % (K1[i] * 100))
            ax[j,0].set_ylabel('k2 = %.3f' % (K2[j] * 100))
            
    plt.tight_layout()
    plt.show()

def test2():
    d = np.linspace(30,250,5)
    grid = 5
    K2 = np.linspace(-1, 1, grid)[::-1] / 10000
    fig, ax = plt.subplots(nrows = grid, ncols = 1,
                           figsize = (8,8), sharex = True)
    for i in range(grid):
        w = weight(d, 0, K2[i])
        ax[i].plot(d, w, lw = 2)
        ax[i].set_title('k2 = %.3f' % (K2[i] * 10000))
            
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    test2()