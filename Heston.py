# Author:
# Stanislav Khrapov
# mailto:khrapovs@gmail.com
# http://sites.google.com/site/khrapovs/

import numpy as np
from numpy.random import multivariate_normal
import matplotlib.pylab as plt

class Heston(object):
    
    def __init__(self):
        self.eps = None
        self.names = ['mu_r','kappa','mu_v','eta','rho']
    
    def drift_S(self, x, theta):
        """Price drift."""
        mu_r, kappa, mu_v, eta, rho = theta
        # If vol is annualized:
        #return mu_r - .5 * (x/252)
        # If std is given in percentages:
        return mu_r - .5 * (x/1e4)
    
    def drift_V(self, x, theta):
        """Volatility drift."""
        mu_r, kappa, mu_v, eta, rho = theta
        return kappa * (mu_v - x)
    
    def diff_S(self, x, theta):
        """Price diffusion."""
        # If vol is annualized:
        #return (x/252) ** .5
        # If std is given in percentages:
        return (x/1e4) ** .5
    
    def diff_V(self, x, theta):
        """Volatility diffusion."""
        mu_r, kappa, mu_v, eta, rho = theta
        return eta * x ** .5
    
    def drift(self, x, theta):
        """Bivariate drift."""
        dS = self.drift_S(x[:,1], theta)
        dV = self.drift_V(x[:,1], theta)
        return np.vstack([dS, dV]).T
    
    def diff(self, x, theta):
        """Bivariate diffusion."""
        dS = self.diff_S(x[:,1], theta)
        dV = self.diff_V(x[:,1], theta)
        return np.vstack([dS, dV]).T
    
    def is_valid(self, theta):
        """Check Feller condition."""
        mu_r, kappa, mu_v, eta, rho = theta
        return 2 * kappa * mu_v - eta ** 2 > 0
    
    def nice_errors(self, e, sdim):
        """Normalize the errors and apply antithetic sampling.
        
        Inputs:
        e -- untreated innovation array
        sdim -- which dimention corresponds to simulation instances?
        """
        e -= e.mean(sdim, keepdims = True)
        e /= e.std(sdim, keepdims = True)
        e = np.concatenate((e, -e), axis = sdim)
        return e
    
    def assign_data(self, x):
        """Break down generic x into meaningful data."""
        # Log-price
        self.logS = x[:,:,0]
        # Volatility
        self.V = x[:,:,1]
        # Log-returns
        self.R = np.zeros_like(self.logS)
        self.R[1:,] = self.logS[1:,] - self.logS[:-1,]

    def simulate(self, x0, theta, h, T, M, S):
        """Simulate bivariate diffusion.
        
        Inputs:
        x0 -- initial value of the process
        theta -- parameters of the model
        h -- time length of the unit interval
        T -- total time length of the series
        M -- number of intermediate steps for Euler discretization
        S -- number os simulations
        """
        self.h = h
        # How many actual data points to generate?
        N = int(float(T) / h)
        # Make sure that rho is always the last parameter
        rho = theta[-1]
        # Covariance matrix for the errors
        cov = [[1., rho],[rho, 1.]]
        # Generate random normal variables only in case if they don't exist
        if self.eps is None or np.shape(self.eps) != (N, M, S, 2):
            e = multivariate_normal(np.zeros(2), cov, size = (N, M, S/2))
            self.eps = self.nice_errors(e, 2)
        
        # Initialize the simulation array
        x = np.ones((N, S, 2)) * x0
        
        def sim(z, e):
            """Iterates from previous value to the next given the shock."""
            loc = self.drift(z, theta) * (h/M)
            scale = self.diff(z, theta) * (h/M) ** .5
            return z + loc + scale * e
        # Itarete for each unit time interval
        for t in xrange(N-1):
            x[t+1] = reduce(sim, self.eps[t], x[t])
        # Break down generic x into meaningful data
        self.assign_data(x)
        

def test():    
    # .15 corresponds to .15*100 = 15% annualized vol
    # That is, vol is already annualized
    # mu_r, kappa, mu_v, eta, rho = .0, .1, .15**2, .09, -.5
    
    # .9 corresponds to (.9*252)**.5 = 15% annualized vol
    # That is, std is given in percentages
    mu_r, kappa, mu_v, eta, rho = .0, .02, .9, .15, -.5
    theta_true = np.array([mu_r, kappa, mu_v, eta, rho])
    
    hes = Heston()
    print 'Parameters are valid: ', hes.is_valid(theta_true)
    
    x0 = [0., mu_v]
    T, h, M, S = 2000, 1., 100, 10
    hes.simulate(x0, theta_true, h, T, M, S)
    
    R, V = hes.R[:,0], hes.V[:,0]
    
    fig, axes = plt.subplots(2,1)
    axes[0].plot(R)
    axes[1].plot(V)
    plt.show()
    
if __name__ == '__main__':

    print 'Run tests...'
    test()