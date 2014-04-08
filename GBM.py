# Author:
# Stanislav Khrapov
# mailto:khrapovs@gmail.com
# http://sites.google.com/site/khrapovs/

import numpy as np
from Model import Model
import pandas as ps
import statsmodels.api as sm
from compare_estimators import compare_estimators

class GBM(Model):
    
    def __init__(self):
        #super(Model, self).__init__()
        Model.__init__(self)
        self.names = ['mu','sigma']

    def drift(self, x, theta):
        mu, sigma = theta
        return mu - .5 * sigma ** 2
    
    def diff(self, x, theta):
        mu, sigma = theta
        return sigma

    def exact_loc(self, x, theta):
        return self.euler_loc(x, theta)
    
    def exact_scale(self, x, theta):
        return self.euler_scale(x, theta)
        
    def estimate_ols_euler(self, x):
        Y = x[1:] - x[:-1]
        X = np.ones(x.shape[0] - 1) * self.h
        results = sm.OLS(Y, X).fit()
        sigma = (results.mse_resid / self.h) ** .5
        mu = results.params[0] + .5 * sigma **2
        return np.array([mu, sigma])
    
    def estimate_ols_exact(self, x):
        return self.estimate_ols_euler(x)
        
    def quasi_likelihood(self, theta, x, scheme):
        return super(GBM, self).quasi_likelihood(theta, x, scheme)
    
    def exact_likelihood(self, theta, x):
        return self.quasi_likelihood(theta, x, 'exact')

def test():
    #%% GBM model

    # At this point we can initialize a model object using the class GBM.
    mu, sigma = .05, .1
    theta_true = np.array([mu, sigma])
    gbm = GBM()

    x0, T, h, M, S = mu, 200, 1., 100, 3
    N = int(float(T) / h)
    gbm.simulate(x0, theta_true, h, M, N, S)

    # Calling another method creates a simple plot.
    gbm.plot_trajectories(3)

    # Use the first path for furtehr estimation.
    logS = gbm.paths[:,0]

    compare_estimators(gbm, logS, theta_true)

if __name__ == '__main__':

    print 'Run tests...'
    test()