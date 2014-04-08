# Author:
# Stanislav Khrapov
# mailto:khrapovs@gmail.com
# http://sites.google.com/site/khrapovs/

import numpy as np
from math import exp, log

from Model import Model
from compare_estimators import compare_estimators

class Vasicek(Model):
    
    def __init__(self):
        #super(Model, self).__init__()
        Model.__init__(self)
        self.names = ['kappa','mu','sigma']
        
    def drift(self, x, theta):
        kappa, mu, sigma = theta
        return kappa * (mu - x)
    
    def diff(self, x, theta):
        kappa, mu, sigma = theta
        return sigma

    def exact_loc(self, x, theta):
        kappa, mu, sigma = theta
        e = exp( - kappa * self.h )
        return (mu - x) * (1 - e)
    
    def exact_scale(self, x, theta):
        kappa, mu, sigma = theta
        e = exp( - 2 * kappa * self.h )
        return ( (1 - e) * sigma ** 2 / (2 * kappa) ) ** .5
    
    def estimate_ols_euler(self, x):
        import statsmodels.api as sm
        Y, X = x[1:], x[:-1]
        X = sm.add_constant(X)    
        results = sm.OLS(Y, X).fit()
        kappa = (1 - results.params[1]) / self.h
        mu = results.params[0] / kappa / self.h
        sigma = (results.mse_resid / self.h) ** .5
        return np.array([kappa, mu, sigma])
    
    def estimate_ols_exact(self, x):
        import statsmodels.api as sm
        Y, X = x[1:], x[:-1]
        X = sm.add_constant(X)
        results = sm.OLS(Y, X).fit()
        kappa = - log(results.params[1]) / self.h
        mu = results.params[0] / (1 - results.params[1])
        sigma = (2 * kappa * results.mse_resid / \
            (1 - results.params[1] ** 2)) ** .5
        return np.array([kappa, mu, sigma])
        
    def quasi_likelihood(self, theta, x, scheme):
        return super(Vasicek, self).quasi_likelihood(theta, x, scheme)
    
    def exact_likelihood(self, theta, x):
        return self.quasi_likelihood(theta, x, 'exact')

def test():
    #%% Vasicek model

    kappa, mu, sigma = 1.5, .5, .1
    theta_true = np.array([kappa, mu, sigma])
    x0, T, h, M, S = mu, 200, 1., 100, 1e3
    N = int(float(T) / h)

    vasicek = Vasicek()
    vasicek.simulate(x0, theta_true, h, M, N, S)

    vasicek.plot_trajectories(3)
    vasicek.plot_final_distr()

    r = vasicek.paths[:,0]


    compare_estimators(vasicek, r, theta_true)

if __name__ == '__main__':

    print 'Run tests...'
    test()