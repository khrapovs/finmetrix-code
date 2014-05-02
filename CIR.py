# Author:
# Stanislav Khrapov
# mailto:khrapovs@gmail.com
# http://sites.google.com/site/khrapovs/

from __future__ import print_function
from math import exp, log
import numpy as np
import statsmodels.api as sm
from scipy.stats import ncx2
        
from Model import Model
from compare_estimators import compare_estimators

class CIR(Model):
    
    def __init__(self):
        Model.__init__(self)
        #super(Model, self).__init__()
        self.names = ['kappa','mu','sigma']
    
    def drift(self, x, theta):
        kappa, mu, sigma = theta
        return kappa * (mu - x)
    
    def diff(self, x, theta):
        kappa, mu, sigma = theta
        return sigma * x ** .5
    
    def is_valid(self, theta):
        kappa, mu, sigma = theta
        return 2 * kappa * mu - sigma ** 2 > 0

    def exact_loc(self, x, theta):
        kappa, mu, sigma = theta
        e = exp( - kappa * self.h )
        return (mu - x) * (1 - e)
    
    def exact_scale(self, x, theta):
        kappa, mu, sigma = theta
        e = exp( - kappa * self.h )
        return ( (x * e + (1 - e) / 2) * (1 - e) * sigma ** 2 / kappa ) ** .5
    
    def estimate_ols_euler(self, x):
        Y = (x[1:] - x[:-1]) / x[:-1] ** .5
        X1 = self.h / x[:-1] ** .5
        X2 = - self.h * x[:-1] ** .5
        X = np.vstack([X1, X2]).T
        results = sm.OLS(Y, X).fit()
        kappa = results.params[1]
        mu = results.params[0] / results.params[1]
        sigma = (results.mse_resid / self.h) ** .5
        return np.array([kappa, mu, sigma])
    
    def estimate_ols_exact(self, x):
        Y, X = x[1:], x[:-1]
        X = sm.add_constant(X)
        results = sm.OLS(Y, X).fit()
        kappa = - log(results.params[1]) / self.h
        mu = results.params[0] / (1 - results.params[1])
        Y = results.resid ** 2
        results = sm.OLS(Y, X).fit()
        sigma2 = kappa * results.params[1] / (1 - exp(- self.h * kappa)) \
            / exp(- self.h * kappa)
        return np.array([kappa, mu, sigma2 ** .5])
    
    def quasi_likelihood(self, theta, x, scheme):
        return super(CIR, self).quasi_likelihood(theta, x, scheme)
    
    def exact_likelihood(self, theta, x):
        kappa, mu, sigma = theta
        e = exp( - kappa * self.h )
        c = 2 * kappa / sigma ** 2 / (1 - e)
        q = 2 * kappa * mu / sigma ** 2 - 1
        v = 2 * c * x[1:]
        df = 2 * (q + 1)
        nc = 2 * c * x[:-1] * e
        l = ncx2.logpdf(v, df, nc) + log(2 * c)
        return - l[np.isfinite(l)].mean()

def test():
    #%% Cox-Ingersoll-Ross model

    kappa, mu, sigma = 1, .2, .2
    theta_true = np.array([kappa, mu, sigma])
    x0, T, h, M, S = mu, 200, 1., 100, 1e2
    N = int(float(T) / h)

    cir = CIR()
    cir.simulate(x0, theta_true, h, M, N, S)
    cir.is_valid(theta_true)

    cir.plot_trajectories(3)
    cir.plot_final_distr()

    r = cir.paths[:,0]

    compare_estimators(cir, r, theta_true)

if __name__ == '__main__':

    print('Run tests...')
    test()