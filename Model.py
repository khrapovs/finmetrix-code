# Author:
# Stanislav Khrapov
# mailto:khrapovs@gmail.com
# http://sites.google.com/site/khrapovs/

import numpy as np
from numpy.linalg import inv
import matplotlib.pylab as plt
from scipy.optimize import minimize
from scipy.stats.kde import gaussian_kde
from scipy.stats import norm

class Model(object):
    
    def __init__(self):
        self.paths = None
        self.eps = None

    def euler_loc(self, x, theta):
        return self.drift(x, theta) * self.h
    
    def euler_scale(self, x, theta):
        return self.diff(x, theta) * self.h ** .5

    def simulate(self, x0, theta, h, M, N, S):
        self.h = h
        self.N = N
        size = (N, M, S)
        self.eps = np.random.normal(size = size, scale = h ** .5)
        x = np.ones((N, S)) * x0
        
        def sim(z, e):
            return z + self.euler_loc(z, theta) / M \
                + self.euler_scale(z, theta) / M ** .5 * e
        
        for n in xrange(N-1):
            x[n+1] = reduce(sim, self.eps[n], x[n])
        
        if S > 1:
            self.paths = x
        else:
            self.paths = x.flatten()
            
    def plot_trajectories(self, num):
        if self.paths is None:
            print 'Simulate data first!'
        else:
            x = np.arange(0, self.h * self.N, self.h)
            plt.plot(x, self.paths[:,:num])
            plt.xlabel('$t$')
            plt.ylabel('$x_t$')
            plt.show()
    
    def plot_final_distr(self):
        if self.paths is None:
            print 'Simulate data first!'
        else:
            data = self.paths[-1]
            density = gaussian_kde(data)
            x = np.linspace(data.min(), data.max(), 1e2)
            plt.plot(x, density(x))
            plt.xlabel('x')
            plt.ylabel('f')
            plt.show()
            
    def quasi_likelihood(self, theta, x, scheme):
        loc = {'euler' : self.euler_loc, 'exact' : self.exact_loc}
        scale = {'euler' : self.euler_scale, 'exact' : self.exact_scale}
        m = loc[scheme](x[:-1], theta)
        s = scale[scheme](x[:-1], theta)
        return - norm.logpdf(x[1:] - x[:-1], loc = m, scale = s).mean()
    
    def moment(self, theta, x):
        loc = self.exact_loc(x[:-1], theta)
        scale = self.exact_scale(x[:-1], theta)
        X1 = x[1:] - x[:-1] - loc
        X2 = (x[1:] - x[:-1]) ** 2 - loc ** 2 - scale ** 2
        X = np.vstack([X1, X2])
        Z = np.vstack([np.ones_like(x[:-1]), x[:-1]])
        new_shape = (X.shape[0] * Z.shape[0], X.shape[1])
        g = np.reshape(X[:,np.newaxis,:] * Z[np.newaxis,:,:], new_shape)
        return np.mat(g)
    
    def gmm_objective(self, theta, x):
        g = self.moment(theta, x)
        W = inv(g * g.T)
        gT = g.mean(1)
        J = gT.T * W * gT
        return float(J)
    
    def sml_likelihood(self, theta, x, M, S):
        # M: number of subintervals
        # S: number of simulations
        size = (M-1, S, self.N-1)
        
        # Generate random normal variables only in case if they don't exist
        if self.eps is None or np.shape(self.eps) != size:
            self.eps = np.random.normal(size = size)
        
        def sim(z, e):
            return z + self.euler_loc(z, theta) / M \
                + self.euler_scale(z, theta) / M ** .5 * e
        z = reduce(sim, self.eps, x[:-1])
        
        loc = z + self.euler_loc(z, theta) / M
        scale = self.euler_scale(z, theta) / M ** .5
        pdf = norm.pdf(x[1:], loc = loc, scale = scale).mean(0)
        
        return -np.log(pdf).mean()
    
    def estimate_qmle(self, x, theta0, scheme):
        res = minimize(self.quasi_likelihood, theta0, args = (x, scheme),
                       method = 'Nelder-Mead', options = {'disp': False})
        return res.x
    
    def estimate_mle(self, x, theta0):
        res = minimize(self.exact_likelihood, theta0, args = (x,),
                       method = 'Nelder-Mead', options = {'disp': False})
        return res.x
    
    def estimate_gmm(self, x, theta0):
        res = minimize(self.gmm_objective, theta0, args = (x,),
                       method = 'Nelder-Mead', options = {'disp': False})
        return res.x
    
    def estimate_sml(self, x, theta0, M, S):
        res = minimize(self.sml_likelihood, theta0, args = (x, M, S),
                       method = 'Nelder-Mead', options = {'disp': False})
        return res.x