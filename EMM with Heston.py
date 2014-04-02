# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# EMM with Heston

# <markdowncell>

# Author:<br>
# Stanislav Khrapov<br>
# <a href="mailto:khrapovs@gmail.com">khrapovs@gmail.com</a><br>
# http://sites.google.com/site/khrapovs/<br>

# <markdowncell>

# Set up the environment.

# <codecell>

import numpy as np
import pandas as ps
import matplotlib.pylab as plt
from scipy.optimize import minimize

# <markdowncell>

# Printing options for nicer output.

# <codecell>

mpl.rcParams.update({'font.size': 10})
np.set_printoptions(precision = 3, suppress = True)
ps.set_option('float_format', '{:8.3f}'.format)

# <codecell>

class CIR(object):
    
    def __init__(self):
        self.names = ['kappa','mu','sigma']
    
    def drift(self, x, theta):
        kappa, mu, sigma = theta
        return kappa * (mu - x)
    
    def diff(self, x, theta):
        kappa, mu, sigma = theta
        return sigma * x ** .5
    
    def euler_loc(self, x, theta):
        return self.drift(x, theta) * self.h
    
    def euler_scale(self, x, theta):
        return self.diff(x, theta) * self.h ** .5

    def is_valid(self, theta):
        kappa, mu, sigma = theta
        return 2 * kappa * mu - sigma ** 2 > 0
    
    def simulate(self, x0, theta, h, T, M, S):
        N = int(float(T) / h)
        self.h = h
        self.N = N
        e = np.random.normal(size = (N, M, S/2), scale = h ** .5)
        # Antithetic sampling
        self.eps = np.concatenate((e, -e), axis = 2)
        x = np.ones((N, S)) * x0
        
        def sim(z, e):
            return z + self.euler_loc(z, theta) / M + self.euler_scale(z, theta) / M ** .5 * e
        
        for t in xrange(N-1):
            x[t+1] = reduce(sim, self.eps[t], x[t])
        
        if S > 1:
            self.paths = x
        else:
            self.paths = x.flatten()

# <codecell>

kappa, mu, sigma = .05, .2, .02
theta_true = np.array([kappa, mu, sigma])
x0, T, h, M, S = mu, 1e3, 1., 10, 1e2

cir = CIR()
%time cir.simulate(x0, theta_true, h, T, M, S)

print cir.paths[:,0].shape

plt.plot(cir.paths[:,0])
plt.show()

# <codecell>

class Heston(object):
    
    def __init__(self):
        self.eps = None
        self.names = ['mu_r','kappa','mu_v','eta','rho']
    
    def drift(self, x, theta):
        mu_r, kappa, mu_v, eta, rho = theta
        return np.vstack([mu_r - .5 * x[:,1] ** 2, kappa * (mu_v - x[:,1])]).T
    
    def diff(self, x, theta):
        mu_r, kappa, mu_v, eta, rho = theta
        return np.vstack([x[:,1] ** .5, eta * x[:,1] ** .5]).T
    
    def euler_loc(self, x, theta):
        return self.drift(x, theta) * self.h
    
    def euler_scale(self, x, theta):
        return self.diff(x, theta) * self.h ** .5

    def is_valid(self, theta):
        mu_r, kappa, mu_v, eta, rho = theta
        return 2 * kappa * mu_v - eta ** 2 > 0
    
    def simulate(self, x0, theta, h, T, M, S):
        N = int(float(T) / h)
        self.h = h
        self.N = N
        size = (N, M, S, 2)
        rho = theta[-1]
        cov = [[1, rho],[rho, 1]]
        # Generate random normal variables only in case if they don't exist
        if self.eps is None or np.shape(self.eps) != size:
            e = np.random.multivariate_normal(np.zeros(2), cov, size = (N, M, S/2)) * h ** .5
            # Antithetic sampling
            self.eps = np.concatenate((e, -e), axis = 2)
        
        x = np.ones((N, S, 2)) * x0
        
        def sim(z, e):
            return z + self.euler_loc(z, theta) / M + self.euler_scale(z, theta) / M ** .5 * e
        
        for t in xrange(N-1):
            x[t+1] = reduce(sim, self.eps[t], x[t])
        
        self.S = np.exp(x[:,:,0])
        self.R = np.log(self.S[1:,] / self.S[:-1,])
        self.V = x[:,:,1]

# <codecell>

mu_r, kappa, mu_v, eta, rho = .01, .05, .2, .09, -.9
theta_true = np.array([mu_r, kappa, mu_v, eta, rho])
x0, T, h, M, S = [1, mu], 1e3, 1., 10, 2

hes = Heston()
%time hes.simulate(x0, theta_true, h, T, M, S)

print hes.eps.shape

fig, axes = plt.subplots(2,1)
axes[0].plot(hes.R[:,0])
axes[1].plot(hes.V[:,0])
plt.show()

# <codecell>

def aux_likelihood(self, theta, x):
    from scipy.stats import norm
    m, s = theta
    return - norm.logpdf(x, loc = m, scale = s).mean()

def estimate_amle(self, x, theta0):
    res = minimize(self.aux_likelihood, theta0, args = (x,),
                   method = 'Nelder-Mead', options = {'disp': True})
    return res.x

Heston.aux_likelihood = aux_likelihood
Heston.estimate_amle = estimate_amle

R = hes.R[:,0]

delta = hes.estimate_amle(R, [1.,1.])
print delta

# <codecell>

def moment(self, delta, x):
    m, s = delta
    g1 = (m - x) / s ** 2
    g2 = 1./s - (x - m) ** 2 / s ** 3
    return np.mat(np.vstack([g1, g2]))

def emm_f(self, theta, x, delta):
    from numpy.linalg import inv
    
    x0, T, h, M, S = [1, mu], 1e2, 1., 100, 100
    hes.simulate(x0, theta, h, T, M, S)
    x = hes.R.flatten()
    
    g = self.moment(delta, x)
    W = inv(g * g.T)
    gT = g.mean(1)
    J = gT.T * W * gT
    return float(J)

def estimate_emm(self, x, theta0, delta):
    res = minimize(self.emm_f, theta0, args = (x, delta),
                   method = 'Nelder-Mead', options = {'disp' : True, 'maxiter' : 100})
    return res.x

Heston.moment = moment
Heston.emm_f = emm_f
Heston.estimate_emm = estimate_emm

# <codecell>

print hes.emm_f(theta_true/2, R, delta)
print hes.emm_f(theta_true/4, R, delta)

# <codecell>

%time theta_emm = hes.estimate_emm(R, theta_true/2, delta)

print theta_true
print theta_emm

