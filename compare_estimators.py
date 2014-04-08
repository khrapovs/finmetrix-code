# Author:
# Stanislav Khrapov
# mailto:khrapovs@gmail.com
# http://sites.google.com/site/khrapovs/

import numpy as np
import pandas as ps
import matplotlib as mpl
import matplotlib.pylab as plt

ps.set_option('float_format', '{:8.3f}'.format)

def compare_estimators(model, x, theta_true):
    theta_ols_euler = model.estimate_ols_euler(x)
    theta_ols_exact = model.estimate_ols_exact(x)
    theta_qmle_euler = model.estimate_qmle(x, theta_ols_exact, 'euler')
    theta_qmle_exact = model.estimate_qmle(x, theta_ols_exact, 'exact')
    theta_mle = model.estimate_mle(x, theta_ols_exact)
    theta_sml = model.estimate_sml(x, theta_ols_exact, 10, 100)
    theta_gmm = model.estimate_gmm(x, theta_ols_exact)
    
    data = [theta_true, theta_ols_euler, theta_ols_exact,
            theta_qmle_euler, theta_qmle_exact, theta_mle,
            theta_sml, theta_gmm]
    index = ['True', 'OLS euler', 'OLS exact', 'QMLE euler',
             'QMLE exact', 'MLE', 'SML', 'GMM']
    df = ps.DataFrame(data, index = index, columns = model.names)
    df['logL'] = df.apply(lambda y: - model.exact_likelihood(y, x), axis = 1)
    df['Rank'] = df['logL'].rank(ascending = False).astype(int)
    
    print df