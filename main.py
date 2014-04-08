# Author:
# Stanislav Khrapov
# mailto:khrapovs@gmail.com
# http://sites.google.com/site/khrapovs/

import numpy as np
import pandas as ps
import matplotlib as mpl
import matplotlib.pylab as plt

from GBM import GBM
from Vasicek import Vasicek
from CIR import CIR
from compare_estimators import compare_estimators

# Printing options for nicer output.
mpl.rcParams.update({'font.size': 10})
np.set_printoptions(precision = 3, suppress = True)
ps.set_option('float_format', '{:8.3f}'.format)

#%% Real data

import zipfile
import StringIO
import datetime as dt

p = '../../../Research/data/OxfordMan/oxfordmanrealizedvolatilityindices.zip'
zf = zipfile.ZipFile(p)
name = zf.namelist()[0]
data = StringIO.StringIO(zf.read(name))
f = lambda x: dt.datetime.strptime(str(int(x)), '%Y%m%d')
raw = ps.read_csv(data, skiprows = [0,1], converters = {'DateID' : f})
raw = raw.rename(columns = {raw.columns[0] : 'date'}).set_index('date')
RV = raw[['SPX2.rv']].dropna()

RV.plot()
plt.show()

RV = RV[:200]

cir = CIR()
cir.h, cir.N = 1., RV.shape[0]
V = np.log( 1 + np.array(RV).flatten() * 25200 )

theta_ols_euler = cir.estimate_ols_euler(V)
theta_ols_exact = cir.estimate_ols_exact(V)
theta_qmle_euler = cir.estimate_qmle(V, theta_ols_exact, 'euler')
theta_qmle_exact = cir.estimate_qmle(V, theta_ols_exact, 'exact')
theta_mle = cir.estimate_mle(V, theta_ols_exact)
theta_sml = cir.estimate_sml(V, theta_ols_exact, 10, 100)
theta_gmm = cir.estimate_gmm(V, theta_ols_exact)

data = [theta_ols_euler, theta_ols_exact, theta_qmle_euler,
        theta_qmle_exact, theta_mle, theta_sml, theta_gmm]
index = ['OLS euler', 'OLS exact', 'QMLE euler', 'QMLE exact',
         'MLE', 'SML', 'GMM']
cols = ['kappa','mu','sigma']
df = ps.DataFrame(data, index = index, columns = cols)
df['logL'] = df.apply(lambda x: - cir.exact_likelihood(x, V), axis = 1)

print df