import numpy as np
from matplotlib.mlab import prctile
import matplotlib.pyplot as plt
import pylab
from pylab import *
import statsmodels.api as sm  # recommended import according to the docs
import argparse


dir = '/home/jzh/Dropbox/Research/Anomaly_Detection/Experimental_Results'

KL = np.load(dir + '/N_5/eta_KL_mf.npz')

n_range = KL['n_range']
KL_actual = KL['KL_actual']
KL_wc_1 = KL['KL_wc_1']
KL_wc_2 = KL['KL_wc_2']

KL.close()

# print n_range
# print KL_actual[17, :]
# print KL_wc[0]
# print KL_actual.shape

L = len(n_range)

parser = argparse.ArgumentParser()
parser.add_argument("-j", default=0, type=int, \
                    help="index of n_range; should be a positive integer less than the length of n_range")
args = parser.parse_args()

j = args.j
if j < 0 or j >= L:
    raise Exception("Invalid j; please specify -j to be a positive integer less than the length of n_range.")

n = n_range[j]
KL_1 = KL_actual[j, :]

ecdf_1 = sm.distributions.ECDF(KL_1)
x_1 = np.linspace(min(KL_1), max(KL_1), num=100)
y_1 = ecdf_1(x_1)

KL_2 = KL_wc_1[j, :]
ecdf_2 = sm.distributions.ECDF(np.array(KL_2).tolist())
x_2 = np.linspace(min(KL_2), max(KL_2), num=100)
y_2 = ecdf_2(x_2)

KL_3 = KL_wc_2[j, :]
ecdf_3 = sm.distributions.ECDF(np.array(KL_3).tolist())
x_3 = np.linspace(min(KL_3), max(KL_3), num=100)
y_3 = ecdf_2(x_3)

KL_actual, = plt.plot(x_1, y_1, "r")
KL_wc_1, = plt.plot(x_2, y_2, "b--")
KL_wc_2, = plt.plot(x_3, y_3, "g--")

plt.legend([KL_actual, KL_wc_1, KL_wc_2], ["actual", "estimated_1", "estimated_2 (simplified)"], loc=4)
plt.title('Empirical CDF of the relative entropy ($N = 5$, $n = %d$)'%n)

pylab.ylim(0, 1.01)

savefig(dir + '/N_5/CDF_comp_mf_%d.eps'%j)

# plt.show()