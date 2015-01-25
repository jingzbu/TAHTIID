#!/usr/bin/env python
"""
This file contains all the threshold calculation methods
"""
from __future__ import absolute_import, division

__author__ = "Jing Zhang"
__email__ = "jingzbu@gmail.com"
__status__ = "Development"


import numpy as np
from numpy import linalg as LA
from math import log
from matplotlib.mlab import prctile

from TAHTIID.util.util import mu_mf,  sample_path_mf, H_est_mf, Sigma_est_mf, W_est, KL_est_mf, HoeffdingRuleIID


class ThresBase(object):
    def __init__(self, N, beta, n, mu_0, mu_1, H_1, Sigma_1, W_1):
        self.N = N  # N is the row dimension of the original transition matrix Q
        self.beta = beta  # beta is the false alarm rate
        self.n = n  # n is the number of samples
        self.mu_0 = mu_0
        self.mu_1 = mu_1
        self.H_1 = H_1
        self.Sigma_1 = Sigma_1
        self.W_1 = W_1

class ThresActual(ThresBase):
    """ Computing the actual (theoretical) K-L divergence and threshold
    """
    def ThresCal(self):
        SampNum = 5000
        self.KL = []
        for i in range(0, SampNum):
            # print(self.mu_1[0][0], self.n)
            x = sample_path_mf(self.mu_1[0], self.n)
            # print(x)
            # assert(1 == 2)
            self.KL.append(KL_est_mf(x, self.mu_1[0]))  # Get the actual relative entropy (K-L divergence)
        self.eta = prctile(self.KL, 100 * (1 - self.beta))
        KL = self.KL
        eta = self.eta
        return KL, eta

class ThresWeakConv(ThresBase):
    """ Estimating the K-L divergence and threshold by use of weak convergence
    """
    def ThresCal(self):
        self.KL, self.eta1, self.eta2 = HoeffdingRuleIID(self.beta, self.H_1, self.W_1, self.n)
        KL = self.KL
        eta1 = self.eta1
        eta2 = self.eta2
        return KL, eta1, eta2

class ThresSanov(ThresBase):
    """ Estimating the threshold by use of Sanov's theorem
    """
    def ThresCal(self):
        self.eta = - log(self.beta) / self.n
        eta = self.eta
        return eta