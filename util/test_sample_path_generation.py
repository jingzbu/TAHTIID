#!/usr/bin/env python
""" A library of utility functions that will be used by Simulator
"""
from __future__ import absolute_import, division

__author__ = "Jing Zhang"
__email__ = "jingzbu@gmail.com"
__status__ = "Development"

import argparse
import numpy as np
from numpy import linalg as LA
from math import sqrt, log
from scipy.stats import chi2
from matplotlib.mlab import prctile
import matplotlib.pyplot as plt
import pylab
from pylab import *

import statsmodels.api as sm


def rand_x(p):
    """
    Generate a random variable in 0, 1, ..., (N-1) given a distribution vector p
    N is the dimension of p
    ----------------
    Example
    ----------------
    p = [0.5, 0.5]
    x = []
    for j in range(0, 20):
        x.append(rand_x(p))
    print x
    ----------------
    [1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0]
    """
    p = np.array(p)
    N = p.shape
    #assert(abs(sum(p) - 1) < 1e-5)
    u = np.random.rand(1, 1)
    i = 0
    s = p[0]
    while (u > s).all() and i < N[0]-1:
        i = i + 1
        s = s + p[i]
    index = i

    return index

def sample_path_mf(mu, n):
    """
    Simulate a sample path on {0, 1, ..., n-1} given a distribution mu
    ----------------
    The program assumes that the values of the i.i.d. random variables are labeled 0, 1, ..., N-1
    ----------------
    mu: the distribution, a 1 x N vector
    ----------------
    n: the number of samples
    N: the total number of value labels of the i.i.d. random variables
    ----------------
    Example
    ----------------
    """
    #assert(abs(sum(mu_0) - 1) < 1e-5)
    x = []
    for i in range(0, n):
        t = rand_x(mu)
        x.append(t)

    return x

def mu_mf(dim_mu):
    """
    Randomly generate the distribution mu
    dim_mu: the dimension of mu; equals N
    Example
    ----------------
    """
    weight = []
    for i in range(0, dim_mu):
        t = np.random.randint(1, high=dim_mu, size=1)
        weight.append(t)
    mu = np.random.rand(1, dim_mu)
    for i in range(0, dim_mu):
        mu[0, i] = (weight[i][0]) * mu[0, i]
    mu = mu / (sum(mu[0, :]))
    assert(abs(sum(mu[0, :]) - 1) < 1e-5)

    return mu

def mu_est_mf(x, N):
    """
    Estimate the distribution mu
    x: a sample path of the i.i.d. sequence
    N: the obtained mu should be a 1 x N vector
    Example
    ----------------
    >>> x = [0, 1, 2, 2, 1, 2, 1,0, 1, 2, 2, 1, 2, 1, 0, 1, 2, 2, 1, 2, 1, 3, 3, 3, 3, 1, 1]
    >>> N = 4
    >>> mu_1 = mu_est_mf(x, N)
    >>> print(mu_1)
    [ 0.11111111  0.40740741  0.33333333  0.14814815]
    """
    eps = 1e-8
    gama = []
    for j in range(0, N):
        t = (x.count(j)) / (len(x))
        if t < eps:
            t = eps
        gama.append(t)
    for j in range(0, N):
        gama[j] = gama[j] / sum(gama)  # Normalize the estimated probability law
    gama = np.array(gama)
    mu = gama

    return mu

def KL_est_mf(x, mu):
    """
    Estimate the relative entropy (K-L divergence)
    x: a sample path of the i.i.d. sequence
    mu: the true distribution; a 1 x N vector
    """
    mu = np.array(mu)
    N = len(mu)

    # Compute the empirical distribution
    gama = mu_est_mf(x, N)

    # Compute the relative entropy (K-L divergence)
    d = []
    for i in range(0, N):
        t = gama[i] * (log(gama[i] - log(mu[i])))
        d.append(t)
    KL = sum(sum(d))

    return KL

def H_est_mf(mu):
    """
    Estimate the Hessian
    Example
    ----------------
    >>> mu = [0.5, 0.25, 0.25]
    >>> print(H_est_mf(mu))
    [[ 2.  0.  0.]
     [ 0.  4.  0.]
     [ 0.  0.  4.]]
    """
    mu = np.array(mu)
    H = np.diag(1.0 / mu)
    # N = len(mu)
    # H = np.zeros((N, N))
    #
    # for i in range(0, N):
    #     for j in range(0, N):
    #         if i == j:
    #             H[i, i] = 1.0 / mu[i]
    #         else:
    #             H[i, j] = 0
    return H

def Sigma_est_mf(mu):
    """
    Estimate the covariance matrix of the empirical measure
    Example
    ----------------
    >>> mu = np.array([[0.5, 0.25, 0.25]])
    >>> print(Sigma_est_mf(mu))
    [[ 0.25   -0.125  -0.125 ]
     [-0.125   0.1875 -0.0625]
     [-0.125  -0.0625  0.1875]]
    """
    Sigma = np.diag(mu[0]) - np.dot(np.transpose(mu), mu)

    # Ensure Sigma to be symmetric
    Sigma = (1.0 / 2) * (Sigma + np.transpose(Sigma))

    # Ensure Sigma to be positive semi-definite
    D, V = LA.eig(Sigma)
    D = np.diag(D)
    Q, R = LA.qr(V)
    N = len(mu[0])
    for i in range(0, N):
        if D[i, i] < 0:
            D[i, i] = 0
    Sigma = np.dot(np.dot(Q, D), LA.inv(Q))

    return Sigma

def W_est(Sigma, SampNum):
    """
    Generate samples of W
    ----------------
    Sigma: the covariance matrix; an N x N matrix
    SampNum: the length of the sample path
    """
    N, _ = Sigma.shape
    assert(N == _)
    W_mean = np.zeros((1, N))
    W = np.random.multivariate_normal(W_mean[0, :], Sigma, (1, SampNum))

    return W

def HoeffdingRuleIID(beta, H, W, FlowNum):
    """
    Estimate the K-L divergence and the threshold by use of weak convergence
    ----------------
    beta: the false alarm rate
    H: the Hessian
    W: a sample path of the Gaussian empirical measure
    FlowNum: the number of flows
    ----------------
    """
    _, SampNum, N = W.shape

    # Estimate K-L divergence using 2nd-order Taylor expansion
    KL = []
    for j in range(0, SampNum):
        tt = (1.0 / 2) * (1.0 / FlowNum) * np.dot(W[0, j, :], H)
        t = np.dot(tt, W[0, j, :])
        # print t.tolist()
        # break
        KL.append(t)
    # Get the threshold
    eta1 = prctile(KL, 100 * (1 - beta))

    # Using the simplified formula
    eta2 = 1.0 / (2 * FlowNum) * chi2.ppf(1 - beta, N - 1)

    # print(KL)
    # assert(1 == 2)
    return KL, eta1, eta2

def SamplePathGen(N):
    # Generate the actual distribution (PL) mu_0
    mu_0 = mu_mf(N)[0]

    # Generate a sample path of the i.i.d. sequence with length n_1; this path is used to estimate the actual PL
    n_1 = 10 * N  # the length of a sample path
    x_1 = sample_path_mf(mu_0, n_1)

    # Compute the estimated PL mu_1
    mu_1 = mu_est_mf(x_1, N)

    # Compute the estimate of the Hessian
    H_1 = H_est_mf(mu_1)

    # Compute the estimate of the covariance matrix
    mu_1 = np.array(mu_1, ndmin=2)
    Sigma_1 = Sigma_est_mf(mu_1)

    # Get an estimated sample path of W
    SampNum = 5
    W_1 = W_est(Sigma_1, SampNum)

    return mu_0, x_1, mu_1, H_1, Sigma_1, W_1


mu_0, x_1, mu_1, H_1, Sigma_1, W_1 = SamplePathGen(4)

print mu_0, x_1, mu_1, H_1, Sigma_1, W_1

# mu = [0.5, 0.25, 0.25]
# H = H_est_mf(mu)
# print H
#
# mu = np.array([[0.5, 0.25, 0.25]])
# print(mu)
#
# print(mu[0])
#
# mu = np.array(mu)
# print(np.dot(np.transpose(mu), mu))
#
# Sigma = Sigma_est_mf(mu)
# print Sigma
#
# SampNum = 10
# W = W_est(Sigma, SampNum)
# print W
# print W.shape
#
#
# # a = np.dot(W[0, 1, :]), H, np.transpose(np.dot(W[0, 1, :]))
# a = np.dot(W[0, 1, :], H)
# b = np.dot(a, W[0, 1, :])
# print a, b

# KL, eta1, eta2 = HoeffdingRuleIID(0.001, H, W, 100)
#
# print KL, eta1, eta2


# df = 4
# KL_1 = chi2.rvs(df, size=5000)
#
# ecdf_1 = sm.distributions.ECDF(KL_1)
# x_1 = np.linspace(min(KL_1), max(KL_1), num=100)
# y_1 = ecdf_1(x_1)
# KL_actual, = plt.plot(x_1, y_1, 'r')
# pylab.ylim(0, 1.01)
# plt.show()



# n = 10
# x = sample_path_mf(mu, n)
# print x
#
# N = 3
# mu_1 = mu_est_mf(x, N)
# print(mu_1)
#
# KL = KL_est_mf(x, mu)
# print KL


# mu = mu_mf(4)[0]
# print mu
#
#
# x = sample_path_mf(mu, 2000)
# print x
#
# gama = mu_est_mf(x, 4)
# print gama