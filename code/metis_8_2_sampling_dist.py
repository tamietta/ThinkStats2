# *************************************
#            Exercise 8.2
# *************************************

'''
Suppose you draw a sample with size n = 10 from an exponential distribution with λ = 2. Simulate this experiment 1000 times and plot the sampling distribution of the estimate L. Compute the standard error of the estimate and the 90% confidence interval.

Repeat the experiment with a few different values of n and make a plot of standard error versus n.

'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import bisect
from itertools import chain

sns.set_style('whitegrid', rc={'grid.linestyle': ':'}) 

# Uncomment for Jupyter Notebook
# %matplotlib inline

class CDF(object):
    def __init__(self, data):
        self.cdf = self._create_cdf(data)

    def _create_cdf(self, data):
        cdf = pd.DataFrame(pd.Series(data).value_counts().sort_index())
        cdf.rename(columns={cdf.columns[0]: 'freq'}, inplace=True)

        cdf['cum_freq'] = cdf.freq.cumsum()
        
        total_freq = cdf.cum_freq.iloc[-1]
        cdf['prob'] = cdf.cum_freq / total_freq

        return cdf

    def get_values(self):
        return self.cdf.index.values

    def get_probs(self):
        return self.cdf.prob.values

    def percentile(self, rank):
        rank /= 100
        index = bisect.bisect_left(self.cdf.prob.values, rank)

        return self.cdf.index[index]

    def ci(self, lower_rank, upper_rank):
        ci_lower = self.percentile(lower_rank)
        ci_upper = self.percentile(upper_rank)

        return ci_lower, ci_upper

    def plot_cdf(self, title, xlabel):
        fig, ax = plt.subplots()

        vals = self.get_values()
        probs = self.get_probs()

        ax.plot(vals, probs)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Cumulative Probability')

        ci_5, ci_95 = self.ci(5, 95)

        ax.vlines(x=[ci_5, ci_95], ymin=0, ymax=1, 
                    linestyles='dotted', label='90% Confidence Interval')
        ax.legend()


def estimate_lambda(lam=2, n_samples=10, n_trials=1000):
    lambdas = np.zeros(n_trials)

    for i in range(n_trials):
        samples = np.random.exponential(scale=1/lam, size=n_samples)
        lambdas[i] = 1/samples.mean()

    return lambdas


def mse(data, parameter):
    data = np.array(data)
    data -= parameter
    mse = data.dot(data) / data.size

    return mse


def se_experiment(samples_sizes, lam=2, n_trials=1000):
    data = {}

    for n in samples_sizes:
        lambdas = estimate_lambda(n_samples=n, lam=lam, n_trials=n_trials)
        data[n] = np.sqrt(mse(lambdas, lam))

    return data


def plot_sampling_dist(data, bins, title, xlabel, ylabel):
    fig, ax = plt.subplots()

    ax.hist(data, bins=bins)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def plot(x, y, label, title, xlabel, ylabel):
    fig, ax = plt.subplots()

    ax.plot(x, y, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()



if __name__ == '__main__':
    
    # Estimates for L with lambda = 2, n = 10, trials = 1000

    lambdas = estimate_lambda()
    cdf_lambdas = CDF(lambdas)

    plot_sampling_dist(lambdas, bins=100,
                       title='Sampling Distribution of Estimates for '
                              r"$ L = \frac{1}{\bar{x}}$", 
                       xlabel='Estimate for L',
                       ylabel='Frequency')

    cdf_lambdas.plot_cdf(title='CDF for Sampling Distribution of Estimates for L',
                     xlabel='Estimates for L')

    se = np.sqrt(mse(lambdas, 2))
    print('Standard error of the estimate for L: {:.3f}'.format(se))

    ci = cdf_lambdas.ci(5, 95)
    print('90% confidence interval for the estimate for L$: [{0[0]:.3f}, {0[1]:.3f}]'.format(ci))

    
    # Trends for SE with different sample sizes, lambda = 2, trials = 1000

    sample_sizes = chain(range(100, 1001, 100), range(1000, 5001, 1000), [10000])
    data = se_experiment(sample_sizes)

    plot(x=list(data.keys()), y=list(data.values()), 
         label='Standard Error',
         title='SE of Sampling Distributions of Different Sample Sizes', 
         xlabel='Sample Size',
         ylabel='Standard Error')

    plt.show()

