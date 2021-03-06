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

# Uncomment for Jupyter Notebook.
# %matplotlib inline

class CDF(object):
    '''
    Converts and stores data in CDF-relevant format of sorted values, frequency,
    cumulative frequency and cumulative probability.

    The CDF class approximates a continuous CDF from discrete data.
    '''
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
        return self.cdf.prob

    def mean(self):
        vals = self.cdf.index.values
        counts = self.cdf.freq.values
        size = self.cdf.cum_freq.iloc[-1]

        return vals.dot(counts) / size

    def percentile(self, rank):
        '''
        Algorithm *interpolates* the percentile value of the given rank.
        '''
        rank /= 100
        index = bisect.bisect_left(self.cdf.prob.values, rank)

        lower_rank = self.cdf.prob.iloc[index]
        upper_rank = self.cdf.prob.iloc[index+1]

        lower_val = self.cdf.index[index]
        upper_val = self.cdf.index[index+1]

        prop = (rank - lower_rank)/(upper_rank - lower_rank)

        return lower_val + (prop * (upper_val - lower_val))

    def rank(self, value):
        '''
        Algorithm *interpolates* the percentile rank of the given value.
        '''
        index = bisect.bisect_left(self.cdf.index.values, value)

        lower_val = self.cdf.index[index]
        upper_val = self.cdf.index[index+1]

        lower_rank = self.cdf.prob.iloc[index]
        upper_rank = self.cdf.prob.iloc[index+1]

        prop = (value - lower_val)/(upper_val - lower_val)

        return lower_rank + (prop * (upper_rank - lower_rank))

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
    '''
    Creates sampling distribution data for estimates of exponential
    distribution rate parameter, lambda.
    '''
    lambdas = np.zeros(n_trials)

    for i in range(n_trials):
        samples = np.random.exponential(scale=1/lam, size=n_samples)
        lambdas[i] = 1/samples.mean()

    return lambdas


def mse(estimates, parameter):
    '''
    Calculates mean squared error
    '''
    estimates = np.array(estimates)
    errors = estimates- parameter
    mse = errors.dot(errors) / errors.size

    return mse


def se_experiment(samples_sizes, lam=2, n_trials=1000):
    '''
    Returns the standard error of sampling distributions of different
    sample sizes.
    '''
    data = {}

    for n in samples_sizes:
        lambdas = estimate_lambda(n_samples=n, lam=lam, n_trials=n_trials)
        data[n] = np.sqrt(mse(lambdas, lam))

    return data


def plot_sampling_dist(data, bins, title, xlabel, ylabel):
    '''
    Plots a normalised histogram from given data.
    '''
    fig, ax = plt.subplots()

    ax.hist(data, bins=bins, normed=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def plot_line(x, y, title, xlabel, ylabel, label=None):
    '''
    Plots a line graph of the relation between x and y.
    '''
    fig, ax = plt.subplots()

    ax.plot(x, y, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if label:
        ax.legend()



if __name__ == '__main__':
    
    # create sampling distribution for lambda estimates
    # lambda=2, n_samples=10, n_trials=1000
    lambdas = estimate_lambda

    # create CDF object from sampling distribution data
    cdf_lambdas = CDF(lambdas)

    # plot normalised histogram.
    plot_sampling_dist(lambdas, bins=100,
                       title='Sampling Distribution of Estimates for '
                              r"$ L = \frac{1}{\bar{x}}$", 
                       xlabel='Estimate for L',
                       ylabel='Density')

    # plot CDF graph
    cdf_lambdas.plot_cdf(title='CDF for Sampling Distribution of Estimates for L',
                         xlabel='Estimates for L')

    se = np.sqrt(mse(lambdas, 2))
    print('Standard error of the estimate for L: {:.3f}'.format(se))

    ci = cdf_lambdas.ci(5, 95)
    print('90% confidence interval for the estimate for L$: [{0[0]:.3f}, {0[1]:.3f}]'.format(ci))

    # set range of sample sizes
    sample_sizes = chain(range(100, 1001, 100), range(1000, 5001, 1000), [10000])

    # run simulation trials
    data = se_experiment(sample_sizes)

    # Plot SE for different sample sizes
    plot_line(x=list(data.keys()), y=list(data.values()), 
              label='Standard Error',
              title='SE of Sampling Distributions of Different Sample Sizes', 
              xlabel='Sample Size',
              ylabel='Standard Error')

    plt.show()

