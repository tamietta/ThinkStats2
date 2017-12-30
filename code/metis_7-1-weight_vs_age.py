# *************************************
#            Exercise 7.1 
# *************************************

'''
Using data from the NSFG, make a scatter plot of birth weight versus mother’s age. 

Plot percentiles of birth weight versus mother’s age. 

Compute Pearson’s and Spearman’s correlations. 

How would you characterize the relationship between these variables?

'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import first

sns.set_style('whitegrid', rc={'grid.linestyle': ':'})

# Uncomment for Jupyter Notebook
# %matplotlib inline


def percentile(data, rank):
    '''
    INPUT: data, numpy 1D array
           rank, integer (0-100)

    OUTPUT: percentile value

    Returns the value in the data set corresponding to the percentile rank.
    '''
    data.sort() 

    n = data.size - 1
    rank /= 100
    index = int(rank * n)

    return data[index]


def plot_scatter(x, y, title, xlabel, ylabel, alpha):
    '''
    INPUT: x, y, 1D array-like
           title, xlabel, ylabel, string
           alpha, float

    OUTPUT: None

    Plots a scatter graph of x-values against y-values.
    '''
    fig, ax = plt.subplots()

    ax.scatter(x, y, alpha=alpha)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def plot_percentiles(x, percentiles, labels, title, xlabel, ylabel):
    '''
    INPUT: x, y, 1D array-like
           percentiles, list of 1D array-likes
           labels, list of strings
           title, xlabel, ylabel, string

    OUTPUT: None

    Plots a line graph of x-values against given percentiles of y-values.
    '''
    fig, ax = plt.subplots()

    for p, label in zip(percentiles, labels):
        ax.plot(x, p, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()


def pearson_corr(x, y):
    '''
    INPUT: x, y, numpy 1D arrays

    OUTPUT: corr, float between [-1, 1]

    Returns the Pearson product-moment correlation between x and y.
    Uses the unbiased estimator for variance.
    '''
    x_mu = x.mean()
    y_mu = y.mean()
    x_var = x.var(ddof=1)
    y_var = y.var(ddof=1)

    x_dev = x - x_mu
    y_dev = y - y_mu

    cov = x_dev.dot(y_dev) / (x_dev.size - 1)
    corr = cov/np.sqrt(x_var*y_var)

    return corr


def pearson_corr_cov(x, y):
    '''
    INPUT: x, y, numpy 1D arrays

    OUTPUT: corr, float between [-1, 1]

    Returns the Pearson product-moment correlation between x and y.
    Uses the unbiased estimator for variance.
    Algorithm utilises the covariance matrix.
    Faster performance than pearson_corr function.
    '''
    X = np.stack((x, y), axis=1) # matrix of example rows and variable columns
    
    X_means = X.mean(0) # calculate means for each variable column
    X_dev = X - X_means # mean-normalisation

    cov_M = X_dev.T.dot(X_dev) / (x.size - 1) # create covariance matrix

    cov = cov_M[0, 1] # covariance of x and y
    var_xy = cov_M.diagonal() # variances of x and y along diagonal
    corr = cov/np.sqrt(np.prod(var_xy))

    return corr


def spearman_rank(data):
    '''
    INPUT: data, 1D array-like

    OUTPUT: rank, list

    Returns a list of the corresponding ranks of each element in 'data'.
    '''
    data_sort = np.sort(data)
    rank_dict = {k: v for v, k in enumerate(data_sort)}

    rank = np.array([rank_dict[d] for d in data])

    return rank



def spearman_corr(x, y):
    '''
    INPUT: x, y, 1D array-like

    OUTPUT: rank_corr, float between [-1, 1]

    Returns a list of the corresponding ranks of each element in 'data'.
    '''
    x_rank = spearman_rank(x)
    y_rank = spearman_rank(y)

    rank_corr = pearson_corr(x_rank, y_rank)

    return rank_corr



if __name__ == '__main__':
    
    # Read in and filter relevant data

    live, firsts, others = first.MakeFrames()

    data = live.loc[:, ['agepreg', 'totalwgt_lb']]
    data.dropna(inplace=True)


    # Compute birth weight percentiles for each age group

    lower = int(data.agepreg.min())        # first bin lower bound
    upper = int(data.agepreg.max()) + 4    # last bin upper bound
    bins = np.arange(lower, upper, 3)
    indices = np.digitize(data.agepreg, bins)

    groups = data.groupby(indices)

    years = []
    wgt_25 = []
    wgt_50 = []
    wgt_75 = []

    for i, group in groups:
        years.append(group.agepreg.mean())
        wgt_25.append(percentile(group.totalwgt_lb.values, 25))
        wgt_50.append(percentile(group.totalwgt_lb.values, 50))
        wgt_75.append(percentile(group.totalwgt_lb.values, 75))


    # Plot scatter graph of mother's age against birth weight

    plot_scatter(data.agepreg, data.totalwgt_lb,
                 "Mother's Age against Baby's Birth Weight",
                 "Mother's Age",
                 "Birth Weight (lb)",
                 0.08)

    
    # Plot scatter graph of mother's age against birth weight

    plot_percentiles(years, 
                     [wgt_25, wgt_50, wgt_75], 
                     ['25th', '50th', '75th'],
                     "Mother's Age against Birth Weight Percentiles",
                     "Mother's Age",
                     "Birth Weight (lb)")

    # Compute Pearson's product-moment correlation coefficient

    corr = pearson_corr(data.agepreg, data.totalwgt_lb)
    print('Mother\'s age to birth weight correlation: {:f}'.format(corr))

    # Compute Spearman's rank correlation

    rank_corr = spearman_corr(data.agepreg, data.totalwgt_lb)
    print('Mother\'s age to birth weight rank correlation: {:f}'.format(rank_corr))

    plt.show()
