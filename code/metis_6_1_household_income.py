# *************************************
#            Exercise 6.1
# *************************************

'''
The distribution of income is famously skewed to the right. In this exercise, we’ll measure how strong that skew is. The Current Population Survey (CPS) is a joint effort of the Bureau of Labor Statistics and the Census Bureau to study income and related variables. Data collected in 2013 is available from http://www.census.gov/hhes/www/cpstables/032013/hhinc/toc.htm. I downloaded hinc06.xls, which is an Excel spreadsheet with information about household income, and converted it to hinc06.csv, a CSV file you will find in the repository for this book. You will also find hinc2.py, which reads this file and transforms the data.

The dataset is in the form of a series of income ranges and the number of respondents who fell in each range. The lowest range includes respondents who reported annual household income “Under $5000.” The highest range includes respondents who made “$250,000 or more.”

To estimate mean and other statistics from these data, we have to make some assumptions about the lower and upper bounds, and how the values are distributed in each range. hinc2.py provides InterpolateSample, which shows one way to model this data. It takes a DataFrame with a column, income, that contains the upper bound of each range, and freq, which contains the number of respondents in each frame.

It also takes log_upper, which is an assumed upper bound on the highest range, expressed in log10 dollars. The default value, log_upper=6.0 represents the assumption that the largest income among the respondents is  106106 , or one million dollars.

InterpolateSample generates a pseudo-sample; that is, a sample of household incomes that yields the same number of respondents in each range as the actual data. It assumes that incomes in each range are equally spaced on a log10 scale.

Compute the median, mean, skewness and Pearson’s skewness of the resulting sample. What fraction of households reports a taxable income below the mean? How do the results depend on the assumed upper bound?
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hinc import ReadData
from metis_8_2_sampling_dist import CDF, plot_line

def income_data():
    '''
    Returns DataFrame of income data with the following columns:
    [income_lower', 'income_upper', 'freq', 'cum_freq', 'width', 'density', 'prob', 'cum_prob']
    '''
    # get DataFrame of income data
    # cols = 'income' (upper bound), 'freq', 'cumsum' (cumulative freq), 'ps' (probability)
    df = ReadData()

    df.rename(columns={'income': 'income_upper',
                       'cumsum': 'cum_freq'}, 
              inplace=True)
    
    # increment all $ XX99 upper bounds by 1
    df.income_upper += 1

    # create income lower bound column by shifting upper bound values by 1 index position
    df.loc[:, 'income_lower'] = df.income_upper.shift(1)
    df.loc[0, 'income_lower'] = 0 
    
    # set upper-most bound for income
    last_row = df.index[-1]
    df.loc[last_row, 'income_upper'] = 1.0e+06  # 1 million
    
    # add column for income range width
    df.loc[:, 'width'] = df.income_upper - df.income_lower
    
    # add column for probability density (i.e. probability = density * width)
    total_freq = df.cum_freq.iloc[-1]
    df.loc[:, 'density'] = df.freq / (total_freq * df.width)  # equivalent: rel_freq/width
    
    # add column for probability
    df.loc[:, 'prob'] = df.density * df.width
    
    # add column for cumulative probability
    df.loc[:, 'cum_prob'] = df.prob.cumsum()

    # subset and order desired columns
    df = df[['income_lower', 'income_upper', 'freq', 'cum_freq', 'width', 'density', 'prob', 'cum_prob']]

    return df


def interpolate_data(income_data):
    '''
    Interpolates individual incomes and associated densities from grouped incomes.
    The interpolation assumes uniform distribution of frequencies within each
    income and density range.
    '''
    incomes = []

    # iterate over income groups
    n_rows = income_data.shape[0]
    for i in range(n_rows):
        # get number of respondents for income group
        freq = income_data.freq[i]

        # evenly divide income width
        step = income_data.width[i]/income_data.freq[i]
        
        # get lower bound for income group
        lower = income_data.income_lower.iloc[i]
        
        # evenly distribute incomes for each respondent across the group
        interpolated = lower + (step * np.arange(1, freq+1))
        incomes.append(interpolated)

    # concatenating interpolated-data arrays into one array
    incomes = np.concatenate(incomes)

    return incomes


def moment_about_zero(data, k):
    '''
    Defined as the mean of the sum of each random variable raised to the k-th power.
    The first moment about zero is the MLE mean.

    Calculates the k-th moment about zero (also known as 'raw moment')
    '''
    data = np.array(data)
    moment = np.sum(data**k) / data.size

    return moment


def moment_about_mean(data, k):
    '''
    Defined as the mean k-th error.
    The second moment about the mean is the MSE or the (biased) variance.

    Calculates the k-th moment about the mean (also known as 'central moment')
    '''
    data = np.array(data)

    mu = data.mean()
    error = data - mu
    moment = np.sum(error**k) / data.size

    return moment


def standardised_moment(data, k):
    '''
    Defined as the k-th moment about the mean divided by the k-th standard deviation
    The standardised moment is unitless and independent of differences in data scales.

    Calculates the k-th standardised moment.
    '''
    data = np.array(data)

    kth_moment = moment_about_mean(data, k)
    sd = data.std(ddof=0) # biased standard deviation

    std_moment = kth_moment / sd**k

    return std_moment


def sample_skewness(data):
    '''
    Returns the sample skewness (g1) defined as the 3rd standardised moment.
    '''
    g1 = standardised_moment(data, 3)    
    
    return g1


def pearson_median_skewness(data, mean=None, median=None):
    '''
    Returns the Pearson median skewness (gp) defines as 3*(median)
    '''
    cdf = CDF(data)
   
    median = cdf.percentile(50)
    mean = moment_about_zero(data, 1)
    std = np.sqrt(moment_about_mean(data, 2))
    
    gp = 3*(mean - median)/std

    return gp


if __name__ == '__main__':
    df = income_data()
    incomes = interpolate_data(df)
    cdf_incomes = CDF(incomes)


    plot_line(df.income_upper, df.density, 
              label=None, 
              title='Approximaated PDF of U.S. Household Incomes', 
              xlabel='Income ($)', 
              ylabel='Probability Density')

    plot_line(cdf_incomes.get_values(), cdf_incomes.get_probs(), 
              label=None,
              title='Approximated CDF of U.S. Household Incomes', 
              xlabel='Income ($)', 
              ylabel='Cumulative Probability')
    
    mean = incomes.mean()
    median = cdf_incomes.percentile(50)
    sample_skew = sample_skewness(incomes)
    pearson_skew = pearson_median_skewness(incomes)
    mean_rank = cdf_incomes.rank(mean)

    print('Mean income: {}'.format(mean))
    print('Median income: {}'.format(median))
    print('Sample skewness: {}'.format(sample_skew))
    print('Pearson\'s median skewness: {}'.format(pearson_skew))
    print('Percentage of incomes below the mean: {}'.format(mean_rank))

