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
    densities = []

    # iterate over income groups
    n_rows = income_data.shape[0]
    for i in range(n_rows):

        # get number of respondents in group
        size = income_data.freq.iloc[i]
        
        # get lower and upper bounds for income group
        lower = income_data.income_lower.iloc[i]
        upper = income_data.income_upper.iloc[i]
        
        # evenly distribute incomes for each respondent across the group
        incomes.append(np.linspace(lower, upper, size))
        
        # get bounds for density of group
        if i == 0:
            prev = 0 
            curr = income_data.density[i]
        else:
            # prev determined by density of previous group
            prev = income_data.density[i-1]

            # curr determined by density of current group
            curr = income_data.density[i]
        
        # evenly distribute densities across group density    
        densities.append(np.linspace(prev, curr, size))

    # create DataFrame by concatenating arrays of each group's interpolated data
    df_income = pd.DataFrame({'income': np.concatenate(incomes), 
                              'density': np.concatenate(densities)})

    return df_income


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

    std_moment - kth_moment / sd**k

    return std_moment


