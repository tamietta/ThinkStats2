# *************************************
#            Exercise 2.4 
# *************************************

'''
Using the variable totalwgt_lb, investigate whether first babies are lighter or heavier than others. 
Compute Cohen’s d to quantify the difference between the groups. 
How does it compare to the difference in pregnancy length?

'''

import numpy as np
import pandas as pd

import first

live, first, others = first.MakeFrames()

def cohens_d(data1, data2):
    '''
    INPUT: data1, numpy 1D array/pandas Series
           data2, numpy 1D array/pandas Series

    OUTPUT: cohens_d, float

    Computes Cohen's d for the two groups.
    Cohen's d is a test statistic measuring the difference in means as a 
    proportion of the pooled standard deviation.
    '''
    mean1 = data1.mean()
    mean2 = data2.mean()

    var1 = data1.var(ddof=1)
    var2 = data2.var(ddof=1)

    pooled_var = (var1 * data1.size + var2 * data2.size)/(data1.size + data2.size)

    cohens_d = abs(mean1 - mean2)/np.sqrt(pooled_var)

    return cohens_d

# Cohen's d for Pregnancy Length

preg_length = cohens_d(first.prglngth, others.prglngth)

# Cohen's d for Total Weight

total_weight = cohens_d(first.totalwgt_lb, others.totalwgt_lb)
