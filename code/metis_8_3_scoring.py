# *************************************
#            Exercise 8.3
# *************************************

'''
In games like hockey and soccer, the time between goals is roughly exponential. So you could estimate a team’s goal-scoring rate by observing the number of goals they score in a game. This estimation process is a little different from sampling the time between goals, so let’s see how it works.

Write a function that takes a goal-scoring rate, lam, in goals per game, and simulates a game by generating the time between goals until the total time exceeds 1 game, then returns the number of goals scored.

Write another function that simulates many games, stores the estimates of lam, then computes their mean error and RMSE.

Is this way of making an estimate biased? Plot the sampling distribution of the estimates and the 90% confidence interval. What is the standard error? What happens to sampling error for increasing values of lam?

'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from metis_8_2_sampling_dist import CDF, plot_sampling_dist

sns.set_style('whitegrid', rc={'grid.linestyle': ':'})

def simulate_game(lam):
    goals = 0
    num_games = 0

    while num_games < 1:
        next_goal = np.random.exponential(scale=1/lam, size=1) # in units of games (e.g. 0.5 = 1/2 game)

        num_games += next_goal
        goals += 1

    return goals - 1 # to account for going over 1 game


def simulate_multiple_games(lam, n_games):
    lambdas = np.zeros(n_games)
    
    for i in range(n_games):
        lambdas[i] = simulate_game(lam)

    return lambdas


def mean_error(estimates, parameter):
    estimates = np.array(estimates)
    errors = estimates - parameter
    mean_err = errors.mean()

    return mean_err


def mse(estimates, parameter):
    estimates = np.array(estimates)
    errors = estimates - parameter
    mse = errors.dot(errors) / estimates.size

    return mse


if __name__ == '__main__':
    lam = 4
    n_games = 10000

    num_goals = simulate_game(lam)

    lambdas = simulate_multiple_games(lam, n_games)
    mean_err = mean_error(lambdas, lam)
    root_mse = np.sqrt(mse(lambdas, lam))

    cdf = CDF(lambdas)
    cdf.plot_cdf(title='CDF of Lambda Estimates from Simulation',
                 xlabel='Lambda Estimates')

    plot_sampling_dist(lambdas, bins=15,
                       title='Sampling Distribution of Lambda Estimates',
                       xlabel='Lambda Estimates',
                       ylabel='Frequency')

    ci = cdf.ci(5, 95)

    print('Actual lambda: {} goals/game'.format(lam))
    print('Goals per game from one simulation: {}'.format(num_goals))
    print('Mean of lambda estimates: {}'.format(lambdas.mean()))
    print('\nNumber of simulated games: {}'.format(n_games))
    print('Mean error: {}'.format(mean_err))
    print('Standard error: {}'.format(root_mse))
    print('90% confidence interval: ({0[0]:.3f}, {0[1]:.3f})'.format(ci))

    plt.show()



