import stats
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis


# get an array of normally sampled data
mu, sigma = 0, 1
scale = 30
#s = np.random.standard_cauchy(mu, sigma, 1000)
#s = np.random.standard_cauchy(1000)
s = np.random.weibull(1.5, 1000)
#s = s[(s>-10) & (s<10)]
#s = np.random.laplace(mu, sigma, 1000)
#s = np.random.normal(mu, sigma, 1000)
scipy.io.savemat('/home/tabitha/Desktop/laplace.mat', mdict={'arr': s})

# verify the mean and variance ?
print(abs(mu - np.mean(s)) < 0.01)
print(abs(sigma - np.std(s, ddof=1)) < 0.01)

count, bins = np.histogram(s, 300, density=False)
plt.bar(bins[:-1], count, align='edge')
print('skew: ', stats.my_skew(s), 'kurtosis: ', stats.my_kurtosis(s))
print('Scipy: skew: ', skew(count), 'kurtosis: ', kurtosis(count))

plt.pause(10)
