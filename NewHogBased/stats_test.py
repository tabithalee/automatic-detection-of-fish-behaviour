import stats
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis


# get an array of normally sampled data
mu, sigma = 0, 1
scale = 30
#s = np.random.standard_cauchy(mu, sigma, 1000000)
s = np.random.weibull(1.5, 1000000)
#s = s[(s>-10) & (s<10)]

# verify the mean and variance ?
print(abs(mu - np.mean(s)) < 0.01)
print(abs(sigma - np.std(s, ddof=1)) < 0.01)

count, bins = np.histogram(s, 300, density=False)
plt.bar(bins[:-1], count, align='edge')
print('skew: ', stats.my_skew(count), 'kurtosis: ', stats.my_kurtosis(count))
print('Scipy: skew: ', skew(count), 'kurtosis: ', kurtosis(count))

plt.pause(10)
