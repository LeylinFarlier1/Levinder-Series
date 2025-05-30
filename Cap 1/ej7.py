# 7. Suppose X is distributed normally with a mean of 5 and a standard deviation
 #of 2. What is the expected value of 10X? What is the expected value of 20X?
# What are the variance and standard deviations of 5X and of 10X?

import numpy as np

X = np.random.normal(loc=5, scale=2, size=1000)
mean_X = X.mean()
std_dev_X = X.std()
var_X = X.var()
a1 = 10
a2 = 20
mean_10X = a1 * mean_X
mean_20X = a2 * mean_X
var_5X = (5 ** 2) * var_X
var_10X = (a1 ** 2) * var_X
std_dev_5X = np.sqrt(var_5X)
std_dev_10X = np.sqrt(var_10X)
print(f"Mean of X: {mean_X}, Standard Deviation of X: {std_dev_X}, Variance of X: {var_X}")
print(f"Expected value of 10X: {mean_10X}")
print(f"Expected value of 20X: {mean_20X}")
print(f"Variance of 5X: {var_5X}, Standard Deviation of 5X: {std_dev_5X}")
print(f"Variance of 10X: {var_10X}, Standard Deviation of 10X: {std_dev_10X}")