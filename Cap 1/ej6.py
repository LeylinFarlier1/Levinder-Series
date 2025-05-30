 #6. Suppose midterm grades were distributed normally, with a mean of 70 and a
 #standard deviation of 10. Suppose further that the professor multiplies each
 #exam by 1.10 as a curve. Calculate the new mean, standard deviation, and
 #variance of the curved midterm grades.
import numpy as np

midterm_grade = np.random.normal(loc=70, scale=10, size=1000)
mean = midterm_grade.mean()
std_dev = midterm_grade.std()
var = midterm_grade.var()
a=1.1
a_mean = a * mean
a_std_dev = a * std_dev
a_var = a**2 * var

print(f"Mean: {mean}, Standard Deviation: {std_dev}, Variance: {var}")
print (f"a * Mean: {a_mean}, a * Standard Deviation: {a_std_dev}, a^2 * Variance: {a_var}")