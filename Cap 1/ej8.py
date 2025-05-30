 #8. Suppose that two exams (the midterm and the final) usually have averages of
 #70 and 80, respectively. They have standard deviations of 10 and 7, and their
 #correlation is 0.80. What is their covariance? Suppose that the exams were not
 #weighted equally. Rather, in calculating the course grade, the midterm carries a
 #weight of 40% andthe final has a weight of 60%. Whatisthe expected grade for
 #the course? What is the variance and standard deviation for the course grade?
import numpy as np
# Datos de los exámenes
midterm_mean = 70
midterm_std = 10
final_mean = 80
final_std = 7
correlation = 0.80
# Cálculo de la covarianza
covariance = correlation * midterm_std * final_std
# Cálculo de la nota esperada del curso
midterm_weight = 0.4
final_weight = 0.6
expected_grade = (midterm_weight * midterm_mean) + (final_weight * final_mean)
# Cálculo de la varianza y desviación estándar de la nota del curso
var_course = (midterm_weight ** 2 * midterm_std ** 2) + (final_weight ** 2 * final_std ** 2) + \
             (2 * midterm_weight * final_weight * covariance)
std_course = np.sqrt(var_course)
# Mostrar resultados
print(f"Covariance: {covariance}")
print(f"Expected course grade: {expected_grade}")
print(f"Variance of course grade: {var_course}")
print(f"Standard deviation of course grade: {std_course}")
