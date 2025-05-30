 #9. Suppose that an exam has an average grade of 75 and a standard deviation of
 #10. Suppose that the professor decided to curve the exams by adding five points
 #to everyone’s score. What are the mean, standard deviation and variance of the
# curved exam

exam_mean= 75
exam_std = 10
exam_curve = 5
curved_mean = exam_mean + exam_curve
curved_std = exam_std  
curved_var = exam_std ** 2  # Variance does not change with a constant addition
print(f"Original Mean: {exam_mean}, Original Standard Deviation: {exam_std}")
print(f"Curved Mean: {curved_mean}, Curved Standard Deviation: {curved_std}")
print(f"Curved Variance: {curved_var}")