#10. Suppose that in country A, the price of a widget has a mean of $100 and a
 #variance of $25. Country B has a fixed exchange rate with A, so that it takes
 #two B-dollars to equal one A-dollar. What is the expected price of a widget in
 #B-dollars? What is its variance in B-dollars? What would the expected price
 #and variance equal if the exchange rate were three-to-one?
 
price_A_mean = 100
price_A_variance = 25
TCb_a= 2
# Expected price in B-dollars
price_B_mean = price_A_mean * TCb_a
# Variance in B-dollars
price_B_variance = (TCb_a ** 2) * price_A_variance
tcb_a_3 = 3
price_b_mean_2 = price_A_mean * tcb_a_3
price_b_variance_2 = (tcb_a_3 ** 2) * price_A_variance
# mostrar resultados
print(f"Expected price of a widget in B-dollars (2:1 exchange rate): {price_B_mean}")
print(f"Variance of price in B-dollars (2:1 exchange rate): {price_B_variance}")
print(f"Expected price of a widget in B-dollars (3:1 exchange rate): {price_b_mean_2}")
print(f"Variance of price in B-dollars (3:1 exchange rate): {price_b_variance_2}")
