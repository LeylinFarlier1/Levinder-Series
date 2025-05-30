# calculate the first three lags of the theoretical ACFs
 #implied by the following AR(2) processes:
 #(a) Xt = 0.50Xt−1 −0.20Xt−2 +et
 #(b) Xt =−0.50Xt−1 −0.20Xt−2 +et

import numpy as np
b1, b2 = 0.50, -0.20  # Coefficients for AR(2) process (a)

def theoretical_acf_ar2(b1, b2, lags):
    # Initialize ACF with ρ₀ = 1
    acf = [1.0]
    
    if lags >= 1:
        # Solve Yule-Walker for ρ₁
        rho1 = b1 / (1 - b2)
        acf.append(rho1)
        
        if lags >= 2:
            # Solve for ρ₂
            rho2 = b1**2 * rho1 + b2
            acf.append(rho2)
    
    # For lags ≥ 3, use recursion: ρₛ = b₁ ρₛ₋₁ + b₂ ρₛ₋₂
    for s in range(3, lags + 1):
        rho_s = b1 * acf[s-1] + b2 * acf[s-2]
        acf.append(rho_s)
    
    return acf[:lags+1]  # Return ACF up to desired lag

# Calculate the first three lags of the theoretical ACF for process (a)
lags = 10

acf_a = theoretical_acf_ar2(b1, b2, lags)
# Calculate the first three lags of the theoretical ACF for process (b)
b1_b, b2_b = -0.50, -0.20  # Coefficients for AR(2) process (b)
acf_b = theoretical_acf_ar2(b1_b, b2_b, lags)

# Print the results
print("Theoretical ACF for process (a):", acf_a)
print("Theoretical ACF for process (b):", acf_b)

#plot the results
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.stem(range(lags + 1), acf_a, linefmt='b-', markerfmt='bo', basefmt=' ', label='ACF (a)')
plt.stem(range(lags + 1), acf_b, linefmt='r--', markerfmt='ro', basefmt=' ', label='ACF (b)')
plt.title('Theoretical ACF for AR(2) Processes')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


