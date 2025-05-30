import numpy as np

def theoretical_acf_ar3(b1, b2, b3, lags):
    # Initialize ACF with ρ₀ = 1
    acf = [1.0]  
    
    # Solve Yule-Walker equations for ρ₁, ρ₂, ρ₃
    if lags >= 1:
        # ρ₁ = b₁ ρ₀ + b₂ ρ₁ + b₃ ρ₂  → but ρ₀=1
        # ρ₂ = b₁ ρ₁ + b₂ ρ₀ + b₃ ρ₁
        # We need to solve a linear system for ρ₁, ρ₂, ρ₃
        # Let's set up the equations:
        # ρ₁ = b₁ + b₂ ρ₁ + b₃ ρ₂  → (1 - b₂) ρ₁ - b₃ ρ₂ = b₁
        # ρ₂ = b₁ ρ₁ + b₂ + b₃ ρ₁  → - (b₁ + b₃) ρ₁ + ρ₂ = b₂
        A = np.array([
            [1 - b2, -b3],
            [-(b1 + b3), 1]
        ])
        b = np.array([b1, b2])
        rho1, rho2 = np.linalg.solve(A, b)
        acf.extend([rho1, rho2])
        
        # Now compute ρ₃ using recursion
        if lags >= 3:
            rho3 = b1 * rho2 + b2 * rho1 + b3 * 1  # ρ₃ = b₁ ρ₂ + b₂ ρ₁ + b₃ ρ₀
            acf.append(rho3)
    
    # For lags ≥ 4, use recursion: ρₛ = b₁ ρₛ₋₁ + b₂ ρₛ₋₂ + b₃ ρₛ₋₃
    for s in range(4, lags + 1):
        rho_s = b1 * acf[s-1] + b2 * acf[s-2] + b3 * acf[s-3]
        acf.append(rho_s)
    
    return acf[:lags+1]  # Return ACF up to desired lag

b1, b2, b3 = -0.5, -0.3, 0.2  # Example AR(3) coefficients
lags = 10
acf = theoretical_acf_ar3(b1, b2, b3, lags)
print(acf)
    #plot the results
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.stem(range(lags + 1), acf, linefmt='b-', markerfmt='bo', basefmt=' ', label='ACF AR(3)')    
plt.title('Theoretical ACF for AR(3) Process')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
