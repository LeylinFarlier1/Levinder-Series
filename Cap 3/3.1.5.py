# Theoretical Partial ACFs (PACFs) are more difficult to derive, so we only outline their general properties.
# PACFs are similar to ACFs but remove the effects of intermediate lags.
# For example, the PACF at lag 2 removes the effect of autocorrelation from lag 1.
# Likewise, the PACF at lag 3 removes effects from lags 1 and 2.
# 
# A useful rule of thumb:
# - Theoretical PACFs are the "mirrored opposites" of ACFs.
# - For an AR(p) process, the ACF dies down exponentially,
#   while the PACF has spikes at lags 1 through p, then zero afterward.
# - For an MA(q) process, the ACF has non-zero spikes up to lag q and zero afterward,
#   while the PACF gradually damps toward zero, often oscillating slightly.

#Theoretical (a)ACFand(b)PACFofAR(1):Xt = 0.50Xt−1 + et
import numpy as np
import matplotlib.pyplot as plt

def theoretical_acf_ar1(b1, lags):
    """
    Calculate the theoretical ACF for an AR(1) process, setting ACF at lag 0 to 0.
    """
    acf = [0.0]  # ρ₀ = 0
    if lags >= 1:
        rho1 = b1  # ρ₁ = b₁
        acf.append(rho1)
        for s in range(2, lags + 1):
            rho_s = b1 * acf[s - 1]
            acf.append(rho_s)
    return acf[:lags + 1]

def theoretical_pacf_ar1(b1, lags):
    """
    Calculate the theoretical PACF for an AR(1) process, setting PACF at lag 0 to 0.
    """
    pacf = [0.0]  # φ₀ = 0
    if lags >= 1:
        phi1 = b1  # φ₁ = b₁
        pacf.append(phi1)
        pacf.extend([0.0] * (lags - 1))
    return pacf[:lags + 1]

# Coefficient for AR(1)
b1 = 0.5
# Number of lags to calculate
lags = 30
# Calculate the theoretical ACF and PACF
acf = theoretical_acf_ar1(b1, lags)
pacf = theoretical_pacf_ar1(b1, lags)
# Print the results
print("Theoretical ACF for AR(1):", acf)
print("Theoretical PACF for AR(1):", pacf)
# Plot the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.stem(range(lags + 1), acf, linefmt='b-', markerfmt='bo', basefmt=' ', label='ACF AR(1)')
plt.title('Theoretical ACF for AR(1) Process')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.legend()
plt.grid()
plt.subplot(1, 2, 2)
plt.stem(range(lags + 1), pacf, linefmt='r-', markerfmt='ro', basefmt=' ', label='PACF AR(1)')
plt.title('Theoretical PACF for AR(1) Process')
plt.xlabel('Lags')
plt.ylabel('PACF')
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
