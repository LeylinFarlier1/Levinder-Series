# Suppose you are given the dataset rGDPgr.dta, which contains data on seasonally
# adjusted real GDP growth rates, quarterly, from 1947 Q2 through 2017 Q2.
# Alternatively, you can download it from the Federal Reserve’s website, and tsset
# the data accordingly.
#
# How should we model the real GDP growth rate? As an AR(p) process? An MA(q)?
# Or an ARMA(p,q)? And of what order p or q?
#
# The standard approach is to calculate the empirical ACF and PACF exhibited by the data,
# and compare them to the characteristic (i.e., theoretical) ACF/PACF implied by various models.
#
# So, our first step is to calculate the empirical ACF and PACF.

 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf 
import fredapi

# Configurar FRED
fred = fredapi.Fred(api_key='9409c2325851d01678597f599dfde89c')

# Descargar datos del PIB real (1947-2017)
gdp_data = fred.get_series('GDPC1', start='1947-01-01',
                            end='2017-06-30')
# Crear DataFrame y renombrar columna
gdp_df = pd.DataFrame(gdp_data, columns=['GDP'])
gdp_df.index = pd.to_datetime(gdp_df.index)

# Calcular crecimiento logarítmico trimestral (%) - CONSIGNA: 1947 Q2 a 2017 Q2
gdp_df['Growth Rate'] = 100 * np.log(gdp_df['GDP']).diff()
gdp_growth = gdp_df['Growth Rate'].dropna().loc['1947-04-01':'2017-06-30']  # Q2 1947 a Q2 2017

# Paso 1: Análisis exploratorio (CONSIGNA)
fig, ax = plt.subplots(3, 1, figsize=(12, 12))

# Serie temporal
ax[0].plot(gdp_growth)
ax[0].axhline(0, color='r', linestyle='--')
ax[0].set_title('Crecimiento Trimestral del PIB Real (1947 Q2 - 2017 Q2)')
ax[0].set_ylabel('% Cambio')

# ACF y PACF empíricas
plot_acf(gdp_growth, lags=20, ax=ax[1], title='ACF Empírica')
plot_pacf(gdp_growth, lags=20, ax=ax[2], title='PACF Empírica')

plt.tight_layout()
plt.show()

from statsmodels.tsa.arima.model import ARIMA   
# Paso 2: Ajustar modelos AR(p), MA(q) y ARMA(p,q)
# Ajustar modelo AR(2)
model_ar2 = ARIMA(gdp_growth, order=(2, 0, 0))
results_ar2 = model_ar2.fit()
print("Resumen del modelo AR(2):")
print(results_ar2.summary())
# Ajustar modelo MA(2)
model_ma2 = ARIMA(gdp_growth, order=(0, 0, 2))
results_ma2 = model_ma2.fit()
print("Resumen del modelo MA(2):")
print(results_ma2.summary())

#cual mdoelo es mejor?
from statsmodels.tsa.stattools import adfuller
# Prueba de Dickey-Fuller aumentada para verificar estacionariedad
adf_result = adfuller(gdp_growth)
print(f"ADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")
# Interpretación de la prueba ADF
if adf_result[1] < 0.05:
    print("La serie es estacionaria (rechazamos H0)")
else:
    print("La serie no es estacionaria (no rechazamos H0)")
# Paso 3: Comparar ACF y PACF teóricas con las empíricas
def theoretical_acf_ar2(b1, b2, lags):
    acf = [1.0]  # ρ₀ = 1
    if lags >= 1:
        rho1 = b1 / (1 - b2)
        acf.append(rho1)
    if lags >= 2:
        rho2 = b1**2 * rho1 + b2
        acf.append(rho2)
    for s in range(3, lags + 1):
        rho_s = b1 * acf[s-1] + b2 * acf[s-2]
        acf.append(rho_s)
    return acf
# Obtener coeficientes AR(2) estimados
b1_hat, b2_hat = results_ar2.arparams
# Calcular ACF teórica
lags = 40
acf_theoretical_ar2 = theoretical_acf_ar2(b1_hat, b2_hat, lags)
# Graficar ACF empírica y teórica
fig, ax = plt.subplots(1, 1, figsize=(12, 5))
ax.stem(range(lags + 1), acf_theoretical_ar2, linefmt='b-', markerfmt='bo', basefmt=' ', label='ACF Teórica AR(2)')
from statsmodels.tsa.stattools import acf
acf_sample = acf(gdp_growth, nlags=lags, fft=True)
ax.stem(range(lags + 1), acf_sample, linefmt='g--', markerfmt='go', basefmt=' ', label='ACF Empírica')
ax.axhline(0, color='red', linestyle='--')
ax.set_title(f'ACF Teórica vs. Empírica de un Proceso AR(2)\n'
             f'β₁={b1_hat:.3f}, β₂={b2_hat:.3f} (muestrales)')
ax.set_xlabel('Lags')
ax.set_ylabel('ACF')
ax.legend()
plt.tight_layout()
plt.show()

# Paso 4: Comparar AIC y BIC de todos los modelos juntos
print("Comparación de AIC y BIC de los modelos:")
print(f"AR(2):  AIC = {results_ar2.aic:.2f}, BIC = {results_ar2.bic:.2f}")
print(f"MA(2):  AIC = {results_ma2.aic:.2f}, BIC = {results_ma2.bic:.2f}")

# AR(3)
model_ar3 = ARIMA(gdp_growth, order=(3, 0, 0))
results_ar3 = model_ar3.fit()
print(f"AR(3):  AIC = {results_ar3.aic:.2f}, BIC = {results_ar3.bic:.2f}")

# Mostrar resumen del mejor modelo según AIC/BIC
aic_values = {
    'AR(2)': results_ar2.aic,
    'MA(2)': results_ma2.aic,
    'AR(3)': results_ar3.aic
}
bic_values = {
    'AR(2)': results_ar2.bic,
    'MA(2)': results_ma2.bic,
    'AR(3)': results_ar3.bic
}
best_aic = min(aic_values, key=aic_values.get)
best_bic = min(bic_values, key=bic_values.get)
print(f"\nMejor modelo según AIC: {best_aic}")

# Definir función para ACF teórica de AR(3)
def theoretical_acf_ar3(b1, b2, b3, lags):
    acf = [1.0]  # ρ₀ = 1
    if lags >= 1:
        rho1 = b1 / (1 - b2 - b3)
        acf.append(rho1)
    if lags >= 2:
        rho2 = (b1 * acf[1] + b2) / (1 - b3)
        acf.append(rho2)
    if lags >= 3:
        rho3 = b1 * acf[2] + b2 * acf[1] + b3 * acf[0]
        acf.append(rho3)
    for s in range(4, lags + 1):
        rho_s = b1 * acf[s-1] + b2 * acf[s-2] + b3 * acf[s-3]
        acf.append(rho_s)
    return acf

# Obtener coeficientes AR(3) estimados
b1_hat_ar3, b2_hat_ar3, b3_hat_ar3 = results_ar3.arparams

# Calcular ACF teórica para AR(3)
acf_theoretical_ar3 = theoretical_acf_ar3(b1_hat_ar3, b2_hat_ar3, b3_hat_ar3, lags)
# Graficar ACF empírica y teórica para AR(3)
fig, ax = plt.subplots(1, 1, figsize=(12, 5))
ax.stem(range(lags + 1), acf_theoretical_ar3, linefmt='b-', markerfmt='bo', basefmt=' ', label='ACF Teórica AR(3)')
acf_sample_ar3 = acf(gdp_growth, nlags=lags, fft=True)
ax.stem(range(lags + 1), acf_sample_ar3, linefmt='g--', markerfmt='go', basefmt=' ', label='ACF Empírica AR(3)')
ax.axhline(0, color='red', linestyle='--')
ax.set_title(f'ACF Teórica vs. Empírica de un Proceso AR(3)\n'
             f'β₁={b1_hat_ar3:.3f}, β₂={b2_hat_ar3:.3f}, β₃={b3_hat_ar3:.3f} (muestrales)')
ax.set_xlabel('Lags')
ax.set_ylabel('ACF')
ax.legend()
plt.tight_layout()
plt.show()
# Comparar AIC y BIC del modelo AR(3)
print(f"AIC AR(3): {results_ar3.aic}, BIC AR(3): {results_ar3.bic}")
