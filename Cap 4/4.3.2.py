# =============================================================================
# Condiciones de Estacionariedad para un Proceso AR(2)
# -----------------------------------------------------------------------------
# Un proceso AR(2):    Xt = β1*Xt−1 + β2*Xt−2 + et
# 
# Es estacionario si las raíces del polinomio inverso característico:
#     z² - β1*z - β2 = 0
# 
# están **dentro del círculo unitario** (|z| < 1).
# 
# Para garantizarlo, se deben cumplir simultáneamente estas tres condiciones:
#     1. β1 + β2 < 1
#     2. β2 - β1 < 1
#     3. |β2| < 1
# 
# Estas tres restricciones delimitan un triángulo en el plano (β1, β2),
# conocido como el **Triángulo de Stralkowski**.
# 
# Además:
# - Si las raíces son complejas (parte imaginaria ≠ 0), el proceso presentará 
#   oscilaciones.
# - Si están dentro del triángulo pero debajo de la parábola invertida:
#     β2 < - (β1²) / 4
#   entonces el proceso tendrá oscilaciones **estables**.
# - Si están fuera del triángulo, las oscilaciones serán **explosivas**.
# =============================================================================


# Condiciones de Estacionariedad para un Proceso AR(2)
import matplotlib.pyplot as plt
import numpy as np
# Definir los coeficientes β1 y β2
beta1 = 0.5
beta2 = 0.3

# Creo una serie AR(2) estacionaria y otra no estacionaria
n = 100
# Generar serie AR(2) estacionaria
# Generar serie AR(2) estacionaria
serie_ar2 = np.zeros(n)
for t in range(2, n):
    et = np.random.normal(0, 1)  # Ruido blanco
    serie_ar2[t] = beta1 * serie_ar2[t - 1] + beta2 * serie_ar2[t - 2] + et

# Generar serie AR(2) no estacionaria
beta1_no_estacionario = -0.9  # Coeficiente AR(2) fuera del rango estacionario
serie_ar2_no_estacionario = np.zeros(n)
for t in range(2, n):
    et = np.random.normal(0, 1)  # Ruido blanco
    serie_ar2_no_estacionario[t] = beta1_no_estacionario * serie_ar2_no_estacionario[t - 1] + beta2 * serie_ar2_no_estacionario[t - 2] + et

# Graficar ambas series en un solo subplot (dos filas)
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axs[0].plot(serie_ar2, marker='o', linestyle='-', color='blue', label='Serie AR(2) Estacionaria')
axs[0].axhline(0, color='red', linestyle='--', label='Media = 0')
axs[0].set_title('Serie AR(2) Estacionaria')
axs[0].set_ylabel('Valor')
axs[0].legend()

axs[1].plot(serie_ar2_no_estacionario, marker='o', linestyle='-', color='orange', label='Serie AR(2) No Estacionaria')
axs[1].axhline(0, color='red', linestyle='--', label='Media = 0')
axs[1].set_title('Serie AR(2) No Estacionaria')
axs[1].set_xlabel('Tiempo')
axs[1].set_ylabel('Valor')
axs[1].legend()

plt.tight_layout()
plt.show()
#con la función autoreg de statsmodels encontramos el modulo del polinomio característico
from statsmodels.tsa.ar_model import AutoReg
# Crear un DataFrame con la serie AR(2) estacionaria
import pandas as pd 

# DataFrames para ambas series
df_ar2 = pd.DataFrame({'serie': serie_ar2})
df_ar2_no_estacionario = pd.DataFrame({'serie_no_estacionaria': serie_ar2_no_estacionario})

# Ajustar modelos AR(2)
modelo_ar2 = AutoReg(df_ar2['serie'], lags=2).fit()
modelo_ar2_no_estacionario = AutoReg(df_ar2_no_estacionario['serie_no_estacionaria'], lags=2).fit()

# Reporte comparativo
print("="*60)
print("Comparativa de Modelos AR(2)")
print("="*60)
print("Modelo AR(2) Estacionario")
print(modelo_ar2.summary())
print("Raíces del polinomio característico (estacionario):")
print(modelo_ar2.roots)
modulos_estacionario = np.abs(modelo_ar2.roots)
print("Módulos de las raíces (estacionario):", modulos_estacionario)
if np.all(modulos_estacionario > 1):
    print("=> La serie estacionaria ES estacionaria (todas las raíces > 1)")
else:
    print("=> La serie estacionaria NO es estacionaria (alguna raíz <= 1)")
print("-"*60)
print("Modelo AR(2) No Estacionario")
print(modelo_ar2_no_estacionario.summary())
print("Raíces del polinomio característico (no estacionario):")
print(modelo_ar2_no_estacionario.roots)
modulos_no_estacionario = np.abs(modelo_ar2_no_estacionario.roots)
print("Módulos de las raíces (no estacionario):", modulos_no_estacionario)
if np.all(modulos_no_estacionario > 1):
    print("=> La serie NO estacionaria ES estacionaria (todas las raíces > 1)")
else:
    print("=> La serie NO estacionaria NO es estacionaria (alguna raíz <= 1)")
print("="*60)