# La Función de Autocorrelación Parcial (PACF) muestra la correlación entre pares ordenados (Xt, Xt+k),
# eliminando el efecto de las variables intermedias Xt+1, Xt+2, ..., Xt+k−1.
# Este procedimiento se implementa de forma natural mediante análisis de regresión.

# Por ejemplo, en una regresión del tipo:
#     Y = β0 + β1X + β2Z
# el coeficiente β1 representa la relación entre Y y X, manteniendo constante el efecto de Z.

# Denotamos el coeficiente de autocorrelación parcial entre Xt y Xt+k como φ_kk
# (siguiendo la notación de Pankratz, 1983 y 1991).

# Para calcular la PACF empírica a distintos rezagos (lags), se ajustan regresiones lineales como sigue:

# Rezago 1 (lag 1):
#     Xt = φ_10 + φ_11 * Xt-1 + et     --> φ_11 es la PACF en el lag 1

# Rezago 2 (lag 2):
#     Xt = φ_20 + φ_21 * Xt-1 + φ_22 * Xt-2 + et     --> φ_22 es la PACF en el lag 2

# Rezago 3 (lag 3):
#     Xt = φ_30 + φ_31 * Xt-1 + φ_32 * Xt-2 + φ_33 * Xt-3 + et     --> φ_33 es la PACF en el lag 3

# Y así sucesivamente, estimando:
#     Xt = φ_k0 + φ_k1 * Xt-1 + φ_k2 * Xt-2 + ... + φ_kk * Xt-k + et

# Finalmente, la PACF de la serie X se define como el conjunto de los coeficientes φ_kk:
#     PACF(X) = {φ_11, φ_22, φ_33, ..., φ_kk}

# Estos valores indican la autocorrelación parcial en cada rezago k, útil para identificar el orden de un modelo AR(k).
# Esta metodología manual permite entender en profundidad cómo se construye la PACF
# y por qué refleja la autocorrelación neta entre Xt y Xt-k, eliminando el efecto
# de los rezagos intermedios.

# Luego, estimaremos la PACF de forma más rápida utilizando comandos predefinidos
# en bibliotecas de Python como statsmodels (por ejemplo, con plot_pacf o pacf).

# Mostraremos que ambas aproximaciones (la forma extensa y la forma rápida) son equivalentes,
# validando así el uso de métodos automatizados.

# A partir de esa validación, utilizaremos el enfoque más directo (la forma corta)
# en los cálculos posteriores para ahorrar tiempo y facilitar el análisis.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import pacf
import statsmodels.api as sm

# Generar datos de una serie temporal AR(4)
n = 1000
np.random.seed(42)  # Para reproducibilidad
e_t = np.random.normal(0, 1, n)  # Ruido blanco
b1 = 0.5  # Coeficiente AR(1)
b2 = 0.2  # Coeficiente AR(2)
b3 = 0.1  # Coeficiente AR(3)
b4 = 0.1  # Coeficiente AR(4)
X = np.zeros(n)  # Inicializar la serie temporal
for i in range(4, n):
    X[i] = b1 * X[i-1] + b2 * X[i-2] + b3 * X[i-3] + b4 * X[i-4] + e_t[i]  # AR(4) con ruido blanco
# Graficar las últimas 100 observaciones de X
plt.figure(figsize=(10, 4))
plt.plot(X[-100:], label='Últimas 100 observaciones de X')
plt.axhline(np.mean(X), color='red', linestyle='--', label='Media')
plt.title('Últimas 100 observaciones de X')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.legend()
plt.tight_layout()
plt.show()
# Estimar la PACF manualmente
# Convertir X a una Serie de pandas para facilitar el manejo
X = pd.Series(X)

# Crear un DataFrame con todos los rezagos necesarios y eliminar filas con NaN
df = pd.DataFrame({
    'Xt': X,
    'Xt_1': X.shift(1),
    'Xt_2': X.shift(2),
    'Xt_3': X.shift(3),
    'Xt_4': X.shift(4)
}).dropna().reset_index(drop=True)

# PACF lag 1
modelo = sm.OLS(df['Xt'], df[['Xt_1']]).fit()
modelo_summary = modelo.summary()

# PACF lag 2
modelo_2 = sm.OLS(df['Xt'], df[['Xt_1', 'Xt_2']]).fit()
modelo_2_summary = modelo_2.summary()

# PACF lag 3
modelo_3 = sm.OLS(df['Xt'], df[['Xt_1', 'Xt_2', 'Xt_3']]).fit()
modelo_4 = sm.OLS(df['Xt'], df[['Xt_1', 'Xt_2', 'Xt_3', 'Xt_4']]).fit()

# Extraer PACF manualmente: φ_kk es el coeficiente del último rezago en cada regresión
pacf_manual = [
    modelo.params['Xt_1'],  # φ_11
    modelo_2.params['Xt_2'],  # φ_22
    modelo_3.params['Xt_3'],  # φ_33
    modelo_4.params['Xt_4']   # φ_44
]

print("PACF manual:")
print(f"φ_11: {pacf_manual[0]:.4f}, φ_22: {pacf_manual[1]:.4f}, φ_33: {pacf_manual[2]:.4f}, φ_44: {pacf_manual[3]:.4f}")

# Comparar con función automática de statsmodels
pacf_auto = pacf(X, nlags=4)
print("\nPACF usando statsmodels:")
print(f"φ_11: {pacf_auto[1]:.4f}, φ_22: {pacf_auto[2]:.4f}, φ_33: {pacf_auto[3]:.4f}, φ_44: {pacf_auto[4]:.4f}")