# Comparación: Random Walk con Deriva vs. Tendencia Determinista

# Ambos modelos son NO estacionarios, pero por razones distintas:

# 1. Random Walk con deriva:
#    Xt = β0 + Xt−1 + et → Xt = t*β0 + e1 + e2 + ... + et
#    - Media: E(Xt) = t*β0 → crece linealmente con el tiempo
#    - Varianza: Var(Xt) = t*σ² → también crece con el tiempo
#    - Solución: aplicar primera diferencia → ΔXt = β0 + et

# 2. Tendencia determinista:
#    Xt = β0 + β1*t + et
#    - Media: E(Xt) = β0 + β1*t → también crece linealmente con el tiempo
#    - Varianza: Var(Xt) = σ² → constante en el tiempo
#    - Solución: eliminar la tendencia vía regresión de Xt sobre t y trabajar con los residuos

# A simple vista, ambas series pueden parecer similares porque sus medias crecen con el tiempo.
# Sin embargo, la varianza creciente del random walk con deriva permite diferenciarlos formalmente.

# En resumen, aunque ambos son no estacionarios, el random walk con deriva tiene varianza creciente,
# mientras que la tendencia determinista tiene varianza constante. La forma de tratarlos es diferente:
# - Random Walk con deriva: aplicar primera diferencia para obtener una serie estacionaria.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
# Configuración de la semilla para reproducibilidad
np.random.seed(42)
# Número de observaciones
n = 400
# Coeficientes de la tendencia determinista
beta0 = 100.0  # Intercepto
beta1 = -0.1  # Pendiente de la tendencia
# Generar serie con tendencia determinista
serie_tendencia = np.zeros(n)
for t in range(n):
    et = np.random.normal(0, 1)  # Ruido blanco
    serie_tendencia[t] = beta0 + beta1 * t + et
# Convertir la serie en un DataFrame
df_tendencia = pd.DataFrame({'serie_tendencia': serie_tendencia})
# Ajustar una regresión lineal para remover la tendencia
modelo_tendencia = ols('serie_tendencia ~ np.arange(n)', data=df_tendencia).fit()
# Obtener los residuos (serie detrendada)
residuos = modelo_tendencia.resid
# Convertir los residuos a un DataFrame
df_residuos = pd.DataFrame({'residuos': residuos})
# Graficar la serie original y los residuos
#genero una serie random walk con deriva
beta0_rw = -0.3  # Deriva del Random Walk
X_rw = np.zeros(n)
for t in range(1, n):
    e_rw = np.random.normal(0, 1)  # Ruido blanco
    X_rw[t] = beta0_rw + X_rw[t - 1] + e_rw  # Random Walk con Drift
# Convertir la serie random walk con deriva a un DataFrame
df_rw = pd.DataFrame({'serie_rw': X_rw})
# calcular la primera diferencia de la serie random walk con deriva
df_rw['diff_rw'] = df_rw['serie_rw'].diff().dropna()
# Graficar la serie original y los residuos, la serie random walk con deriva y la primera diferencia
fig, axs = plt.subplots(2, 2, figsize=(14, 8))

window = 20

# Serie con tendencia determinista
axs[0, 0].plot(df_tendencia['serie_tendencia'], label='Serie')
var_tend = df_tendencia['serie_tendencia'].rolling(window).var()
ax2 = axs[0, 0].twinx()
ax2.plot(var_tend, color='red', alpha=0.7, label='Varianza móvil (ventana=20)')
axs[0, 0].set_title('Serie con Tendencia Determinista')
axs[0, 0].set_xlabel('Tiempo')
axs[0, 0].set_ylabel('Valor')
ax2.set_ylabel('Varianza móvil')
lines, labels = axs[0, 0].get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
axs[0, 0].legend(lines + lines2, labels + labels2, loc='upper right')

# Residuos (serie detrendada)
axs[1, 0].plot(df_residuos['residuos'])
axs[1, 0].set_title('Residuos (Serie Detrendada)')
axs[1, 0].set_xlabel('Tiempo')
axs[1, 0].set_ylabel('Valor')

# Serie Random Walk con deriva
axs[0, 1].plot(df_rw['serie_rw'], label='Serie')
var_rw = df_rw['serie_rw'].rolling(window).var()
ax4 = axs[0, 1].twinx()
ax4.plot(var_rw, color='red', alpha=0.7, label='Varianza móvil (ventana=20)')
axs[0, 1].set_title('Random Walk con Deriva')
axs[0, 1].set_xlabel('Tiempo')
axs[0, 1].set_ylabel('Valor')
ax4.set_ylabel('Varianza móvil')
lines, labels = axs[0, 1].get_legend_handles_labels()
lines2, labels2 = ax4.get_legend_handles_labels()
axs[0, 1].legend(lines + lines2, labels + labels2, loc='upper right')

# Primera diferencia del Random Walk con deriva
axs[1, 1].plot(df_rw['diff_rw'])
axs[1, 1].set_title('Primera Diferencia RW con Deriva')
axs[1, 1].set_xlabel('Tiempo')
axs[1, 1].set_ylabel('Valor')

plt.tight_layout()
plt.show()