# Modelo con tendencia determinista:
# Xt = β0 + β1*t + et
# - β0: constante
# - β1: coeficiente de tendencia
# - t: tiempo
# - et: error IID con media cero y varianza constante

# Media del proceso: E(Xt) = β0 + β1*t → NO estacionaria
# Varianza: Var(Xt) = Var(et) = σ² → estacionaria

# Al tomar primeras diferencias:
# ΔXt = Xt − Xt−1 = β1 + et − et−1
# → Este proceso tiene una raíz unitaria MA (moving average)
# → No debe tratarse con diferenciación, sino eliminando la tendencia directamente

# ¿Cómo se trabaja con tendencias determinísticas?
# En lugar de aplicar primeras diferencias, lo adecuado es estimar y remover la tendencia lineal.
# Esto se hace ajustando una regresión lineal de Xt sobre el tiempo (t) y obteniendo los residuos.
# Los residuos resultantes (Xt − E(Xt)) representan una serie sin tendencia, y se espera que sean estacionarios.
# De esta forma, se "detrenda" la serie respetando su estructura estadística sin inducir raíces MA innecesarias.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuración de la semilla para reproducibilidad
np.random.seed(42)
# Número de observaciones
n = 500
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
from statsmodels.formula.api import ols
modelo_tendencia = ols('serie_tendencia ~ np.arange(n)', data=df_tendencia).fit()
# Obtener los residuos (serie detrendada)
residuos = modelo_tendencia.resid
# Convertir los residuos a un DataFrame
df_residuos = pd.DataFrame({'residuos': residuos})
# Graficar la serie original y los residuos
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(df_tendencia['serie_tendencia'], label='Serie con Tendencia', color='blue')

plt.title('Serie con Tendencia Determinista')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.legend()

# Subplot 2: Residuos (Serie Detrendada)
plt.subplot(3, 1, 2)
plt.plot(df_residuos['residuos'], label='Residuos (Serie Detrendada)', color='green')
mean_residuos = df_residuos['residuos'].mean()
plt.axhline(mean_residuos, color='red', linestyle='--', label=f'Media: {mean_residuos:.2f}')
plt.title('Residuos de la Serie Detrendada')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.legend()

# Subplot 3: Primera Diferencia
plt.subplot(3, 1, 3)
diff_serie = np.diff(df_tendencia['serie_tendencia'])
plt.plot(diff_serie, label='Primera Diferencia', color='purple')
mean_diff = diff_serie.mean()
plt.axhline(mean_diff, color='red', linestyle='--', label=f'Media: {mean_diff:.2f}')
plt.title('Primera Diferencia de la Serie')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.legend()

plt.tight_layout()
plt.show()
# Imprimir el resumen del modelo
print(modelo_tendencia.summary())
# Resumen del modelo de tendencia
# =============================================================================