# ------------------------------------------------------------
# Notas: Estacionalidad determinística (6.1.1)
# ------------------------------------------------------------

# Contexto:
# Se puede incorporar estacionalidad en modelos de series temporales
# mediante variables dummy como regresores exógenos.

# Modelo base:
# Xt = et, donde et ~ iid N(0, σ²)  → Ruido blanco

# Si los datos son trimestrales, podemos añadir efectos estacionales
# determinísticos fijos en cada trimestre:

# Ejemplo 1: Usando las cuatro dummies estacionales (D1, D2, D3, D4):
# - D1 = 1 si es primer trimestre, 0 en otro caso
# - D2 = 1 si es segundo trimestre, etc.

# Modelo con intercepto estacional explícito:
# Xt = 5*D1 + 10*D2 - 3*D3 + 2*D4 + et

# Ejemplo 2: Enfoque equivalente (evita multicolinealidad):
# Se incluye una constante (intercepto) y se omite una dummy (ej. D1)

# Modelo con baseline implícito:
# Xt = 5 + 5*D2 - 8*D3 + 3*D4 + et
# Aquí, D1 (primer trimestre) es la categoría base.

# Interpretación de la media esperada en cada trimestre:
# - Q1: E(Xt) = 5            (referencia / baseline)
# - Q2: E(Xt) = 5 + 5  = 10
# - Q3: E(Xt) = 5 - 8  = -3
# - Q4: E(Xt) = 5 + 3  = 8

# Conclusión:
# Las variables dummy permiten modelar estacionalidad de manera determinística,
# y sus coeficientes representan diferencias respecto al trimestre base.

# Generar randow walk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
# Configuración de la semilla para reproducibilidad
np.random.seed(42)
# Número de observaciones
n = 50


e_t = np.random.normal(0, 1, n)  # Ruido blanco
trimestres = np.tile([1, 2, 3, 4], n // 4 + 1)[:n]
x_t = np.zeros(n)
for t in range(n):
    efecto_estacional = 0
    if trimestres[t] == 1:
        efecto_estacional = 5
    elif trimestres[t] == 2:
        efecto_estacional = 10
    elif trimestres[t] == 3:
        efecto_estacional = -3
    elif trimestres[t] == 4:
        efecto_estacional = 2
    x_t[t] = efecto_estacional + e_t[t]

# Crear un DataFrame para la regresión
df = pd.DataFrame({'x_t': x_t, 'trimestre': trimestres})

# Crear variables dummy para cada trimestre
# get_dummies creará columnas como 'trimestre_1', 'trimestre_2', etc.
# El dtype=int las convierte a 0s y 1s.
df_dummies = pd.get_dummies(df['trimestre'], prefix='D', dtype=int)

# Unir las dummies al DataFrame principal
df = pd.concat([df, df_dummies], axis=1)

# Ajustar el modelo de regresión: Xt = β1*D1 + β2*D2 + β3*D3 + β4*D4 + et
# Usamos '-1' en la fórmula para omitir el intercepto global y estimar
# explícitamente el coeficiente para cada dummy estacional.
# Los nombres de las columnas dummy generadas por get_dummies serán D_1, D_2, etc.
modelo_estacional_explicito = ols('x_t ~ D_1 + D_2 + D_3 + D_4 - 1', data=df).fit()

# Mostrar los resultados del modelo
print("Modelo con interceptos estacionales explícitos (D1, D2, D3, D4):")
print(modelo_estacional_explicito.summary())

# Graficar la serie original y los valores ajustados por el modelo
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['x_t'], label='Serie Original (x_t)', alpha=0.7)
plt.plot(df.index, modelo_estacional_explicito.fittedvalues, label='Valores Ajustados por el Modelo', linestyle='--')
plt.title('Serie Original vs. Ajustados (Modelo: x_t ~ D_1 + D_2 + D_3 + D_4 - 1)')
plt.ylabel('Valor')
plt.legend()
plt.grid(True)
plt.show()