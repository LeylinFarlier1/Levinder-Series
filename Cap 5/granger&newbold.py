# ------------------------------------------------------------
# Notas sobre Granger y Newbold (1974): Regresión Espuria
# ------------------------------------------------------------

# Dos series de tiempo independientes con raíz unitaria (no estacionarias)
# pueden producir relaciones estadísticamente significativas al ser
# regresadas una sobre otra, aun si no tienen relación causal.

# Esto genera una "regresión espuria":
# - Coeficientes significativos
# - p-valores bajos (falsamente significativos)
# - R² altos (falsamente explicativos)

# Incluso sin deriva (tendencia), las caminatas aleatorias tienden a
# moverse y superponerse en ciertos tramos, aparentando correlación.

# ------------------------------------------------------------
# Notas: Replicación de una relación espuria entre series
# ------------------------------------------------------------

# Objetivo:
# Simular dos series independientes con raíz unitaria (random walks)
# y demostrar que al regresarlas una sobre otra, se obtiene una
# relación espuria (significativa pero falsa).

# Pasos para replicar el experimento de Granger y Newbold:

# 1. Fijar la semilla aleatoria para reproducibilidad.
# 2. Generar dos caminatas aleatorias independientes:
#    - Cada serie es una suma acumulada de errores aleatorios normales.
# 3. Graficar ambas series en el tiempo para visualizar la similitud aparente.
# 4. Regresar Y sobre X mediante OLS.
# 5. Guardar:
#    - Coeficiente estimado
#    - p-valor
#    - R²
#    - Estadístico de Durbin-Watson
# 6. Repetir los pasos anteriores (simulación Monte Carlo) muchas veces:
#    - Evaluar qué porcentaje de regresiones muestra significancia espuria.
#    - Verificar que el porcentaje excede el nivel de significancia nominal (ej. 5%).

# Resultado esperado:
# Aproximadamente 60-70% de las regresiones muestran p-valores < 0.05,
# a pesar de que las series no están relacionadas causalmente.

# Conclusión:
# Reforzamos el punto de que no se debe hacer regresión entre variables
# con raíz unitaria sin comprobar cointegración.
# Importar bibliotecas necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.stats.stattools import durbin_watson # Importar durbin_watson
# Configuración de la semilla para reproducibilidad
np.random.seed(1234)
# Número de observaciones
n = 400
# Generar dos series independientes con raíz unitaria (random walks)
def generar_random_walk(n):
    e_t = np.random.normal(0, 1, n)  # Ruido blanco
    x_t = np.zeros(n)
    for t in range(1, n):
        x_t[t] = x_t[t - 1] + e_t[t]  # Suma acumulada
    return x_t
x_t = generar_random_walk(n)
y_t = generar_random_walk(n)
# Convertir las series en un DataFrame
df = pd.DataFrame({'x_t': x_t, 'y_t': y_t})
# Graficar ambas series en el tiempo
plt.figure(figsize=(14, 6))
plt.plot(df['x_t'], label='Serie X (Random Walk)', color='blue')
plt.plot(df['y_t'], label='Serie Y (Random Walk)', color='orange')
plt.title('Series X e Y (Random Walks)')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.legend()
plt.show()
# Regresar Y sobre X mediante OLS
modelo = ols('y_t ~ x_t', data=df).fit()
# Guardar los resultados
coeficiente_estimado = modelo.params['x_t']
p_valor = modelo.pvalues['x_t']
r_squared = modelo.rsquared
estadistico_durbin_watson = durbin_watson(modelo.resid) # Calcular Durbin-Watson con los residuosultados
print(f"Coeficiente estimado: {coeficiente_estimado:.4f}")
print(f"P-valor: {p_valor:.4f}")
print(f"R²: {r_squared:.4f}")
print(f"Estadístico de Durbin-Watson: {estadistico_durbin_watson:.4f}")
# Repetir el experimento muchas veces (simulación Monte Carlo)
def simulacion_monte_carlo(iteraciones, n):
    resultados = []
    for _ in range(iteraciones):
        x_t = generar_random_walk(n)
        y_t = generar_random_walk(n)
        df = pd.DataFrame({'x_t': x_t, 'y_t': y_t})
        modelo = ols('y_t ~ x_t', data=df).fit()
        resultados.append({
            'coeficiente_estimado': modelo.params['x_t'],
            'p_valor': modelo.pvalues['x_t'],
            'r_squared': modelo.rsquared,
            'estadistico_durbin_watson': durbin_watson(modelo.resid) # Calcular Durbin-Watson con los residuos
        })
    return pd.DataFrame(resultados)

# Ejecutar la simulación Monte Carlo
iteraciones = 1000
resultados_monte_carlo = simulacion_monte_carlo(iteraciones, n)
# Calcular el porcentaje de regresiones con p-valor < 0.05
porcentaje_significativo = (resultados_monte_carlo['p_valor'] < 0.05).mean() * 100
print(f"Porcentaje de regresiones con p-valor < 0.05: {porcentaje_significativo:.2f}%")
# Graficar la distribución de los p-valores
plt.figure(figsize=(10, 6))
plt.hist(resultados_monte_carlo['p_valor'], bins=30, color='skyblue', edgecolor='black')
plt.axvline(0.05, color='red', linestyle='--', label='Nivel de Significancia (0.05)')
plt.title('Distribución de P-Valores en Simulación Monte Carlo')
plt.xlabel('P-Valor')
plt.ylabel('Frecuencia')
plt.legend()
plt.show()
# Este código demuestra la regresión espuria entre dos series independientes con raíz unitaria.
# La simulación Monte Carlo refuerza la idea de que las regresiones entre series no cointegradas pueden producir resultados engañosos.