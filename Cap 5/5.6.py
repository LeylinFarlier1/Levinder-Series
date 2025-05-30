# Sobre la eliminación de la tendencia: Diferenciación vs Detrendización

# Existen dos tipos comunes de procesos no estacionarios:
# - Procesos con raíz unitaria (diferencia-estacionarios)
# - Procesos con tendencia determinista (tendencia-estacionarios)

# Ambos tienen una media que cambia con el tiempo, pero:
# - Los procesos con raíz unitaria (como un Random Walk con deriva) tienen varianza creciente.
# - Los procesos con tendencia determinista tienen varianza constante.

# Estrategias adecuadas:
# - Si el proceso tiene una raíz unitaria: aplicar la primera diferencia.
# - Si el proceso tiene una tendencia determinista: eliminar la tendencia mediante regresión lineal y trabajar con los residuos (detrendizar).

# Importancia:
# - Diferenciar incorrectamente una serie con tendencia determinista distorsiona las propiedades del error.
# - Detrendizar incorrectamente una serie con raíz unitaria produce residuos con varianza creciente (no estacionarios), que puede llevar a conclusiones engañosas.

# Conclusión:
# Antes de aplicar transformaciones, es esencial identificar correctamente el tipo de no estacionariedad. Esto asegura que el tratamiento aplicado (diferenciación o detrendización) sea apropiado y que el análisis posterior no esté sesgado por errores en la varianza o la estructura de dependencia.
#ante la duda, puedes aplicar ambas transformaciones y ver cuál es la que produce una menor varianza en la variable.
# Comparación: Random Walk con Deriva vs. Tendencia Determinista
# Importar bibliotecas necesarias
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
#aplicar erroneamente la primera diferencia a la serie con tendencia determinista
#aplicamos la primera diferencia a la serie con tendencia determinista
df_tendencia['diff_tendencia'] = df_tendencia['serie_tendencia'].diff().dropna()
# Graficar la serie original y los residuos, la serie random walk con deriva y la primera diferencia
tamaños_muestrales = [100, 1000, 10000, 100000]

# Iterar sobre los tamaños muestrales y generar gráficos para cada uno
def generar_graficos_para_tamanos(tamanos_muestrales):
    for tamano in tamanos_muestrales:
        # Generar serie con tendencia determinista
        serie_tendencia = np.zeros(tamano)
        for t in range(tamano):
            et = np.random.normal(0, 1)  # Ruido blanco
            serie_tendencia[t] = beta0 + beta1 * t + et

        # Convertir la serie en un DataFrame
        df_tendencia = pd.DataFrame({'serie_tendencia': serie_tendencia})

        # Ajustar una regresión lineal para remover la tendencia
        modelo_tendencia = ols('serie_tendencia ~ np.arange(tamano)', data=df_tendencia).fit()
        residuos = modelo_tendencia.resid
        df_residuos = pd.DataFrame({'residuos': residuos})

        # Aplicar la primera diferencia
        df_tendencia['diff_tendencia'] = df_tendencia['serie_tendencia'].diff().dropna()

        # Graficar
        fig, axs = plt.subplots(2, 2, figsize=(14, 8))

        # Subplot 1: Serie con Tendencia Determinista
        axs[0, 0].plot(df_tendencia['serie_tendencia'], label='Serie con Tendencia', color='blue')
        axs[0, 0].set_title(f'Serie con Tendencia Determinista (n={tamano})')
        axs[0, 0].legend()
        axs[0, 0].set_ylabel('Valor')

        # Subplot 2: Residuos (Serie Detrendada)
        axs[0, 1].plot(df_residuos['residuos'], label='Residuos (Detrended)', color='green')
        axs[0, 1].set_title('Residuos tras Detrendización')
        axs[0, 1].legend()
        axs[0, 1].set_ylabel('Valor')

        # Subplot 3: Primera diferencia de la serie con tendencia determinista
        axs[1, 0].plot(df_tendencia['diff_tendencia'].dropna(), label='Primera Diferencia', color='red')
        axs[1, 0].set_title('Primera Diferencia')
        axs[1, 0].legend()
        axs[1, 0].set_ylabel('Valor')

        # Subplot 4: Varianzas
        var_residuos = np.var(df_residuos['residuos'])
        var_diff = np.var(df_tendencia['diff_tendencia'].dropna())
        axs[1, 1].bar(['Detrended', 'Diff'], [var_residuos, var_diff], color=['green', 'red'])
        axs[1, 1].set_title('Comparación de Varianzas')
        axs[1, 1].set_ylabel('Varianza')

        plt.tight_layout()
        plt.show()

# Llamar a la función con los tamaños muestrales
generar_graficos_para_tamanos(tamaños_muestrales)