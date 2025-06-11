# ------------------------------------------------------------
# Notas: 6.1.3 Additive Seasonality
# ------------------------------------------------------------

# La estacionalidad aditiva en modelos ARIMA se introduce al agregar un término 
# AR (autorregresivo) o MA (media móvil) con rezago igual a la frecuencia estacional.

# Ejemplo: Para datos trimestrales (frecuencia 4), se agregan:
# - Término AR(4): captura persistencia estacional.
# - Término MA(4): captura choques estacionales.

# Modelo AR estacional (trimestral):
# Xt = β₄ * Xt−4 + et
# Notación operador rezago: (1 − L⁴β₄)Xt = et
# → Modelo estacionario de ruido blanco con componente AR estacional.

# Modelo MA estacional (trimestral):
# Xt = ut + γ₄ * ut−4
# Notación operador rezago: Xt = (1 + L⁴γ₄)ut

# L⁴: operador de rezago estacional (lag 4).
# β₄, γ₄: parámetros estacionales del modelo.

#importo las bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
# Configuración de la semilla para reproducibilidad
np.random.seed(42)
# Número de observaciones
n = 200
# Generar un proceso de ruido blanco
et = np.random.normal(0, 1, n)  # Ruido blanco
# Crear la serie Xt como un proceso de ruido blanco con un componente estacional
Xt = et.copy()  # Xt es simplemente el ruido blanco
# Agregar un componente estacional MA (por ejemplo, trimestral)
for t in range(4, n):
    Xt[t] += 0.5 * et[t - 4]  # Agregar un efecto estacional cada 4 periodos

#Creo otra serie Xt con un componente AR estacional
Xt_ar = et.copy()  # Xt_ar es simplemente el ruido blanco
# Agregar un componente estacional AR (por ejemplo, trimestral)
for t in range(4, n):
    Xt_ar[t] += 0.3 * Xt_ar[t - 4]  # Agregar un efecto estacional cada 4 periodos

# Graficar las series
fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True) # Ajustar figsize para más espacio vertical

# Fila 1: Serie original de ruido blanco (et)
axs[0].plot(et, marker='.', linestyle='-', color='green', label='Ruido Blanco Original (et)')
axs[0].axhline(0, color='grey', linestyle='--', linewidth=0.8)
axs[0].set_title('Serie Original de Ruido Blanco (et)')
axs[0].set_ylabel('Valor')
axs[0].legend()
axs[0].grid(True, linestyle=':', alpha=0.7)

# Fila 2: Series Xt (MA estacional) y Xt_ar (AR estacional) juntas
axs[1].plot(Xt, marker='o', markersize=4, linestyle='-', color='blue', label='Xt con MA Estacional (Xt = et + 0.5*et-4)')
axs[1].plot(Xt_ar, marker='x', markersize=5, linestyle='--', color='orange', label='Xt_ar con AR Estacional (Xt_ar = et + 0.3*Xt_ar-4)')
axs[1].axhline(0, color='grey', linestyle='--', linewidth=0.8)
axs[1].set_title('Series con Componentes Estacionales Aditivos')
axs[1].set_xlabel('Tiempo')
axs[1].set_ylabel('Valor')
axs[1].legend()
axs[1].grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()

#veo los acf autocorrelaciones de las series
from statsmodels.graphics.tsaplots import plot_acf

fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# ACF de Xt (MA estacional)
plot_acf(Xt, lags=40, ax=axs[0], alpha=0.05)
axs[0].set_title('ACF de Xt (MA Estacional)')

# ACF de Xt_ar (AR estacional)
plot_acf(Xt_ar, lags=40, ax=axs[1], alpha=0.05)
axs[1].set_title('ACF de Xt_ar (AR Estacional)')

# ACF de la serie original de ruido blanco
plot_acf(et, lags=40, ax=axs[2], alpha=0.05)
axs[2].set_title('ACF de Ruido Blanco Original (et)')

plt.tight_layout()
plt.show()

#ploteo los PACF parciales de las series
from statsmodels.graphics.tsaplots import plot_pacf
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
# PACF de Xt (MA estacional)
plot_pacf(Xt, lags=40, ax=axs[0], alpha=0.05)
axs[0].set_title('PACF de Xt (MA Estacional)')
# PACF de Xt_ar (AR estacional)
plot_pacf(Xt_ar, lags=40, ax=axs[1], alpha=0.05)
axs[1].set_title('PACF de Xt_ar (AR Estacional)')
# PACF de la serie original de ruido blanco
plot_pacf(et, lags=40, ax=axs[2], alpha=0.05)
axs[2].set_title('PACF de Ruido Blanco Original (et)')
plt.tight_layout()
plt.show()
