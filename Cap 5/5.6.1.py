#notas 5.6.1 Mistakenly Differencing (Overdifferencing)

# Diferenciar variables integradas las vuelve estacionarias.
# ¿Deberíamos entonces simplemente diferenciar todas las variables para garantizar la estacionariedad?
# No. Así como se puede subdiferenciar (y obtener una variable no estacionaria), también se puede sobrediferenciar,
# volviendo la variable transformada no invertible.
# Como en el cuento de Ricitos de Oro, el grado de diferenciación debe ser "justo".

# Problemas de la sobrediferenciación:
# - Induce una raíz unitaria en los términos MA del proceso → difícil de estimar.
# - Introduce autocorrelación negativa artificial.
# - Aumenta la varianza de los datos.
# - Se pierde una observación por cada diferenciación.
# - Se elimina información sobre el nivel de los datos (constantes), así como sobre las tendencias a medio/largo plazo.
#   → se privilegia la variación de corto plazo.

# Ejemplo: Diferenciar incorrectamente un proceso de ruido blanco:
# Xt = et   → proceso ya estacionario
# Diferenciando: Xt - Xt−1 = et - et−1 = Ẋt

# Resultado:
# - Ẋt es un proceso MA(1) no invertible (tiene raíz unitaria en el término MA).
# - Var(Xt) = Var(et)
# - Var(Ẋt) = Var(et - et−1) = 2Var(et)  → la varianza se duplicó.

# Conclusión:
# - El proceso ya era estacionario.
# - No se mejoró la estacionariedad, sino que se empeoró al introducir ruido.

# Además:
# - La diferenciación de primer orden introduce autocorrelación negativa.
#   → La ACF de un ruido blanco es cero en todos los rezagos.
#   → Pero para Ẋt:
# Corr(Ẋt, Ẋt−1) = Cov(Ẋt, Ẋt−1) / Var(Ẋt)
#                = E[(et − et−1)(et−1 − et−2)] / 2Var(et)
#                = −1/2 < 0

# Nota: Si el proceso fuera realmente ruido blanco, el gráfico ya mostraría estacionariedad,
# por lo que no se debería diferenciar en primer lugar.

# Ejemplo más realista: proceso tendencia-estacionario (trend-stationary):
# Xt = α + βt + et

# Diferenciando:
# Xt - Xt−1 = (α + βt + et) − (α + β(t−1) + et−1)
#           = β + et − et−1 = Ẋt

# Resultado:
# - Proceso MA(1) con coeficiente 1 sobre et−1 → hay raíz unitaria en los términos MA
# - El proceso es no invertible

#Importar bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
# Configuración de la semilla para reproducibilidad
np.random.seed(42)
# Número de observaciones
n = 100
# Generar un proceso de ruido blanco
et = np.random.normal(0, 1, n)  # Ruido blanco
# Crear la serie Xt como un proceso de ruido blanco
Xt = et  # Xt es simplemente el ruido blanco
# Calcular la primera diferencia
dXt = np.diff(Xt)  # Diferencia de primer orden
#comparar la varianza de Xt y dXt
var_Xt = np.var(Xt)
var_dXt = np.var(dXt)
# Graficar la serie original y la serie diferenciada, incluyendo la varianza en el título
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(Xt, label='Xt (Ruido Blanco)', color='blue')
plt.axhline(0, color='red', linestyle='--', label='Media = 0')
plt.title(f'Serie Original: Xt\nVarianza = {var_Xt:.2f}')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(dXt, label='Ẋt (Diferencia de Xt)', color='orange')
plt.axhline(0, color='red', linestyle='--', label='Media = 0')
plt.title(f'Serie Diferenciada: Ẋt\nVarianza = {var_dXt:.2f}')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.legend()

plt.tight_layout()
plt.show()

#graficar la autocorrelación de Xt y dXt
from statsmodels.graphics.tsaplots import plot_acf
# Graficar la autocorrelación de Xt
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plot_acf(Xt, lags=20, ax=plt.gca())
plt.title('Autocorrelación de Xt (Ruido Blanco)')
# Graficar la autocorrelación de dXt
plt.subplot(1, 2, 2)
plot_acf(dXt, lags=20, ax=plt.gca())
plt.title('Autocorrelación de Ẋt (Diferencia de Xt)')
plt.tight_layout()
plt.show()


