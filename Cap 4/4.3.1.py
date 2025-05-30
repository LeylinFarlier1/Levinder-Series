# Consideremos un proceso autorregresivo de primer orden (AR(1)):
# Xt = βXt−1 + et      (Ecuación 4.4)
# donde et es ruido blanco.

# Comportamiento del proceso según el valor de β:
# - Si β > 1, la serie crecerá sin límite (explosiva).
# - Si β < -1, la serie caerá sin límite (también explosiva).
# - Solo si |β| < 1, la serie tendrá un valor esperado constante y se estabiliza (es estacionaria).

# Reescribimos la ecuación usando el operador rezago (lag operator) L:
# Xt = βLXt + et
# (1 - βL)Xt = et

# El término (1 - βL) se conoce como "polinomio en rezagos" o "polinomio característico".

# El proceso AR es estacionario **si y solo si** las raíces del polinomio en rezagos
# tienen valor absoluto mayor que 1.

# Para hallar las raíces, reemplazamos L por z y resolvemos:
# 1 - βz = 0  ⇒  z* = 1/β

# Entonces:
# - Si |z*| > 1 ⇒ el proceso es estacionario
# - Esto equivale a decir que |β| < 1

# >>> RESUMEN:
# Un proceso AR(1) es estacionario si y solo si |β| < 1.
# Esto garantiza que la raíz del polinomio en rezagos sea mayor que 1 en magnitud.


#
# Ejemplo de un proceso AR(1) estacionario
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)
n = 100
beta = 0.5  # Coeficiente AR(1) dentro del rango estacionario (-1, 1)
# Generar serie AR(1) estacionaria
serie_ar1 = np.zeros(n)
for t in range(1, n):
    et = np.random.normal(0, 1)  # Ruido blanco
    serie_ar1[t] = beta * serie_ar1[t - 1] + et
plt.figure(figsize=(10, 5))
plt.plot(serie_ar1, marker='o', linestyle='-', color='blue', label='Serie AR(1)')
plt.axhline(0, color='red', linestyle='--', label='Media = 0')
plt.title('Serie AR(1) estacionaria: β = 0.5')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.legend()
plt.tight_layout()
plt.show()

# Ejemplo de un proceso AR(1) no estacionario
beta_no_estacionario = 1.5  # Coeficiente AR(1) fuera del rango estacionario
# Generar serie AR(1) no estacionaria
serie_ar1_no_estacionario = np.zeros(n)
for t in range(1, n):
    et = np.random.normal(0, 1)  # Ruido blanco
    serie_ar1_no_estacionario[t] = beta_no_estacionario * serie_ar1_no_estacionario[t - 1] + et
plt.figure(figsize=(10, 5))
plt.plot(serie_ar1_no_estacionario, marker='o', linestyle='-', color='orange', label='Serie AR(1) No Estacionaria')
plt.axhline(0, color='red', linestyle='--', label='Media = 0')
plt.title('Serie AR(1) no estacionaria: β = 1.5')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.legend()
plt.tight_layout()
plt.show()

#la funcion AUTOREG de statsmodels permite ajustar un modelo AR(1) a una serie temporal
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
# Crear un DataFrame con la serie AR(1) estacionaria
df_ar1 = pd.DataFrame({'serie_ar1': serie_ar1})
# Ajustar el modelo AR(1)
modelo_ar1 = AutoReg(df_ar1['serie_ar1'], lags=1).fit()
# Imprimir el resumen del modelo
print(modelo_ar1.summary())
#Del resumen, podemos ver el modulo de la raiz del polinomio característico.
print
# Obtener el módulo de la raíz del polinomio característico
raices = modelo_ar1.roots
modulos = np.abs(raices)
print("Módulo de la raíz del polinomio característico:", modulos[0])
#ahora vemos que el modulo es mayor que 1, por lo tanto el proceso es estacionario
# Ahora ajustamos el modelo AR(1) a la serie no estacionaria
df_ar1_no_estacionario = pd.DataFrame({'serie_ar1_no_estacionario': serie_ar1_no_estacionario})
modelo_ar1_no_estacionario = AutoReg(df_ar1_no_estacionario['serie_ar1_no_estacionario'], lags=1).fit()
# Imprimir el resumen del modelo no estacionario
print(modelo_ar1_no_estacionario.summary())
# Obtener el módulo de la raíz del polinomio característico no estacionario
raices_no_estacionario = modelo_ar1_no_estacionario.roots
modulos_no_estacionario = np.abs(raices_no_estacionario)
print("Módulo de la raíz del polinomio característico no estacionario:", modulos_no_estacionario[0])
# Aquí vemos que el módulo es menor que 1, por lo tanto el proceso no es estacionario