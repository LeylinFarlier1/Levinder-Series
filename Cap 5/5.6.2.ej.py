import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

n = 1000

e_t= np.random.normal(0, 2, n)

x_t = np.zeros(n)
for t in range(1, n):
    x_t[t] = e_t[t]   
    

# Calculo los estadisticos de la serie x_t
mean_x_t = np.mean(x_t)
var_x_t = np.var(x_t)
std_x_t = np.std(x_t) 

print(f"Media de x_t: {mean_x_t:.2f}, Varianza de x_t: {var_x_t:.2f}, Desviación Estándar de x_t: {std_x_t:.2f}")      

plot_acf(x_t, lags=5)
plot_pacf(x_t, lags=5)
plt.show()

xt_diff = np.diff(x_t)
mean_xt_diff = np.mean(xt_diff)
var_xt_diff = np.var(xt_diff)

print(f"Media de la primera diferencia de x_t: {mean_xt_diff:.2f}, Varianza de la primera diferencia de x_t: {var_xt_diff:.2f}")
plot_pacf(xt_diff, lags=5)
plot_acf(xt_diff, lags=5)
plt.show()


from statsmodels.tsa.arima.model import ARIMA


modelo_arima = ARIMA(x_t, order=(0, 1, 1)).fit()
print(modelo_arima.summary())