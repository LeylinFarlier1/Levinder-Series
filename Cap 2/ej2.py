# 1. What is equation of the AR(p) process corresponding to the following Stata
# estimation commands?
 #(a) arima X, ar(1/4) nocons
 #(b) arima X, ar(1 2 4) nocons
 #(c) arima X, ar(2 4) nocons
 
# The AR(p) process can be expressed as:
# (a) AR(1/4) nocons:
# X_t = e_t + φ_1 * X_{t-1} + φ_2 * X_{t-2} + φ_3 * X_{t-3} + φ_4 * X_{t-4}

from statsmodels.tsa.ar_model import AutoReg
import numpy as np
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

# Generar datos aleatorios
np.random.seed(0)
X = np.random.normal(size=1000)

# Ajustar el modelo AR(1/4) sin constante
model_a = ARIMA(X, order=(4, 0, 0), trend="n")  # trend="n" quita la constante (nocons)
result_a = model_a.fit()
print(result_a.summary())
# X_t = e_t + φ_1 * X_{t-1} + φ_2 * X_{t-2} + φ_3 * X_{t-3} + φ_4 * X_{t-4}

# Ajustar el modelo AR(1, 2, 4) sin constante
model = AutoReg(X, lags=[1, 2, 4], trend='n')  # 'n' = sin constante
results = model.fit()
print(results.summary())

#Xt = β1Xt−1 + β2Xt−2 +  + β4Xt−4 + et

# Ajustar el modelo AR(2, 4) sin constante
model_c = AutoReg(X, lags=[2, 4], trend='n')  # 'n' = sin constante
results_c = model_c.fit()
print(results_c.summary())
# Xt = β1Xt−2 + β4Xt−4 + et

###Nota: Ninguno de los coeficieentes va a ser significativo porque los datos son aleatorios y no tienen estructura temporal.