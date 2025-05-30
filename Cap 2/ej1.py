#1. Using the definitions in Eqs.(2.3) and (2.4), show whether the purely random
 #process
 #Xt =β0 +et with et ∼iidN(0,1)
 #is mean stationary, variance stationary, and covariance stationary
 
 #Symbolically, for a lag-length of one(2.3),
 #Cov(Xt,Xt+1) = Cov(Xt−1,Xt) 
 #and for all lag-lengths of k (2.4),``
 #Cov(Xt,Xt+k) = Cov(Xt+1,Xt+k+1) = Cov(Xt−1,Xt+k−1).

import numpy as np

beta_0 = 0  #asumo 0 por simplicidad
T = 10000  # Número de observaciones
e_t = np.random.normal(0, 1, T)  # Generar errores iid N(0,1)
X_t = beta_0 + e_t  # Proceso Xt

print(mean_X_t := np.mean(X_t))
print(var_X_t := np.var(X_t))
import pandas as pd

def autocov_pandas(X, lag):
    X = pd.Series(X)
    if lag >= len(X):
        return np.nan
    return X[:-lag].cov(X[lag:])

# Calcular la covarianza para lag 1
cov_X_t_1 = autocov_pandas(X_t, 1)
# Calcular la covarianza para lag k
cov_X_t_k = autocov_pandas(X_t, 2)  # Por ejemplo, para k=2
# Imprimir resultados
print(f"Cov(X_t, X_t+1) (lag 1): {cov_X_t_1}")
print(f"Cov(X_t, X_t+k) (lag k): {cov_X_t_k}")
