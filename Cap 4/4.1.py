 #¿Qué es la estacionariedad?

# Una serie Xt es "estacionaria en media" (mean stationary) si:
# E(Xt) = μ
# es decir, la media es constante para todos los periodos t.

# Es "estacionaria en varianza" (variance stationary) si:
# Var(Xt) = σ²
# la varianza es constante en el tiempo.

# Una serie es "estacionaria en autocovarianza" (auto-covariance stationary) si:
# Cov(Xt, Xt+k) = Cov(Xt+a, Xt+k+a)
# Es decir, la covarianza entre Xt y Xt+k depende solo del desfase k,
# no del tiempo t en sí.

# Lo importante es la distancia entre las observaciones, no el punto en el tiempo.

# Ejemplo:
# Cov(X1, X4) = Cov(X5, X8) = Cov(X11, X14) = Cov(Xt, Xt+3)
# Aquí la distancia k = 3 se mantiene constante, por lo tanto la covarianza también

#seteo la media de la variable y su dispersion. como a su vezun ruido blanco que la genere
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)
n = 100
mu = 0
sigma = 10

# Generar serie estacionaria: media y varianza constantes, ruido blanco
serie = mu + sigma * np.random.randn(n)

plt.figure(figsize=(10, 5))
plt.plot(serie, marker='o', linestyle='-', color='blue', label='Serie')
plt.fill_between(range(n), mu - sigma, mu + sigma, color='green', alpha=0.2, label='±1σ')
plt.axhline(mu, color='red', linestyle='--', label='Media')
plt.title('Serie estacionaria: media y varianza constantes')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.legend()
plt.tight_layout()
plt.show()