import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston

# Cargamos la libreria
boston = load_boston()

# print(boston.DESCR)

# Formula para minimzar el error cuadratico medio (MCO):
# $\beta = (X^{T}X)^{-1}X^{T}Y$

X = np.array(boston.data[:, 5])
Y = np.array(boston.target)

plt.scatter(X, Y, alpha=0.3)

# Agrego columnas de 1 para el termino indeoendiente/ordenada al origen
X = np.array([np.ones(len(X)), X]).T

B = np.linalg.inv(X.T @ X) @ X.T @ Y  # El @ hace la multiplicacion matricial

plt.plot([4, 9], [B[0] + B[1] * 4, B[0] + B[1] * 9], 'r')
plt.show()
