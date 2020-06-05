"""
4/6/20
File: Gradient_Descent.py
Author: Facundo Martin Cabrera
Email: cabre94@hotmail.com f.cabre94@gmail.com
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: 
"""

import matplotlib.pyplot as plt
import numpy as np


def func(var):
	return np.sin(1 / 2 * var[0]**2 - 1 / 4 * var[1]**2 + 3) * np.cos(2 * var[0] + 1 - np.e**var[1])


res = 100

_X = np.linspace(-2.5, 2.5, res)
_Y = np.linspace(-2.5, 2.5, res)

_Z = np.zeros((res, res))

for ix, x in enumerate(_X):
	for iy, y in enumerate(_Y):
		_Z[iy, ix] = func([x, y])

# plt.contour(_X,_Y,_Z,100)
plt.contourf(_X, _Y, _Z, 100)
plt.colorbar()

Theta = np.random.rand(2) * 4 - 2

h = 0.001
lr = 0.001

plt.plot(Theta[0], Theta[1], 'ow')

grad = np.zeros(2)

for _ in range(10000):

	for it, th in enumerate(Theta):
		T = np.copy(Theta)
		T[it] = T[it] + h

		deriv = (func(T) - func(Theta)) / h

		grad[it] = deriv

	Theta = Theta - lr * grad
	if _ % 100 == 0:
		plt.plot(Theta[0], Theta[1], '.r')

plt.plot(Theta[0], Theta[1], 'og')
plt.show()
