"""
4/6/20
File: Ataque_adversario.py
Author: Facundo Martin Cabrera
Email: cabre94@hotmail.com f.cabre94@gmail.com
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: Vamos a trucar el sistema de red neuroal inceptiion V3, diseñada por google
para clasificar imagenes en 1000 categorias diferentes.
Vamos a descargar el modelo ya entrenado, chequearlo, y despues generar una imagen
adversaria para romperla
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras import backend as K
from keras.applications.inception_v3 import InceptionV3, decode_predictions
from keras.preprocessing import image

# Cargamos el modelo
iv3 = InceptionV3()  # La primera vez tarda un rato al descargarlo

# iv3.summary()  # si lo printeamos nos mostraria las capas del modelo

# La imagen tiene que ser de 299x299 pixeles
dog = image.img_to_array(image.load_img("./Perro.jpg", target_size=(299, 299)))
beer = image.img_to_array(image.load_img("./Cerveza_Hacked.png", target_size=(299, 299)))

# kk = image.load_img("./Cerveza.jpg", target_size=(299, 299))
# plt.imshow(kk)
# plt.show()
# plt.savefig("Cerveza.png")


# V3 usa escalas de intensidades de -1 a 1, asi que hay que reescalearlo
dog = (2 / 255) * dog - 1
beer = (2 / 255) * beer - 1

# print(x.shape)  # Nos da el alto, el ancho de la imagen y la profundidad de los pixeles

# La red nos pide que le pasemos 4 cosas. La primera es cuantas imagenes le vamos a pasar, y las otras 3 son el alto,
# ancho y profundidad del pixel (¿Pero no la habiamos ajustado a 299x299)

dog = dog.reshape([1, dog.shape[0], dog.shape[1], dog.shape[2]])
beer = beer.reshape([1, beer.shape[0], beer.shape[1], beer.shape[2]])

# x = iv3.predict(dog)
y = iv3.predict(beer)

# decode_predictions(x)
decode_predictions(y)

"""
Ahora es donde la vamos a romper
"""

# Vamos a sacar el nodo de entrada y por donde la sacamos (primera y ultima capa)
inp_layer = iv3.layers[0].input
out_layer = iv3.layers[-1].output

print(1)
# Vamos a intentar de que nos prediga algo que nosotros queremos

target_class = 951  # Creo que es una referencia a un limon en V3

# Queremos maximizar la probabilidad de que la clase 951 vaya aumentando

# Le pedimos a una funcion de coste que la clase maximizada sea la que queremos
loss = out_layer[0, target_class]  # No termine de entender esta linea

# Creamos el gradiente pero no entre el error y los parametros, si no sobre la variable de entrada
grad = K.gradients(loss, inp_layer)[0]

# Dentro de gradiente ahora tenemos un  tensor que nos dice en que proporcion tenemos que variar nuestros pixeles de
# entrada Le tenemos que pasar los valores dde entrada y los valores que esperamos de salida
optimize_gradient = K.function([inp_layer, K.learning_phase()], [grad, loss])
# El learning phase es para que funcione en modo de apredizaje y no de testeo.

# Ahora el bucle

adv = np.copy(beer)

plt.imshow(adv[0].astype(np.uint8))
plt.show()
im = Image.fromarray(adv[0].astype(np.uint8))
im.save("Cerveza.png")

pert = 0.01  # Esto es una cota para cuanto puede modificar la imagen. Lo agregamos luego.
max_pert = beer + 0.01
min_pert = beer - 0.01

cost = 0.0

print(2)

while cost < 0.95:
	gr, cost = optimize_gradient([adv, 0])

	# Se lo sumamos a nuestros pixeles

	adv += gr

	adv = np.clip(adv, min_pert, max_pert)  # Esto limita cuanto puede modificar la imagen
	adv = np.clip(adv, -1, 1)

	print("Target cost: ", cost)

hacked_img = np.copy(adv)

adv /= 2
adv += 0.5
adv *= 255

plt.imshow(adv[0].astype(np.uint8))
plt.show()
im = Image.fromarray(adv[0].astype(np.uint8))
im.save("Cerveza_Hacked.png")

# Hasta aca lo unico que le habiamos pedido es que fuera modificando la imagen como para que nos de el resultado que
# queriamos. Ahora, en partes anteriores al codigo, vamos a agregar la condicion de que intente modificar la imagen
# lo menos posible, de manera que para el ojo humano no sea visible el cambio.
