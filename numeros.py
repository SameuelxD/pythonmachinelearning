import tensorflow as tf
import tensorflow_datasets as tfds
import math
import matplotlib.pyplot as plt
#Descargar set de datos MNIST (numeros escritos a manos y etiquetados)

datos,metadatos = tfds.load('mnist',as_supervised=True,with_info=True)

#Obtener en variables separadas los datos de entrenamiento (60k) y (10k)
datos_entrenamiento,datos_pruebas = datos['train'], datos['test']

#Funcion de Normalizacion (Pasar valores pixeles de 0-255 a 0-1) mejora la calidad de aprendizaje en la red

def normalizar(imagenes, etiquetas):
  imagenes = tf.cast(imagenes, tf.float32)
  imagenes /= 255 #Aqui se pasa de 0-255 a 0-1
  return imagenes, etiquetas


#Normalizar los datos de entrenamiento con la funcion que hicimos
datos_entrenamiento = datos_entrenamiento.map(normalizar)
datos_pruebas = datos_pruebas.map(normalizar)

#Agregar a cache (usar memoria en lugar de disco, entrenamiento mas rapido)
datos_entrenamiento = datos_entrenamiento.cache()
datos_pruebas = datos_pruebas.cache()

clases = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

#Codigo para mostrar imagenes del set, no es necesario ejecutarlo, solo imprime unos numeros :)

# plt.figure(figsize=(10,10))

# for i, (imagen, etiqueta) in enumerate(datos_entrenamiento.take(25)):
#   imagen = imagen.numpy().reshape((28,28))
#   plt.subplot(5,5,i+1)
#   plt.xticks([])
#   plt.yticks([])
#   plt.grid(False)
#   plt.imshow(imagen, cmap=plt.cm.binary)
#   plt.xlabel(clases[etiqueta])

# plt.show()  

# Crear el modelo (Modelo denso , regular , sin redes convolucionales todavia)

# modelo = tf.keras.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28,28,1)), # recibe la imagen de 28x28 pixeles para un solo canal a blanco y negro

#   tf.keras.layers.Dense(units=50,activation='relu'), # capas ocultas con 50 neuronas cada una con activacion relu
#   tf.keras.layers.Dense(units=50,activation='relu'),

#   tf.keras.layers.Dense(10, activation='softmax') # capa de salida con activacion softmax para que nos de la prediccion
# ])


# Crear el modelo (Modelo convolucional , con capas de convolucion y agrupacion )

modelo = tf.keras.Sequential([
  # Creamos una red convolucional de 32 nucleos de 3x3
  tf.keras.layers.Conv2D(32,(3,3), input_shape=(28,28,1), activation='relu'), # recibe la imagen de 28x28 pixeles para un solo canal a blanco y negro
  tf.keras.layers.MaxPooling2D(2,2), # capa de agrupacion maxima de 2x2
  # Creamos una red convolucional de 32 nucleos de 3x3
  
  # capa de convolucion con 64 filtros y una agrupacion maxima
  tf.keras.layers.Conv2D(64,(3,3), input_shape=(28,28,1), activation='relu'), # recibe la imagen de 28x28 pixeles para un solo canal a blanco y negro
  tf.keras.layers.MaxPooling2D(2,2), # capa de agrupacion maxima de 2x2

  tf.keras.layers.Flatten(), # convierte nuestro resultado de imagen cuadrada en un vector simple que podemos usar con las capas regulares de adelante

  tf.keras.layers.Dense(units=100,activation='relu'), # capa oculta con 100 neuronas con activacion relu
  
  tf.keras.layers.Dense(10, activation='softmax') # capa de salida con activacion softmax para que nos de la prediccion
])

#Compilar el modelo
modelo.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

#Los numeros de datos de entrenamiento y pruebas (60k y 10k)
num_datos_entrenamiento = metadatos.splits["train"].num_examples
num_datos_pruebas = metadatos.splits["test"].num_examples

#Trabajar por lotes
TAMANO_LOTE=32

#Shuffle y repeat hacen que los datos esten mezclados de manera aleatoria
#para que el entrenamiento no se aprenda las cosas en orden
datos_entrenamiento = datos_entrenamiento.repeat().shuffle(num_datos_entrenamiento).batch(TAMANO_LOTE)
datos_pruebas = datos_pruebas.batch(TAMANO_LOTE)

historial = modelo.fit(
  datos_entrenamiento,
  epochs=60,  # entrenamiento por 60 epocas
  steps_per_epoch=math.ceil(num_datos_entrenamiento/TAMANO_LOTE)
)