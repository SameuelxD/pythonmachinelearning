import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import math
import os

# Descargar y cargar los datos
datos, metadatos = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
datos_entrenamiento, datos_pruebas = datos['train'], datos['test']
nombres_clases = metadatos.features['label'].names
print(nombres_clases)

# Normalizar datos originales de 0 a 255 para convertirlos de 0 a 1
def normalizer(imagenes, etiquetas):
    imagenes = tf.cast(imagenes, tf.float32) / 255.0
    return imagenes, etiquetas

# Normalizar los datos de entrenamiento y pruebas con la función
datos_entrenamiento = datos_entrenamiento.map(normalizer)
datos_pruebas = datos_pruebas.map(normalizer)

# Agregar a cache (usar memoria en lugar de disco, entrenamiento más rápido)
datos_entrenamiento = datos_entrenamiento.cache()
datos_pruebas = datos_pruebas.cache()

# Definir el tamaño del lote
tamano_lote = 32

# Configurar el pipeline de datos
datos_entrenamiento = datos_entrenamiento.shuffle(buffer_size=10000).batch(tamano_lote).prefetch(tf.data.AUTOTUNE)
datos_pruebas = datos_pruebas.batch(tamano_lote).prefetch(tf.data.AUTOTUNE)

# Crear el modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),  # Aplana las imágenes de 28x28x1 a un vector de 784 elementos
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')  # Para clasificación de 10 clases
])

# Compilar el modelo
modelo.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Entrenamiento
num_ej_entrenamiento = metadatos.splits["train"].num_examples
num_ej_pruebas = metadatos.splits["test"].num_examples

# Entrenar el modelo
historial = modelo.fit(
    datos_entrenamiento,
    epochs=5,
    steps_per_epoch=math.ceil(num_ej_entrenamiento / tamano_lote),
    validation_data=datos_pruebas
)

# Graficar la pérdida durante el entrenamiento
plt.figure()
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"], label='Pérdida de entrenamiento')
plt.plot(historial.history["val_loss"], label='Pérdida de validación')
plt.legend()
plt.show()

# Evaluar el modelo
for imagenes_prueba, etiquetas_prueba in datos_pruebas.take(1):
    imagenes_prueba = imagenes_prueba.numpy()
    etiquetas_prueba = etiquetas_prueba.numpy()
    predicciones = modelo.predict(imagenes_prueba)

def graficar_imagen(i, arr_predicciones, etiquetas_reales, imagenes):
    arr_predicciones, etiqueta_real, img = arr_predicciones[i], etiquetas_reales[i], imagenes[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img[..., 0], cmap=plt.cm.binary)

    etiqueta_prediccion = np.argmax(arr_predicciones)
    color = 'blue' if etiqueta_prediccion == etiqueta_real else 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(nombres_clases[etiqueta_prediccion],
                                         100 * np.max(arr_predicciones),
                                         nombres_clases[etiqueta_real]),
               color=color)

def graficar_valor_arreglo(i, arr_predicciones, etiqueta_real):
    arr_predicciones, etiqueta_real = arr_predicciones[i], etiqueta_real[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    grafica = plt.bar(range(10), arr_predicciones, color="#777777")
    plt.ylim([0, 1])
    etiqueta_prediccion = np.argmax(arr_predicciones)

    grafica[etiqueta_prediccion].set_color('red')
    grafica[etiqueta_real].set_color('blue')

# Graficar predicciones y valores
filas = 5
columnas = 5
num_imagenes = filas * columnas
plt.figure(figsize=(2 * 2 * columnas, 2 * filas))
for i in range(num_imagenes):
    plt.subplot(filas, 2 * columnas, 2 * i + 1)
    graficar_imagen(i, predicciones, etiquetas_prueba, imagenes_prueba)
    plt.subplot(filas, 2 * columnas, 2 * i + 2)
    graficar_valor_arreglo(i, predicciones, etiquetas_prueba)
plt.show()

imagen = imagenes_prueba[10]
imagen = np.array([imagen])
prediccion = modelo.predict(imagen)

print("Prediccion: " + nombres_clases[np.argmax(prediccion[0])])

#Exportacion del modelo a h5
modelo.save('modelo_exportado.h5')

# Crear un directorio para el modelo convertido
if not os.path.exists('tfjs_target_dir'):
    os.makedirs('tfjs_target_dir')

# Ejecutar el comando de conversión
# !tensorflowjs_converter --input_format keras modelo_exportado.h5 tfjs_target_dir