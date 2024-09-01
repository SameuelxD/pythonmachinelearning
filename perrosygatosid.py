import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import TensorBoard

# Ruta donde descomprimiste el archivo
data_dir = 'C:/Users/USER/tensorflow_datasets/kagglecatsanddogs_5340/PetImages'

# Configuración del generador de datos para imágenes en blanco y negro
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Uso de validación con un 20% de los datos de entrenamiento
)

# Generador de datos para entrenamiento
train_generator = train_datagen.flow_from_directory(
    directory=data_dir,
    target_size=(150, 150),
    batch_size=32,
    color_mode='grayscale',  # Convertir las imágenes a blanco y negro
    class_mode='binary',
    subset='training'  # Usar la parte de entrenamiento
)

# Listas para almacenar imágenes y etiquetas
x = []
y = []

try:
    # Iterar sobre el generador para recoger todos los batches
    for data_batch, labels_batch in train_generator:
        x.extend(data_batch)
        y.extend(labels_batch)
        
        # Romper el loop si se ha recogido todo el dataset
        if len(x) >= train_generator.samples:
            break
except Exception as e:
    print(f"Error occurred: {e}")

# Convertir listas a numpy arrays
x = np.array(x)
y = np.array(y)

# Crear un DataFrame para mapear índices a etiquetas
class_indices = train_generator.class_indices
class_labels = {v: k for k, v in class_indices.items()}

# Configurar la visualización
fig, axes = plt.subplots(5, 5, figsize=(12, 12))  # Tamaño de la figura ajustado para 5x5
axes = axes.flatten()

for img, label, ax in zip(x[:25], y[:25], axes):
    # Las imágenes ya están en escala de grises después de la conversión
    ax.imshow(img.squeeze(), cmap='gray')  # Usar colormap 'gray' para mostrar imágenes en blanco y negro
    ax.axis('off')
    # Convertir la etiqueta binaria a texto
    label_text = class_labels[int(label)]
    ax.set_title(label_text, fontsize=8, pad=2)

plt.tight_layout()
plt.show()

# Imprimir el tamaño de las listas de imágenes y etiquetas para verificar
print(f'Número de imágenes almacenadas: {len(x)}')
print(f'Número de etiquetas almacenadas: {len(y)}')

# Asegurarse de que las imágenes están normalizadas
x = np.array(x).astype(float) / 255

# Modelo Denso ajustado para imágenes en blanco y negro
modeloDenso = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(150, 150, 1)),  # Ajustar el tamaño de entrada para un solo canal
    tf.keras.layers.Dense(150, activation='relu'),
    tf.keras.layers.Dense(150, activation='relu'), # Dos capas densas de 150 cada una
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Modelo red neuronal convolucional
modeloConvolucional = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,1)), # 3 pares de capas convolucionales y de agrupación máxima
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),  # Aplanar para las capas densas
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Modelo convolucional con técnica dropout
modeloDropout = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,1)), # 3 pares de capas convolucionales y de agrupación máxima
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.5),  # Técnica de dropout para evitar el sobreajuste
    tf.keras.layers.Flatten(),  # Aplanar para las capas densas
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilar los modelos
modeloDenso.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

modeloConvolucional.compile(optimizer='adam',
                            loss='binary_crossentropy',
                            metrics=['accuracy'])

modeloDropout.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

#La variable de tensorboard se envia en el arreglo de "callbacks" (hay otros tipos de callbacks soportados)
#En este caso guarda datos en la carpeta indicada en cada epoca, de manera que despues
#Tensorboard los lee para hacer graficas
tensorboardDenso = TensorBoard(log_dir='logs/denso')
modeloDenso.fit(x, y, batch_size=32,
                validation_split=0.15,
                epochs=100,
                callbacks=[tensorboardDenso])

# Resumen de cada modelo para verificar
# print("Resumen del Modelo Denso:")
# modeloDenso.summary()

# print("\nResumen del Modelo Convolucional:")
# modeloConvolucional.summary()

# print("\nResumen del Modelo Convolucional con Dropout:")
# modeloDropout.summary()
