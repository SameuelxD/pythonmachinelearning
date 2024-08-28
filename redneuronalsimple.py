import tensorflow as tf; 
import numpy as np;
import keras;
import matplotlib.pyplot as plt;

celsius = np.array([-40,-10,0,8,15,22,38], dtype=float)
fahrenheit = np.array([-40,14,32,46,59,72,100], dtype=float)

# Aprendizaje Automatico y Redes Neuronales , capas y neuronas componen a una red neuronal

capa = tf.keras.layers.Dense(units = 1, input_shape=[1])  #capa densa en keras
modelo = tf.keras.Sequential([capa])  #modelo para enseñar a procesar como aprender y manejar datos

modelo.compile(
    optimizer = tf.keras.optimizers.Adam(0.1), # Permite saber a la red neuronal como ajustar los pesos y sesgos de manera eficiente , para que mejore y no empeore , 0.1 es la tasa de aprendizaje estos le permite gradualmente ir ajustando mejor los pesos y sesgos

    loss='mean_squared_error' # funcion de perdida, considera que una poca cantidad de errores grandes es peor que una gran cantidad de errores pequeños
)

print("Comenzando entrenamiento... ")
historial = modelo.fit(celsius,fahrenheit,epochs=1000,verbose=False) # para entrenarlo usamos la funcion fit con los resultados esperados celsius y fahrenheit y ademas le decimos cuantas veces lo intente
print("Modelo entrenado")







# plt.xlabel("# Epoca") # Ver resultados de la funcion perdida , esta funcion nos dice que tan mal estan los resultados de la red en cada vuelta que dio
# plt.ylabel("Magnitud de perdida")
# plt.plot(historial.history["loss"])
# plt.show() # mostrar la funcion

print("Hagamos una prediccion!") 
entrada=np.array([100.0]) # convertir a un array de NumPy , entrada y celsius
resultado=modelo.predict(entrada)
print("El resultado es" + str(resultado) + " fahrenheit!")

print("Variables internas del modelo")
print(capa.get_weights())

# Variables internas del modelo
# [array([[1.79836]], dtype=float32), array([31.90502], dtype=float32)]


# (Entrada) * (PesoConexion) = (SesgoConexion) + (Salida) = Prediccion

# Entrada 100 Celsius
# Peso Conexion 1.798
# Sesgo Conexion 179.8  
# Salida 31.9 Fahrenheit
# 100 * 1.798 = 179.8 + 31.9 = 211.74  

