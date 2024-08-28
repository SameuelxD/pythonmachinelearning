import tensorflow as tf; 
import numpy as np;
import keras;
import matplotlib.pyplot as plt;

# Red neuronal con 3 capas y 3 neuronas

celsius = np.array([-40,-10,0,8,15,22,38], dtype=float)
fahrenheit = np.array([-40,14,32,46,59,72,100], dtype=float)

oculta1=tf.keras.layers.Dense(units=3,input_shape=[1])
oculta2=tf.keras.layers.Dense(units=3)
salida=tf.keras.layers.Dense(units=1)
modelo=tf.keras.Sequential([oculta1,oculta2,salida])

modelo.compile(
    optimizer = tf.keras.optimizers.Adam(0.1), 
    loss='mean_squared_error' 
)

print("Comenzando entrenamiento... ")
historial = modelo.fit(celsius,fahrenheit,epochs=1000,verbose=False) 
print("Modelo entrenado")

#Exportar el modelo en formato h5
modelo.save('celsius_a_fahrenheit.h5')


#comando  tensorflowjs_converter --input_format keras celsius_a_fahrenheit.h5 celsiusafahrenheit

# plt.xlabel("# Epoca") 
# plt.ylabel("Magnitud de perdida")
# plt.plot(historial.history["loss"])
# plt.show()

# print("Hagamos una prediccion!") 
# entrada=np.array([100.0]) 
# resultado=modelo.predict(entrada)
# print("El resultado es" + str(resultado) + " fahrenheit!")

# print("Variables internas del modelo")
# print(oculta1.get_weights())
# print(oculta2.get_weights())
# print(salida.get_weights())

