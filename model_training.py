import pandas as pd
import numpy as np
from pymongo import MongoClient
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import Callback
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

training_logs = []

# Callback personalizado para capturar los logs de entrenamiento
class TrainingLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        log_message = f"Epoch {epoch + 1}/{self.params['epochs']} - loss: {logs['loss']:.4f}"
        training_logs.append(log_message)

# Función para el entrenamiento del modelo
def train_model():
    global training_logs
    try:
        training_logs = []

        # Conexión a MongoDB
        client = MongoClient("mongodb+srv://edeperezdm:rdGCIpGm2hW55OdH@cluster0.xzpq3.mongodb.net/")
        db = client["calidad_aire"]
        collection = db["mediciones"]

        # Obtener datos de MongoDB
        training_logs.append("Obteniendo datos de MongoDB...")
        data = pd.DataFrame(list(collection.find()))

        # Preprocesamiento
        data['fecha_hora'] = pd.to_datetime(data['fecha_hora'])
        data.set_index('fecha_hora', inplace=True)
        features = data[['pm25', 'pm10', 'temperatura', 'humedad', 'presion']].dropna()

        # Escalado
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(features)

        # Crear dataset con TensorFlow
        def crear_dataset(data, time_step=10):
            X, y = [], []
            for i in range(len(data) - time_step - 1):
                X.append(data[i:(i + time_step)])
                y.append(data[i + time_step, 0])  # PM2.5 como objetivo
            return np.array(X), np.array(y)

        tiempo_paso = 10
        X, y = crear_dataset(scaled_data, tiempo_paso)
        X = X.reshape(X.shape[0], X.shape[1], features.shape[1])

        # Dataset de TensorFlow para optimización
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)

        # Crear modelo
        training_logs.append("Creando y compilando el modelo...")
        model = Sequential([
            LSTM(32, return_sequences=True, input_shape=(X.shape[1], features.shape[1])),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Entrenar el modelo
        training_logs.append("Entrenando el modelo...")
        model.fit(dataset, epochs=10, verbose=0, callbacks=[TrainingLogger()])

        # Evaluar el modelo
        mse_entrenamiento = model.evaluate(dataset, verbose=0)
        training_logs.append(f"Error cuadrático medio (MSE): {mse_entrenamiento}")

        # Guardar el modelo
        model.save("nuevo_modelo_lstm_general.keras")
        training_logs.append("Modelo guardado exitosamente.")
        return "Entrenamiento completado exitosamente."

    except Exception as e:
        training_logs.append(f"Error durante el entrenamiento: {str(e)}")
        return f"Error: {str(e)}"

# Función para obtener logs
def get_training_logs():
    return training_logs
