import pandas as pd
import numpy as np
from pymongo import MongoClient
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import Callback
from sklearn.preprocessing import MinMaxScaler

training_logs = []  # Cambiar el nombre de la lista de logs

# Callback personalizado para capturar los logs de entrenamiento
class TrainingLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        log_message = f"Epoch {epoch + 1}/{self.params['epochs']} - loss: {logs['loss']:.4f}"
        training_logs.append(log_message)

# Función para el entrenamiento del modelo
def train_model():
    global training_logs
    try:
        training_logs = []  # Reiniciar los logs antes de iniciar el entrenamiento

        # Conexión a MongoDB
        client = MongoClient("mongodb+srv://edeperezdm:rdGCIpGm2hW55OdH@cluster0.xzpq3.mongodb.net/")
        db = client["calidad_aire"]
        collection = db["mediciones"]

        # Obtener datos de MongoDB
        training_logs.append("Obteniendo datos de MongoDB...")
        data = pd.DataFrame(list(collection.find()))

        # Preprocesamiento de datos
        data['fecha_hora'] = pd.to_datetime(data['fecha_hora'])
        data.set_index('fecha_hora', inplace=True)

        # Filtrar los datos de interés (PM2.5, PM10, temperatura, humedad, presión)
        features = data[['pm25', 'pm10', 'temperatura', 'humedad', 'presion']].dropna()

        # Escalar los datos
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(features)

        # Crear conjuntos de datos para entrenamiento
        def crear_dataset(data, time_step=1):
            X, y = [], []
            for i in range(len(data) - time_step - 1):
                a = data[i:(i + time_step), :]
                X.append(a)
                y.append(data[i + time_step, 0])  # PM2.5 como objetivo
            return np.array(X), np.array(y)

        # Usar 10 registros para predecir el siguiente
        tiempo_paso = 10
        X, y = crear_dataset(scaled_data, tiempo_paso)

        # Reshape para LSTM [samples, time steps, features]
        X = X.reshape(X.shape[0], X.shape[1], features.shape[1])

        # Crear y compilar el modelo LSTM
        training_logs.append("Creando y compilando el modelo...")
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], features.shape[1])))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1))  # Salida para PM2.5

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Entrenar el modelo con el callback
        training_logs.append("Entrenando el modelo...")
        model.fit(X, y, epochs=10, batch_size=1, verbose=0, callbacks=[TrainingLogger()])

        # Evaluar el modelo
        mse_entrenamiento = model.evaluate(X, y, verbose=0)
        training_logs.append(f"Error cuadrático medio (MSE) en el entrenamiento: {mse_entrenamiento}")

        # Guardar el modelo en formato nativo de Keras
        model.save("nuevo_modelo_lstm_general.keras")
        training_logs.append("Modelo entrenado y guardado exitosamente en formato .keras.")

        return "Entrenamiento completado exitosamente."

    except Exception as e:
        training_logs.append(f"Error durante el entrenamiento: {str(e)}")
        return f"Error: {str(e)}"

# Función para obtener los logs del entrenamiento
def get_training_logs():
    return training_logs
