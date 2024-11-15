import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
import matplotlib.pyplot as plt
from pymongo import MongoClient
import os

# Obtener el directorio base del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cargar el modelo previamente entrenado
model_path = os.path.join(BASE_DIR, "nuevo_modelo_lstm_general.keras")
model = load_model(model_path)

# Función de predicción
def realizar_prediccion():
    try:
        # Ubicaciones a predecir
        ubicaciones = ["El Tunco", "San Salvador Centro", "San Salvador Este"]

        # Crear un DataFrame vacío para almacenar las predicciones
        todas_predicciones = pd.DataFrame()

        # Conexión a MongoDB
        client = MongoClient("mongodb+srv://edeperezdm:rdGCIpGm2hW55OdH@cluster0.xzpq3.mongodb.net/")
        db = client["calidad_aire"]
        collection = db["mediciones"]

        # Filtrar los datos de interés
        data = pd.DataFrame(list(collection.find()))
        data['fecha_hora'] = pd.to_datetime(data['fecha_hora'])
        data.set_index('fecha_hora', inplace=True)

        # Escalar los datos
        scaler = MinMaxScaler(feature_range=(0, 1))
        tiempo_paso = 10  # Definir la cantidad de pasos de tiempo

        # Preparar las predicciones para cada ubicación
        for ubicacion in ubicaciones:
            data_ubicacion = data[data['ubicacion'] == ubicacion].dropna()
            features = data_ubicacion[['pm25', 'pm10', 'temperatura', 'humedad', 'presion']]
            scaled_data = scaler.fit_transform(features)
            last_data = scaled_data[-tiempo_paso:].reshape((1, tiempo_paso, features.shape[1]))

            predicciones = []
            for _ in range(5):
                prediccion = model.predict(last_data)
                predicciones.append(prediccion[0][0])
                nueva_prediccion = np.hstack((prediccion, np.zeros((1, features.shape[1] - 1))))
                nueva_prediccion = nueva_prediccion.reshape(1, 1, features.shape[1])
                last_data = np.append(last_data[:, 1:, :], nueva_prediccion, axis=1)

            predicciones_invertidas = scaler.inverse_transform(
                np.hstack((np.array(predicciones).reshape(-1, 1), np.zeros((5, features.shape[1] - 1))))
            )[:, 0]

            fecha_prediccion = features.index[-1] + timedelta(days=1)
            fechas_predicciones = [fecha_prediccion + timedelta(days=i) for i in range(5)]

            # Cambiar el formato de las fechas a "MM-DD"
            fechas_formateadas = [fecha.strftime("%m-%d") for fecha in fechas_predicciones]

            predicciones_df = pd.DataFrame({
                'Fecha': fechas_formateadas,
                'Predicción PM2.5': predicciones_invertidas,
                'Ubicación': ubicacion
            })

            todas_predicciones = pd.concat([todas_predicciones, predicciones_df], ignore_index=True)

        # Visualizar las predicciones para las tres ubicaciones
        plt.figure(figsize=(12, 6))
        for ubicacion in ubicaciones:
            subset = todas_predicciones[todas_predicciones['Ubicación'] == ubicacion]
            plt.plot(subset['Fecha'], subset['Predicción PM2.5'], marker='o', label=ubicacion)

        plt.title('Predicción de PM2.5 para los próximos 5 días')
        plt.xlabel('Fecha')
        plt.ylabel('Concentración PM2.5 (µg/m³)')
        plt.axvline(x=fechas_formateadas[0], color='gray', linestyle='--', label='Fecha de predicción')
        plt.legend()
        plt.xticks(fechas_formateadas)  # Establecer manualmente las etiquetas del eje X
        plt.tight_layout()

        # Guardar la gráfica como imagen en una ruta relativa
        save_path = os.path.join(BASE_DIR, "prediccion_pm25.png")
        plt.savefig(save_path)
        print(f"Imagen guardada en: {save_path}")

        # Cerrar la figura
        plt.close()

        # Devolver el DataFrame con las predicciones
        return todas_predicciones

    except Exception as e:
        print(f"Error durante la predicción: {e}")

# Ejecutar la función de predicción si se ejecuta directamente
if __name__ == "__main__":
    realizar_prediccion()
