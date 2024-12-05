import numpy as np
import pandas as pd
from pymongo import MongoClient
from keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de MongoDB
MONGO_URI = "mongodb+srv://edeperezdm:rdGCIpGm2hW55OdH@cluster0.xzpq3.mongodb.net/"
DB_NAME = "calidad_aire"
COLLECTION_NAME = "mediciones"

# Cargar modelo LSTM
MODEL_PATH = "nuevo_modelo_lstm_general.keras"
model = load_model(MODEL_PATH)

# Función para categorizar calidad del aire basada en PM2.5
def categorizar_calidad_aire(pm25):
    if pm25 <= 35.4:
        return "Buena"
    elif pm25 <= 55.4:
        return "Moderada"
    else:
        return "Dañina para grupos sensibles"

# Función para obtener datos de MongoDB
def obtener_datos_mongo(start_date, end_date):
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # Consultar registros en el rango de fechas
    query = {"fecha_hora": {"$gte": start_date, "$lt": end_date}}
    data = pd.DataFrame(list(collection.find(query)))

    # Preprocesar los datos
    data['fecha_hora'] = pd.to_datetime(data['fecha_hora'])
    data.set_index('fecha_hora', inplace=True)
    data = data[['pm25', 'pm10', 'temperatura', 'humedad', 'presion']].dropna()
    return data

# Función para crear secuencias temporales
def crear_dataset(data, time_step=10):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), :]
        X.append(a)
        y.append(data[i + time_step, 0])  # PM2.5 como objetivo
    return np.array(X), np.array(y)

# Función para graficar matriz de confusión
def graficar_matriz_confusion(y_true, y_pred, labels, titulo="Matriz de Confusión"):
    # Crear la matriz de confusión
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)

    # Crear el gráfico con Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(titulo)
    plt.xlabel("Clase Predicha")
    plt.ylabel("Clase Verdadera")
    plt.tight_layout()

    # Guardar la imagen como archivo
    plt.savefig("matriz_confusion.png")
    print("Matriz de confusión guardada como 'matriz_confusion.png'")

    # Mostrar la gráfica
    plt.show()

# Función para evaluar el modelo
def evaluar_modelo(start_date, end_date):
    try:
        # Obtener datos de MongoDB
        print("Obteniendo datos de MongoDB...")
        data = obtener_datos_mongo(start_date, end_date)

        # Escalar los datos
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Crear conjuntos de datos
        X, y = crear_dataset(scaled_data)
        X = X.reshape(X.shape[0], X.shape[1], data.shape[1])

        # Dividir en entrenamiento y prueba (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Realizar predicciones
        print("Realizando predicciones...")
        y_pred = model.predict(X_test)

        # Desescalar las predicciones y valores reales
        y_test_desescalado = scaler.inverse_transform(
            np.hstack((y_test.reshape(-1, 1), np.zeros((len(y_test), X_test.shape[2] - 1))))
        )[:, 0]
        y_pred_desescalado = scaler.inverse_transform(
            np.hstack((y_pred.reshape(-1, 1), np.zeros((len(y_pred), X_test.shape[2] - 1))))
        )[:, 0]

        # Calcular métricas
        mae = mean_absolute_error(y_test_desescalado, y_pred_desescalado)
        mse = mean_squared_error(y_test_desescalado, y_pred_desescalado)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_test_desescalado, y_pred_desescalado)

        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.4f}")

        # Graficar predicciones vs valores reales
        plt.figure(figsize=(10, 6))
        plt.plot(y_test_desescalado, label="Valores Reales")
        plt.plot(y_pred_desescalado, label="Predicciones")
        plt.title("Predicciones vs Valores Reales")
        plt.xlabel("Índice")
        plt.ylabel("PM2.5")
        plt.legend()
        plt.show()

        # Categorías y matriz de confusión
        y_test_categorias = pd.Series(y_test_desescalado).apply(categorizar_calidad_aire)
        y_pred_categorias = pd.Series(y_pred_desescalado).apply(categorizar_calidad_aire)

        # Graficar la matriz de confusión
        graficar_matriz_confusion(
            y_test_categorias,
            y_pred_categorias,
            labels=["Buena", "Moderada", "Dañina para grupos sensibles"],
            titulo="Matriz de Confusión - Conjunto de Prueba"
        )

        print("\nReporte de Clasificación:")
        print(classification_report(y_test_categorias, y_pred_categorias))

    except Exception as e:
        print(f"Error durante la evaluación: {e}")

# Ejecutar evaluación con un rango de fechas
if __name__ == "__main__":
    evaluar_modelo("2024-11-01", "2024-11-30")
