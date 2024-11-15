import os
import sys
import locale
import re
from flask import Flask, render_template, jsonify, send_file
from data_collection import start_collection, stop_collection, get_logs as get_collection_logs
from model_training import train_model, get_training_logs
import subprocess
import threading

app = Flask(__name__)

# Configurar la codificación predeterminada
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

# Obtener el directorio base del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Variables globales para controlar las predicciones
is_predicting = False
prediction_logs = []  # Para almacenar los logs de predicción
prediction_message = ""  # Variable para almacenar el mensaje de éxito

# Ruta principal para mostrar los botones y los logs
@app.route('/')
def index():
    return render_template('index.html')

# Rutas para controlar la recolección de datos
@app.route('/start')
def start():
    return start_collection()

@app.route('/stop')
def stop():
    return stop_collection()

# Ruta para obtener los logs de recolección de datos
@app.route('/logs')
def logs():
    return get_collection_logs()

# Ruta para iniciar el entrenamiento del modelo
@app.route('/train', methods=["POST"])
def train():
    try:
        train_model()  # Llamada para entrenar el modelo
        return jsonify({"message": "Entrenamiento completado exitosamente"})
    except Exception as e:
        return jsonify({"message": f"Error durante el entrenamiento: {str(e)}"}), 500

# Ruta para obtener los logs del entrenamiento
@app.route('/training_logs')
def training_logs():
    return {"logs": get_training_logs()}

# Ruta para iniciar la predicción del modelo
@app.route('/predict', methods=["POST"])
def predict():
    global is_predicting
    if not is_predicting:
        is_predicting = True
        prediction_logs.clear()  # Limpiar logs anteriores
        prediction_message = ""  # Limpiar el mensaje de predicción

        # Ejecutar predict.py como un subproceso
        threading.Thread(target=ejecutar_predict_script).start()
        return jsonify({"message": "Predicción iniciada con éxito"})
    else:
        return jsonify({"message": "Ya se está realizando una predicción"}), 400

# Ruta para obtener los logs de predicción
@app.route('/prediction_logs')
def get_prediction_logs():
    global prediction_message
    return {"logs": prediction_logs, "message": prediction_message}

# Ruta para servir la imagen de la predicción
@app.route('/prediccion_imagen')
def prediccion_imagen():
    image_path = os.path.join(BASE_DIR, "prediccion_pm25.png")
    return send_file(image_path, mimetype='image/png')

# Función para ejecutar predict.py como un subproceso
def ejecutar_predict_script():
    global is_predicting, prediction_message
    try:
        # Determina el intérprete de Python según el sistema operativo
        python_interpreter = "python3" if os.name != "nt" else sys.executable

        # Ruta al script predict.py
        script_path = os.path.join(BASE_DIR, "predict.py")

        # Ejecutar predict.py y capturar la salida con la codificación UTF-8
        result = subprocess.run(
            [python_interpreter, script_path],
            capture_output=True,
            text=True,
            encoding='utf-8'  # Especifica la codificación UTF-8
        )
        if result.returncode == 0:
            # Procesar los logs para eliminar líneas con "ETA: 0s" y caracteres no deseados
            output_cleaned = re.sub(r'.*ETA:.*', '', result.stdout)  # Eliminar líneas con "ETA:"
            prediction_logs.append(f"Predicción realizada con éxito: \n{output_cleaned}")
            prediction_message = "Predicción completada exitosamente."
        else:
            prediction_logs.append(f"Error durante la predicción: {result.stderr}")
            prediction_message = "Error durante la predicción."
    except Exception as e:
        prediction_logs.append(f"Error al ejecutar predict.py: {str(e)}")
        prediction_message = "Error al ejecutar predict.py."
    finally:
        is_predicting = False

if __name__ == '__main__':
    app.run(host='localhost', port=5000)
