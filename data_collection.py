from flask import jsonify
import requests
import time
from pymongo import MongoClient
from datetime import datetime
import pytz
import threading

# MongoDB connection
client = MongoClient(
    "mongodb+srv://edeperezdm:rdGCIpGm2hW55OdH@cluster0.xzpq3.mongodb.net/",
    tls=True,
    tlsAllowInvalidCertificates=True
)
db = client["calidad_aire"]
collection = db["mediciones"]

# API Token
token = "cce7ef2e1dd8ff7e5c7920ec2db87c0197bc4ecf"

# Locations to monitor
locations = {
    "El Tunco": "@6966",
    "San Salvador Centro": "@7504",
    "San Salvador Este": "@6967",
    "Guatemala City": "@12515",
    "Madrid": "@5725",
    "Santa Fe, Jalisco": "@6574"
}

# Timezone
timezone = pytz.timezone('America/El_Salvador')

# Flag to control the data collection
is_collecting = False
collection_thread = None
logs = []  # Lista para almacenar los logs

# Function to collect data
def collect_data():
    global is_collecting, logs
    while is_collecting:
        for location_name, location_id in locations.items():
            url = f"https://api.waqi.info/feed/{location_id}/?token={token}"
            response = requests.get(url)

            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'ok':
                    time_of_capture = datetime.now(timezone).strftime('%Y-%m-%d %H:%M:%S')
                    aqi = data['data'].get('aqi', 0) if data['data'].get('aqi') != '-' else 0
                    pm25 = data['data']['iaqi'].get('pm25', {}).get('v', 0)
                    pm10 = data['data']['iaqi'].get('pm10', {}).get('v', 0)
                    temperature = data['data']['iaqi'].get('t', {}).get('v', 0)
                    humidity = data['data']['iaqi'].get('h', {}).get('v', 0)
                    pressure = data['data']['iaqi'].get('p', {}).get('v', 0)

                    document = {
                        "fecha_hora": time_of_capture,
                        "ubicacion": location_name,
                        "aqi": aqi,
                        "pm25": pm25,
                        "pm10": pm10,
                        "temperatura": temperature,
                        "humedad": humidity,
                        "presion": pressure
                    }
                    collection.insert_one(document)
                    log_message = f"Datos guardados para {location_name} en {time_of_capture}"
                    logs.append(log_message)  # Guardar el log
                else:
                    log_message = f"Error: No se pudo obtener los datos de {location_name}. Estado: {data['status']}."
                    logs.append(log_message)
            else:
                log_message = f"Error en la solicitud para {location_name}: {response.status_code}"
                logs.append(log_message)
        
        logs.append("Esperando 5 minutos antes de la próxima recolección...")
        time.sleep(300)  # Wait for 5 minutes

# Funciones para iniciar y detener la recolección de datos
def start_collection():
    global is_collecting, collection_thread
    if not is_collecting:
        is_collecting = True
        collection_thread = threading.Thread(target=collect_data)
        collection_thread.start()
        return "Recolección de datos iniciada"
    else:
        return "La recolección ya está en curso"

def stop_collection():
    global is_collecting
    if is_collecting:
        is_collecting = False
        return "Recolección de datos detenida"
    else:
        return "La recolección ya está detenida"

# Ruta para obtener logs
def get_logs():
    return jsonify(logs)