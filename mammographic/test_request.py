import json
import requests
import pandas as pd

#Datos de entrada como en nuestro modelo (usa los mismos nombres y columnas que se entrenaron)
df = pd.DataFrame([
    {"BIRADS": 4, "Age": 65, "Shape": 3, "Margin": 5, "Density": 3}
])

# Convierte el DataFrame a JSON con el formato que espera nuestra API
data = df.to_dict(orient="records")

# Define los headers (tipo de contenido que estamos enviando y esperando)
headers = {
    "Content-Type": "application/json",
    "Accept": "application/json"
}

# Realiza la petición POST a nuestra API en la ruta /predict
response = requests.post(
    "http://localhost:8000/predict",
    data=json.dumps(data), #data en formato json (diccionario) esperado por nuestra API
    headers=headers #headers definidos, envio json y acepto json
)

# Muestra el resultado
print("Código de respuesta:", response.status_code)
print("Respuesta del servidor:")
print(response.json())