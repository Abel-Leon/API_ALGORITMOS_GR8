#importación de librerias necesarias
from flask import Flask, jsonify, request
import pandas as pd
import pickle

# función para cargar modelo
def load_object(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

app = Flask(__name__)

# Carga el modelo de regresion logística entrenado (ya guardado)
modelo_lr = load_object("modelo_regresion_logistica.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        req_json = request.get_json() #Lee los datos que llegan como JSON en la petición POST (desde el archivo test_request.py)
        input_data = pd.DataFrame(req_json)  # Esperamos una lista de diccionarios
        prediction = modelo_lr.predict(input_data)
        probability = modelo_lr.predict_proba(input_data)

        return jsonify({
            "prediccion": prediction.tolist(),
            "probabilidad": probability.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=8000)