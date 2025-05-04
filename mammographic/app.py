#importación de librerias necesarias
from flask import Flask, request, render_template
import pandas as pd
import pickle
import os
# función para cargar modelo
def load_object(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

app = Flask(__name__)

# Carga el modelo de regresion logística entrenado (ya guardado)
modelo_lr = load_object('mammographic/modelo_regresion_logistica.pkl')

@app.route('/predecir', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            birads = int(request.form['birads'])
            age = int(request.form['age'])
            shape = int(request.form['shape'])
            margin = int(request.form['margin'])
            density = int(request.form['density'])

            input_data = pd.DataFrame([{
                "BIRADS": birads,
                "Age": age,
                "Shape": shape,
                "Margin": margin,
                "Density": density
            }])

            # Realizar predicción
            prediction = modelo_lr.predict(input_data)[0]
            resultado = 'Maligno' if prediction == 1 else 'Benigno'
            probalidad = round(modelo_lr.predict_proba(input_data).max()*100,2) 
            return render_template('formulario.html', resultado=resultado,probalidad=probalidad)

        except Exception as e:
            return render_template('formulario.html', resultado=f"Error: {e}")

    return render_template('formulario.html', resultado=None)

if __name__ == '__main__':
    app.run(debug=True, port=8000)