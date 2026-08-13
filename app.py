"""
Aplicación Flask - Predicción de la edad de abulones (Abalone)
Proyecto de regresión CRISP-DM - Módulo 2
Carga el pipeline entrenado (imputación + escalado + red neuronal MLP)
y expone un formulario web para predecir la edad (en años) a partir de
medidas físicas originales del abulón.
"""
import os
import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "modelo_final_abalone.joblib")
pipeline = joblib.load(MODEL_PATH)

# Variables originales que solicita el formulario, en el mismo orden y
# unidades que el dataset original (todas en pulgadas/gramos según UCI,
# se muestran al usuario con su significado real).
FIELDS = [
    {"name": "Length", "label": "Longitud de la concha", "unit": "mm (medida más larga)", "min": 0.0, "max": 1.0, "step": "0.001"},
    {"name": "Height", "label": "Altura de la concha", "unit": "mm (con la carne dentro)", "min": 0.0, "max": 0.5, "step": "0.001"},
    {"name": "Whole_weight", "label": "Peso total del abulón", "unit": "gramos", "min": 0.0, "max": 3.0, "step": "0.001"},
    {"name": "Shucked_weight", "label": "Peso de la carne (sin concha)", "unit": "gramos", "min": 0.0, "max": 1.5, "step": "0.001"},
    {"name": "Shell_weight", "label": "Peso de la concha seca", "unit": "gramos", "min": 0.0, "max": 1.5, "step": "0.001"},
]


def validate_inputs(form):
    values = {}
    errors = []
    for f in FIELDS:
        raw = form.get(f["name"], "").strip()
        if raw == "":
            errors.append(f"El campo '{f['label']}' es obligatorio.")
            continue
        try:
            val = float(raw)
        except ValueError:
            errors.append(f"El campo '{f['label']}' debe ser un número válido.")
            continue
        if val < 0:
            errors.append(f"El campo '{f['label']}' no puede ser negativo.")
            continue
        if val > f["max"] * 3:
            errors.append(f"El valor de '{f['label']}' parece fuera de un rango físico razonable.")
            continue
        values[f["name"]] = val
    return values, errors


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    errors = []
    submitted_values = {}

    if request.method == "POST":
        submitted_values, errors = validate_inputs(request.form)
        if not errors:
            import pandas as pd
            X_new = pd.DataFrame([submitted_values])[
                ["Shell_weight", "Height", "Shucked_weight", "Length", "Whole_weight"]
          ]
            try:
                pred_age = float(pipeline.predict(X_new)[0])
                prediction = round(pred_age, 2)
            except Exception as e:
                errors.append(f"No fue posible generar la predicción: {e}")

    return render_template("index.html", fields=FIELDS, prediction=prediction,
                            errors=errors, values=submitted_values)


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
  
