from flask import Flask, render_template, request, jsonify
import joblib
from datetime import datetime
from serial_reader import SerialReader

app = Flask(__name__)

# Load trained model
model = joblib.load("allergy_model.pkl")

# Serial connection
reader = SerialReader(port="COM4", baudrate=115200)
reader.connect()

# Store recent prediction history
history = []

# Static project metrics for dashboard cards
metrics = {
    "accuracy": 0.92,
    "precision": [0.91, 0.90, 0.94, 0.92],
    "recall": [0.90, 0.89, 0.95, 0.91],
    "f1": [0.90, 0.89, 0.94, 0.91],
    "labels": ["Egg", "Milk", "Peanut", "Safe"]
}


def get_risk(prediction):
    prediction = prediction.lower()
    if prediction == "peanut":
        return "High"
    elif prediction in ["egg", "milk"]:
        return "Medium"
    return "Low"


def get_confidence(value):
    # Placeholder confidence logic for demo/project use
    # Replace later with predict_proba if needed
    return 90


def read_live_sensor():
    value = reader.read_value()
    return value


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detect", methods=["GET", "POST"])
def detect():
    result = None
    error = None

    if request.method == "POST":
        try:
            sensor_value = float(request.form["sensor_value"])
            prediction = model.predict([[sensor_value]])[0]
            risk = get_risk(prediction)
            confidence = get_confidence(sensor_value)

            result = {
                "prediction": prediction,
                "risk": risk,
                "confidence": confidence,
                "sensor_value": sensor_value
            }

            history.insert(0, {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "sensor_value": sensor_value,
                "prediction": prediction,
                "confidence": confidence,
                "risk": risk
            })

            if len(history) > 10:
                history.pop()

        except Exception as e:
            error = f"Prediction failed: {e}"

    return render_template("detection.html", result=result, error=error)


@app.route("/live_detect")
def live_detect():
    value = read_live_sensor()

    if value is None:
        return jsonify({
            "success": False,
            "message": "No sensor data available"
        })

    prediction = model.predict([[value]])[0]
    risk = get_risk(prediction)
    confidence = get_confidence(value)

    item = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sensor_value": value,
        "prediction": prediction,
        "confidence": confidence,
        "risk": risk
    }

    history.insert(0, item)
    if len(history) > 10:
        history.pop()

    return jsonify({
        "success": True,
        "sensor_value": value,
        "prediction": prediction,
        "risk": risk,
        "confidence": confidence
    })


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", metrics=metrics, history=history)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)