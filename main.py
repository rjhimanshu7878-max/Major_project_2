from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
model = pickle.load(open("car_price_model.pkl", "rb"))


# -------------------------------
# Helper function to extract numbers
# -------------------------------
def extract_number(value):
    try:
        return float(''.join([ch for ch in value if ch.isdigit() or ch == '.']))
    except:
        return 0

# -------------------------------
# Routes
# -------------------------------

@app.route('/')
def home():
    return render_template("index.html")
    # Contact Page
@app.route('/contact', methods=["GET"])
def contact_page():
    return render_template("contact.html")

@app.route('/history', methods=["GET"])
def history_page():
    return render_template("history.html")

# Predict Page (GET → open form)

@app.route('/predict', methods=["GET"])
def predict_page():
    return render_template("predict.html")

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Get input values from form
        year = int(request.form.get("year", 0))
        km_driven = int(request.form.get("km_driven", 0))

        fuel = int(request.form.get("fuel", 0))
        seller_type = int(request.form.get("seller_type", 0))
        transmission = int(request.form.get("transmission", 0))
        owner = int(request.form.get("owner", 0))

        mileage = extract_number(request.form.get("mileage", "0"))
        engine = extract_number(request.form.get("engine", "0"))
        max_power = extract_number(request.form.get("max_power", "0"))
        torque = extract_number(request.form.get("torque", "0"))

        seats = float(request.form.get("seats", 0))

        # Feature order MUST match training
        features = [
            year,
            km_driven,
            fuel,
            seller_type,
            transmission,
            owner,
            mileage,
            engine,
            max_power,
            torque,
            seats
        ]

        # Convert to numpy array
        data = np.array([features])

        # Scale data
        data = scaler.transform(data)

        # Predict price
        prediction = model.predict(data)[0]

        # Format result
        result = f"Estimated Selling Price: ₹ {round(prediction, 2)}"

        return render_template("predict.html", prediction_text=result)

    except Exception as e:
        return render_template("predict.html", prediction_text=f"Error: {str(e)}")

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)