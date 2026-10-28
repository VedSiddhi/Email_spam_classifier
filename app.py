from flask import Flask, render_template, request
import joblib


# -----------------------------------
# Create Flask Application
# -----------------------------------

app = Flask(__name__)


# -----------------------------------
# Load Trained Model
# -----------------------------------

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


# -----------------------------------
# Home Page
# -----------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------------
# Prediction
# -----------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    # Get message from HTML form
    message = request.form.get("message", "").strip()

    # Check if message is empty
    if not message:
        return render_template(
            "index.html",
            prediction=None,
            message=""
        )

    # Convert message into numerical features
    message_vector = vectorizer.transform([message])

    # Make prediction
    prediction = model.predict(message_vector)[0]

    # Convert numerical prediction into text
    if prediction == 1:
        result = "Spam"
    else:
        result = "Not Spam"

    # Send result back to HTML
    return render_template(
        "index.html",
        prediction=result,
        message=message
    )


# -----------------------------------
# Run Application
# -----------------------------------

if __name__ == "__main__":
    app.run(debug=True)
