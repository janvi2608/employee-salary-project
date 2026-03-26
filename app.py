from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("salary_prediction_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict_page")
def predict_page():
    return render_template("predict.html")


@app.route("/predict", methods=["POST"])
def predict():

    age = float(request.form["age"])
    gender = int(request.form["gender"])
    jobrole = int(request.form["jobrole"])
    experience = float(request.form["experience"])
    education = int(request.form["education"])

    department = request.form["department"]
    location = request.form["location"]

    # Simple Encoding (same mapping used during training)
    dept_dict = {
        "Engineering":0,
        "Sales":1,
        "Finance":2,
        "HR":3,
        "Marketing":4,
        "Product":5
    }

    loc_dict = {
        "Austin":0,
        "Seattle":1,
        "New York":2,
        "San Francisco":3,
        "Chicago":4
    }

    dept_encoded = dept_dict.get(department, 0)
    loc_encoded = loc_dict.get(location, 0)

    features = np.array([[age, gender, jobrole, dept_encoded, experience, education, loc_encoded]])

    prediction = model.predict(features)

    return render_template(
        "predict.html",
        prediction_text=f"Predicted Salary: ₹ {round(prediction[0],2)}"
    )


if __name__ == "__main__":
    app.run(debug=True)