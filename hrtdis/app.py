from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = [
        int(request.form['age']),
        int(request.form['sex']),
        int(request.form['cp']),
        int(request.form['trestbps']),
        int(request.form['chol']),
        int(request.form['fbs']),
        int(request.form['restecg']),
        int(request.form['thalach']),
        int(request.form['exang']),
        float(request.form['oldpeak']),
        int(request.form['slope'])
    ]
    input_data = np.array([data])
    prediction = model.predict(input_data)[0]

    result = "You have Heart Disease" if prediction == 1 else "You do not have Heart Disease"

    return render_template("result.html", form_data=request.form, prediction=result)

if __name__ == '__main__':
    app.run(debug=True)


