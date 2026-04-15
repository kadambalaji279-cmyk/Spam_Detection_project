from flask import Flask, render_template, request
from spam_detector import predict_message

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]
    result = predict_message(message)
    return render_template("index.html", prediction=result, msg=message)

if __name__ == "__main__":
    app.run(debug=True)