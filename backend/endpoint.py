from flask import Flask, jsonify, Response
from load_and_predict import predict
import json

app = Flask(__name__)

@app.route("/")
def predict_axis(text, axis):
    pred = predict(text, axis)
    response = resp = Response(response=json.dumps(pred), status=200,  mimetype="text/plain")
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

@app.route("/concreteness/<text>")
def predict_concreteness(text=""):
    return predict_axis(text, "concreteness")

@app.route("/positivity/<text>")
def predict_positivity(text=""):
    return predict_axis(text, "positivity")

@app.route("/sincerity/<text>")
def predict_sincerity(text=""):
    return predict_axis(text, "sincerity")

@app.route("/intensity/<text>")
def predict_intensity(text=""):
    return predict_axis(text, "intensity")
