from flask import Flask, jsonify

# Cloud Run (build from source) s'attend à trouver un objet WSGI nommé "app"
app = Flask(__name__)

@app.route("/hello", methods=["GET"])
def hello():
    return jsonify({"message": "Hello from Cloud Run!"})

@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Root OK from Cloud Run!"})