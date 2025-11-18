from flask import Flask, jsonify
import os

app = Flask(__name__)

@app.route("/hello", methods=["GET"])
def hello():
    return jsonify({"message": "Hello from Cloud Run!"})

@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Root OK from Cloud Run!"})

if __name__ == "__main__":
    # Cloud Run fournit PORT dans les variables d'environnement
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)