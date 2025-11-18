from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/hello", methods=["GET"])
def hello():
    return jsonify({"message": "Hello from Cloud Run!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)