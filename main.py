from flask import Flask, jsonify, request
from functools import wraps
import os

app = Flask(__name__)

API_KEY = os.environ.get("API_KEY")


def require_api_key(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        client_key = request.headers.get("x-api-key")
        if not API_KEY:
            return jsonify({"error": "Server API key not configured"}), 500
        if not client_key or client_key != API_KEY:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return wrapper


@app.get("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.get("/hello")
@require_api_key
def hello():
    return jsonify({"message": "Hello from Cloud Run (authenticated)!"})


@app.post("/agent/query")
@require_api_key
def agent_query():
    body = request.get_json(silent=True) or {}
    question = body.get("question")

    if not question:
        return jsonify({"error": "Missing 'question' in JSON body"}), 400

    # 🔁 Placeholder agent logic (to replace later)
    answer = f"You asked: {question}. (This is a placeholder agent response.)"

    return jsonify({
        "question": question,
        "answer": answer
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
