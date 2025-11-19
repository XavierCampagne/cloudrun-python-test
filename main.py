from flask import Flask, jsonify, request
from flask_cors import CORS
from functools import wraps
import os

# --- ADK imports ---
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, google_search

# --------------------
# Flask setup
# --------------------
app = Flask(__name__)
CORS(app, origins=["https://ticker-ai-agent.vercel.app"])

# Your API KEY for clients
API_KEY = os.environ.get("cloudrun_API_KEY")


# --------------------
# Auth decorator
# --------------------
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


# --------------------
# Agent Query Endpoint
# --------------------
@app.post("/agent/query")
@require_api_key
def agent_query():
    body = request.get_json(silent=True) or {}
    question = body.get("question")

    if not question:
        return jsonify({"error": "Missing 'question' in JSON body"}), 400

    # --- Setup Gemini API key ---
    try:
        google_api_key = os.environ["GOOGLE_AI_API_KEY"]
        os.environ["GOOGLE_API_KEY"] = google_api_key
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
        print("✅ Gemini API key loaded.")
    except Exception as e:
        print(f"❌ Missing GOOGLE_AI_API_KEY env var: {e}")
        return jsonify({"error": "Gemini API key not configured"}), 500

    # --- Agent Definitions ---
    research_agent = Agent(
        name="ResearchAgent",
        model="gemini-2.5-flash-lite",
        instruction=(
            "You are a specialized research agent. "
            "Use the google_search tool to find 2–3 relevant pieces of information "
            "and present the findings with brief citations."
        ),
        tools=[google_search],
        output_key="research_findings",
    )

    summarizer_agent = Agent(
        name="SummarizerAgent",
        model="gemini-2.5-flash-lite",
        instruction=(
            "Here are the research findings: {research_findings}\n"
            "Create a concise summary as a bulleted list with 3–5 key points."
        ),
        output_key="final_summary",
    )

    root_agent = Agent(
        name="ResearchCoordinator",
        model="gemini-2.5-flash-lite",
        instruction=(
            "You orchestrate the workflow:\n"
            "1. Call ResearchAgent.\n"
            "2. Call SummarizerAgent.\n"
            "3. Return the final summary."
        ),
        tools=[AgentTool(research_agent), AgentTool(summarizer_agent)],
    )

    # --- RUN AGENT SYNCHRONOUSLY ---
    runner = InMemoryRunner(agent=root_agent)

    try:
        response = runner.run(question)   # <-- synchronous, Cloud Run friendly
    except Exception as e:
        print(f"❌ Agent execution failed: {e}")
        return jsonify({"error": "Agent execution failed"}), 500

    return jsonify({
        "question": question,
        "answer": str(response)
    })


# --------------------
# Cloud Run Startup
# --------------------
@app.get("/healthz")
def healthz():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"🚀 Starting Flask on port {port}")
    app.run(host="0.0.0.0", port=port)
