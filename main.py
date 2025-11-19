from flask import Flask, jsonify, request
from flask_cors import CORS
from functools import wraps
import os

from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, google_search

app = Flask(__name__)
CORS(app, origins=["https://ticker-ai-agent.vercel.app"])

API_KEY = os.environ.get("cloudrun_API_KEY")  # must match env var in Cloud Run


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


@app.post("/agent/query")
@require_api_key
def agent_query():
    body = request.get_json(silent=True) or {}
    question = body.get("question")

    if not question:
        return jsonify({"error": "Missing 'question' in JSON body"}), 400

    # --- Setup Gemini API key from secret ---
    try:
        google_api_key = os.environ["GOOGLE_AI_API_KEY"]
    except KeyError as e:
        print(f"🔑 Authentication Error: missing GOOGLE_AI_API_KEY env var. Details: {e}")
        return jsonify({"error": "Gemini API key not configured on server"}), 500

    os.environ["GOOGLE_API_KEY"] = google_api_key
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("✅ Gemini API key setup complete.")

    # --- Define agents ---

    research_agent = Agent(
        name="ResearchAgent",
        model="gemini-2.5-flash-lite",
        instruction=(
            "You are a specialized research agent. "
            "Use the google_search tool to find 2–3 relevant pieces of information on "
            "the topic, and present the findings with brief citations."
        ),
        tools=[google_search],
        output_key="research_findings",
    )

    summarizer_agent = Agent(
        name="SummarizerAgent",
        model="gemini-2.5-flash-lite",
        instruction=(
            "Read the provided research findings: {research_findings}\n"
            "Create a concise summary as a bulleted list with 3–5 key points."
        ),
        output_key="final_summary",
    )

    root_agent = Agent(
        name="ResearchCoordinator",
        model="gemini-2.5-flash-lite",
        instruction=(
            "You are a research coordinator. Your goal is to answer the user's query.\n"
            "1. Call `ResearchAgent` to gather information.\n"
            "2. Call `SummarizerAgent` to summarize the findings.\n"
            "3. Return the final summary clearly to the user."
        ),
        tools=[AgentTool(research_agent), AgentTool(summarizer_agent)],
    )

    runner = InMemoryRunner(agent=root_agent)

    # ✅ Use keyword argument user_messages=question (no asyncio.run here)
    try:
        adk_response = runner.run(user_messages=question, quiet=False, verbose=False)
    except Exception as e:
        print(f"❌ Agent execution failed: {e}")
        return jsonify({"error": "Agent execution failed"}), 500

    return jsonify(
        {
            "question": question,
            "answer": str(adk_response),
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
