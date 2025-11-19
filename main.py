from flask import Flask, jsonify, request
from flask_cors import CORS
from functools import wraps
import os

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, google_search
from google.adk.types import Content, Part

app = Flask(__name__)
CORS(app, origins=["https://ticker-ai-agent.vercel.app"])

API_KEY = os.environ.get("cloudrun_API_KEY")
GOOGLE_AI_API_KEY = os.environ.get("GOOGLE_AI_API_KEY")

# Configure Gemini API for ADK
if GOOGLE_AI_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_AI_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"


# ---------------------------------------------------------------------
# AUTH DECORATOR
# ---------------------------------------------------------------------
def require_api_key(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        client_key = request.headers.get("x-api-key")
        if not API_KEY:
            return jsonify({"error": "Server API key not configured"}), 500
        if client_key != API_KEY:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------
# DEFINE AGENTS (created only once at startup)
# ---------------------------------------------------------------------
research_agent = LlmAgent(
    name="ResearchAgent",
    model="gemini-2.5-flash-lite",
    instruction=(
        "Use google_search to find 2–3 key facts about the query. "
        "Return short factual items with citations."
    ),
    tools=[google_search],
    output_key="research_findings",
)

summarizer_agent = LlmAgent(
    name="SummarizerAgent",
    model="gemini-2.5-flash-lite",
    instruction=(
        "Summarize state['research_findings'] in 3–5 bullet points."
    ),
    output_key="final_summary",
)

root_agent = LlmAgent(
    name="ResearchCoordinator",
    model="gemini-2.5-flash-lite",
    instruction=(
        "1) Call ResearchAgent.\n"
        "2) Then call SummarizerAgent.\n"
        "3) Provide the final bullet-point summary."
    ),
    tools=[AgentTool(research_agent), AgentTool(summarizer_agent)],
)

runner = InMemoryRunner(agent=root_agent, app_name="TickerResearchApp")


# ---------------------------------------------------------------------
# SYNCHRONOUS RUN (no asyncio, no awaits)
# ---------------------------------------------------------------------
def run_agent_sync(question: str) -> str:

    events = runner.run(
        user_id="browser_user",
        session_id="browser_session",
        new_message=Content(
            role="user",
            parts=[Part(text=question)]
        )
    )

    collected = []

    for event in events:
        if getattr(event, "content", None):
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    collected.append(part.text)

    if not collected:
        return "(No text response from agent)"

    return "\n".join(collected)


# ---------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------
@app.post("/agent/query")
@require_api_key
def agent_query():
    body = request.get_json(silent=True) or {}
    question = body.get("question")

    if not question:
        return jsonify({"error": "Missing 'question'"}), 400

    try:
        answer = run_agent_sync(question)
    except Exception as e:
        print(f"❌ Agent execution failed: {e}", flush=True)
        return jsonify({"error": "Agent execution failed"}), 500

    return jsonify({"question": question, "answer": answer})


@app.get("/healthz")
def healthz():
    return "ok", 200


@app.get("/")
def root():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
