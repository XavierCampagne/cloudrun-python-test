from flask import Flask, jsonify, request
from flask_cors import CORS
from functools import wraps
import os
import asyncio

from google.adk.agents import Agent, LlmAgent
from google.adk.tools import AgentTool, google_search
from google.adk.runners import InMemoryRunner

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
app = Flask(__name__)
CORS(app, origins=["https://ticker-ai-agent.vercel.app"])

API_KEY = os.environ.get("cloudrun_API_KEY")
GOOGLE_AI_API_KEY = os.environ.get("GOOGLE_AI_API_KEY")

if GOOGLE_AI_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_AI_API_KEY
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")


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


# -----------------------------------------------------------------------------
# Agents + Runner
# -----------------------------------------------------------------------------
research_agent = LlmAgent(
    name="ResearchAgent",
    model="gemini-2.5-flash-lite",
    instruction=(
        "You are a research agent. Use the google_search tool to gather "
        "2–3 relevant facts with citations."
    ),
    tools=[google_search],
    output_key="research_findings",
)

summarizer_agent = LlmAgent(
    name="SummarizerAgent",
    model="gemini-2.5-flash-lite",
    instruction=(
        "Summarize the content in state['research_findings'] into 3–5 bullet points."
    ),
    output_key="final_summary",
)

root_agent = LlmAgent(
    name="Coordinator",
    model="gemini-2.5-flash-lite",
    instruction=(
        "1. Call ResearchAgent.\n"
        "2. Then call SummarizerAgent.\n"
        "3. Return the final summary."
    ),
    tools=[AgentTool(research_agent), AgentTool(summarizer_agent)],
)

runner = InMemoryRunner(agent=root_agent, app_name="TickerResearchApp")


async def run_agent_once(question: str) -> str:
    """Runs the agent end-to-end and returns concatenated text output."""

    session = await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id="web_user",
    )

    text_output = []

    async for event in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=question,
    ):
        # The Cloud Run ADK runner emits events with optional .text property
        txt = getattr(event, "text", None)
        if txt:
            text_output.append(txt)

    if not text_output:
        return "(No text response from agent)"

    return "\n".join(text_output)


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.post("/agent/query")
@require_api_key
def agent_query():
    body = request.get_json(silent=True) or {}
    question = body.get("question")

    if not question:
        return jsonify({"error": "Missing 'question'"}), 400

    if not GOOGLE_AI_API_KEY:
        return jsonify({"error": "Google AI API key missing"}), 500

    try:
        answer = asyncio.run(run_agent_once(question))
    except Exception as e:
        print(f"❌ Agent execution failed: {e}", flush=True)
        return jsonify({"error": "Agent execution failed"}), 500

    return jsonify({"question": question, "answer": answer})


@app.get("/")
def root():
    return jsonify({"status": "ok"})


@app.get("/healthz")
def health():
    return "ok", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
