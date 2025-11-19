from flask import Flask, jsonify, request
from flask_cors import CORS
from functools import wraps
import os
import asyncio

from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, google_search
from google.genai.types import Part, UserContent

# ---------------------------------------------------------------------------
# Flask + CORS
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app, origins=["https://ticker-ai-agent.vercel.app"])

API_KEY = os.environ.get("cloudrun_API_KEY")

GOOGLE_AI_API_KEY = os.environ.get("GOOGLE_AI_API_KEY")
if GOOGLE_AI_API_KEY:
    # ADK uses google-genai under the hood and reads this var
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


# ---------------------------------------------------------------------------
# ADK agents & runner (created once at import time)
# ---------------------------------------------------------------------------
research_agent = Agent(
    name="ResearchAgent",
    model="gemini-2.5-flash-lite",
    instruction=(
        "You are a specialized research agent. "
        "Use the google_search tool to find 2–3 relevant pieces of information "
        "on the topic, and present the findings with brief citations."
    ),
    tools=[google_search],
    output_key="research_findings",
)

summarizer_agent = Agent(
    name="SummarizerAgent",
    model="gemini-2.5-flash-lite",
    instruction=(
        "You receive prior research in state['research_findings'].\n"
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

runner = InMemoryRunner(agent=root_agent, app_name="TickerResearchApp")


async def run_agent_once(question: str) -> str:
    """
    Creates a new session, runs the root agent once with the user's question,
    and returns the concatenated text response.
    """
    # create_session is async in your installed ADK -> we await it
    session = await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id="web_user",
    )

    # Wrap the user question into a Content object
    user_content = UserContent(parts=[Part(text=question)])

    text_chunks: list[str] = []

    # ADK async run loop
    async for event in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=user_content,
    ):
        content = getattr(event, "content", None)
        if not content or not getattr(content, "parts", None):
            continue

        for part in content.parts:
            txt = getattr(part, "text", None)
            if txt:
                text_chunks.append(txt)

    if not text_chunks:
        return "(No text response from agent)"

    return "\n".join(text_chunks)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post("/agent/query")
@require_api_key
def agent_query():
    body = request.get_json(silent=True) or {}
    question = body.get("question")

    if not question:
        return jsonify({"error": "Missing 'question' in JSON body"}), 400

    if not GOOGLE_AI_API_KEY:
        return jsonify({"error": 'Env var "GOOGLE_AI_API_KEY" is not set'}), 500

    try:
        # Run the async agent pipeline in a fresh event loop
        answer = asyncio.run(run_agent_once(question))
    except Exception as e:
        print(f"❌ Agent execution failed: {e}", flush=True)
        return jsonify({"error": "Agent execution failed"}), 500

    return jsonify({"question": question, "answer": answer})


@app.get("/")
def root():
    return jsonify({"status": "ok"})


@app.get("/healthz")
def healthz():
    # For Cloud Run / external health checks
    return "ok", 200


if __name__ == "__main__":
    # Local testing
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
