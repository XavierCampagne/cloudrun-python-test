from flask import Flask, jsonify, request
from flask_cors import CORS
from functools import wraps
import os

from google.genai.types import Part, Content
from google.adk.agents import Agent
from google.adk.tools import AgentTool, google_search
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.runners import Runner
from google.adk.agents.run_config import RunConfig

# -------------------------------------------------------------------
# Flask + CORS
# -------------------------------------------------------------------
app = Flask(__name__)
CORS(app, origins=["https://ticker-ai-agent.vercel.app"])

API_KEY = os.environ.get("cloudrun_API_KEY")

# -------------------------------------------------------------------
# Gemini API key (from Cloud Run secret)
# -------------------------------------------------------------------
GOOGLE_AI_KEY = os.environ.get("GOOGLE_AI_API_KEY")
if GOOGLE_AI_KEY:
    # env var name expected by google-genai / ADK
    os.environ["GOOGLE_API_KEY"] = GOOGLE_AI_KEY
    # stay on API-key mode, not Vertex
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "false"
    print("✅ GOOGLE_API_KEY configured from secret.")
else:
    print("⚠️ GOOGLE_AI_API_KEY not set – Gemini calls will fail.")

# -------------------------------------------------------------------
# ADK setup: session service + agents + runner factory
# -------------------------------------------------------------------
APP_NAME = "Ticker-AI-Agent"
session_service = InMemorySessionService()

# ---- Sub-agents ----
research_agent = Agent(
    name="ResearchAgent",
    model="gemini-2.5-flash-lite",
    instruction=(
        "You are a specialized research agent. "
        "Use the google_search tool to find 2–3 relevant pieces of information "
        "on the given topic and present the findings with brief citations."
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

# ---- Root agent ----
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

def create_runner(session_id: str):
    """
    Create a session + Runner for this session_id.
    (Here we keep it simple and reuse the same id per client.)
    """
    session = session_service.create_session(
        app_name=APP_NAME,
        user_id=session_id,
        session_id=session_id,
    )
    runner = Runner(
        app_name=APP_NAME,
        agent=root_agent,
        session_service=session_service,
    )
    return session, runner

# -------------------------------------------------------------------
# Auth decorator
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# Healthcheck
# -------------------------------------------------------------------
@app.get("/healthz")
def healthz():
    return jsonify({"status": "ok"})


# -------------------------------------------------------------------
# Main endpoint
# -------------------------------------------------------------------
@app.post("/agent/query")
@require_api_key
def agent_query():
    body = request.get_json(silent=True) or {}
    question = body.get("question")

    if not question:
        return jsonify({"error": "Missing 'question' in JSON body"}), 400

    if not GOOGLE_AI_KEY:
        return jsonify({"error": "Gemini API key not configured on server"}), 500

    # Create session + runner
    session_id = "web-session"  # you can later derive it from user info/JWT
    session, runner = create_runner(session_id)

    # Build user message
    user_message = Content(
        role="user",
        parts=[Part.from_text(text=question)],  # <-- keyword arg is required
    )

    # Ask explicitly for TEXT output
    run_config = RunConfig(response_modalities=["TEXT"])

    try:
        events = runner.run(
            user_id=session.user_id,
            session_id=session.id,
            new_message=user_message,
            run_config=run_config,
        )

        full_response_parts = []
        for event in events:
            # Debug so you can see in Cloud Run logs what is happening
            print("🔎 ADK event:", event)

            if event.content and event.content.parts:
                for part in event.content.parts:
                    if getattr(part, "text", None):
                        full_response_parts.append(part.text)
                    elif getattr(part, "function_call", None):
                        print(f"🛠️ Function call: {part.function_call.name}")

        answer = "".join(full_response_parts).strip()
        if not answer:
            answer = "Agent did not return any text response."

        return jsonify({"question": question, "answer": answer})

    except Exception as e:
        print(f"❌ Agent execution failed: {e}")
        return jsonify({"error": "Agent execution failed"}), 500


# -------------------------------------------------------------------
# Local dev
# -------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
