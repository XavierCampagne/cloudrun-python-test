from flask import Flask, jsonify, request
from flask_cors import CORS
from functools import wraps
import os

from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, google_search
from google.genai.types import UserContent, Part

app = Flask(__name__)
CORS(app, origins=["https://ticker-ai-agent.vercel.app"])

# -------- Auth with x-api-key --------
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


# -------- Simple healthcheck (no auth) --------
@app.get("/healthz")
def healthz():
    return jsonify({"status": "ok"})


# -------- Lazy ADK init (global runner) --------
ADK_RUNNER: InMemoryRunner | None = None


def get_runner() -> InMemoryRunner:
    """Create the ADK runner once and reuse it."""
    global ADK_RUNNER
    if ADK_RUNNER is not None:
        return ADK_RUNNER

    # Gemini API key from env (you mapped your Secret Manager secret to this)
    google_api_key = os.environ.get("GOOGLE_AI_API_KEY")
    if not google_api_key:
        raise RuntimeError("GOOGLE_AI_API_KEY not set")
    os.environ["GOOGLE_API_KEY"] = google_api_key
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"

    # --- Define agents ---

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

    # InMemoryRunner gives us a session_service + sync run() helper
    ADK_RUNNER = InMemoryRunner(
        app_name="TickerResearchAgent",
        agent=root_agent,
    )
    print("✅ ADK InMemoryRunner initialized")
    return ADK_RUNNER


# -------- Main endpoint --------
@app.post("/agent/query")
@require_api_key
def agent_query():
    body = request.get_json(silent=True) or {}
    question = body.get("question")

    if not question:
        return jsonify({"error": "Missing 'question' in JSON body"}), 400

    try:
        runner = get_runner()
    except Exception as e:
        print(f"🔑 ADK init error: {e}")
        return jsonify({"error": "Server AI setup failure"}), 500

    try:
        # Create a session for this user/request
        session = runner.session_service.create_session(
            app_name=runner.app_name,
            user_id="web-user",  # you can plug a real user id later
        )

        # Build the user message as UserContent
        content = UserContent(
            parts=[Part(text=question)]
        )

        # Run the agent (sync wrapper around run_async)
        final_text = ""
        for event in runner.run(
            user_id=session.user_id,
            session_id=session.id,
            new_message=content,
        ):
            # We only care about final responses that have text parts
            if not getattr(event, "content", None):
                continue
            for part in event.content.parts:
                if getattr(part, "text", None):
                    final_text += part.text

        if not final_text:
            final_text = "(No text response from agent)"

        return jsonify(
            {
                "question": question,
                "answer": final_text,
            }
        )

    except Exception as e:
        print(f"❌ Agent execution failed: {e}")
        return jsonify({"error": "Agent execution failed"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    # debug=False in Cloud Run, but fine locally
    app.run(host="0.0.0.0", port=port)
