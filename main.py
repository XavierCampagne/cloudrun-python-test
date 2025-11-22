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
news_collector_agent = Agent(
    name="NewsCollectorAgent",
    model="gemini-2.5-flash-lite",
    instruction=(
        "You are a news collector for a financial ticker.\n"
        "Input: a ticker symbol (e.g. 'TSLA') and an optional time window.\n"
        "Use the google_search tool to find the 5–10 most relevant, recent news "
        "about this ticker.\n"
        "For each news item, return a JSON list under key 'news_items' with objects:\n"
        "{'headline': ..., 'source': ..., 'date': ..., 'url': ..., 'short_summary': ...}.\n"
        "short_summary should be 2–3 lines max, in your own words."
    ),
    tools=[google_search],
    output_key="news_items",
)

classifier_agent = Agent(
    name="ClassifierAgent",
    model="gemini-2.5-flash-lite",
    instruction=(
        "You are a financial news classifier.\n"
        "You receive state['news_items'], which is a list of news objects about a ticker.\n"
        "For each news item, add:\n"
        "- 'category' in "
        "['earnings','guidance','macro','rating','product','legal','other']\n"
        "- 'impact_short_term' in ['low','medium','high']\n"
        "- 'impact_long_term' in ['low','medium','high']\n"
        "Return the updated list as JSON under key 'classified_news'."
    ),
    output_key="classified_news",
)

briefing_agent = Agent(
    name="BriefingAgent",
    model="gemini-2.5-flash-lite",
    instruction=(
        "You are generating a decision-oriented briefing for someone who follows a ticker.\n"
        "You receive state['classified_news'].\n"
        "Produce a structured markdown report with these sections:\n"
        "1. Quick Snapshot (3 bullets max)\n"
        "2. Material Events (High Impact)\n"
        "   - List only items with 'high' in impact_short_term or impact_long_term.\n"
        "3. Other Notable Items\n"
        "   - Group medium-impact items by category.\n"
        "4. Noise Filtered Out\n"
        "   - Briefly mention the type of news you ignored (duplicates, generic blogs, etc.).\n"
        "5. Final advice\n"
        "   - Make a conclusion on how the ticker can be impacted short-term, mid-term and long-term.\n"
    ),
    output_key="final_briefing",
)

root_agent = Agent(
    name="TickerBriefingCoordinator",
    model="gemini-2.5-flash-lite",
    instruction=(
        "You create a decision-oriented news briefing for a single stock ticker.\n"
        "Process:\n"
        "1. Call NewsCollectorAgent to fetch and summarize recent news about the ticker provided by the user.\n"
        "2. Call ClassifierAgent to tag and score impact of each news item.\n"
        "3. Call BriefingAgent to generate the final structured briefing.\n"
        "4. Return ONLY the 'final_briefing' to the user."
    ),
    tools=[
        AgentTool(news_collector_agent),
        AgentTool(classifier_agent),
        AgentTool(briefing_agent),
    ],
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
