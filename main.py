from flask import Flask, jsonify, request
from functools import wraps
import os
import asyncio

from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, google_search

app = Flask(__name__)

API_KEY = os.environ.get("API_KEY")  # <-- garde ce nom aligné avec Cloud Run


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
        os.environ["GOOGLE_API_KEY"] = google_api_key
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
        print("✅ Gemini API key setup complete.")
    except Exception as e:
        print(f"🔑 Authentication Error: missing GOOGLE_AI_API_KEY env var. Details: {e}")
        return jsonify({"error": "Gemini API key not configured on server"}), 500

    # --- Define agents ---

    research_agent = Agent(
        name="ResearchAgent",
        model="gemini-2.5-flash-lite",
        instruction=(
            "You are a specialized research agent. "
            "Use the google_search tool to find 2–3 relevant pieces of information on the topic, "
            "and present the findings with brief citations."
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

    async def run_agent():
        return await runner.run_debug(question)

    try:
        adk_response = asyncio.run(run_agent())
    except Exception as e:
        # Log and return a 500 with a simple message
        print(f"❌ Error while running ADK agent: {e}")
        return jsonify({"error": "Agent execution failed"}), 500

    return jsonify({
        "question": question,
        "answer": str(adk_response),
    })
