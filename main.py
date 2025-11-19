from flask import Flask, jsonify, request
from flask_cors import CORS
from functools import wraps
import os

from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, google_search
from google.genai import types  # <-- important

app = Flask(__name__)
CORS(app, origins=["https://ticker-ai-agent.vercel.app"])

API_KEY = os.environ.get("cloudrun_API_KEY")


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

    runner = InMemoryRunner(agent=root_agent)

    # --- Construire le message utilisateur au bon format ---
    content = types.Content(
        role="user",
        parts=[types.Part.from_text(question)],
    )

    try:
        # run() est synchrone et renvoie un générateur d'événements
        events = runner.run(
            user_id="web-user",
            session_id="web-session",
            new_message=content,
        )

        answer_text = ""
        for event in events:
            if event.content and event.content.parts:
                for part in event.content.parts:
                    # certains events n’ont pas de texte (tool calls, etc.)
                    if getattr(part, "text", None):
                        answer_text += part.text

        if not answer_text:
            answer_text = "(No text response from agent)"

        return jsonify(
            {
                "question": question,
                "answer": answer_text,
            }
        )

    except Exception as e:
        print(f"❌ Agent execution failed: {e}")
        return jsonify({"error": "Agent execution failed"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
