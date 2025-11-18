from flask import Flask, jsonify, request
from functools import wraps
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types
import os

app = Flask(__name__)

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


@app.get("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.get("/hello")
@require_api_key
def hello():
    return jsonify({"message": "Hello from Cloud Run (authenticated)!"})


@app.post("/agent/query")
@require_api_key
def agent_query():
    body = request.get_json(silent=True) or {}
    question = body.get("question")

    if not question:
        return jsonify({"error": "Missing 'question' in JSON body"}), 400

    

    try:
        GOOGLE_API_KEY = os.environ["GOOGLE_AI_API_KEY"]
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
        print("✅ Gemini API key setup complete.")
    except Exception as e:
        print(f"🔑 Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")

    answer = f"You asked: {question}. (This is a placeholder agent response.)"

    
    # Research Agent: Its job is to use the google_search tool and present findings.
    research_agent = Agent(
        name="ResearchAgent",
        model="gemini-2.5-flash-lite",
        instruction="""You are a specialized research agent. Your only job is to use the
        google_search tool to find 2-3 pieces of relevant information on the given topic and present the findings with citations.""",
        tools=[google_search],
        output_key="research_findings", # The result of this agent will be stored in the session state with this key.
    )

    print("✅ research_agent created.")
    # Summarizer Agent: Its job is to summarize the text it receives.
    summarizer_agent = Agent(
        name="SummarizerAgent",
        model="gemini-2.5-flash-lite",
        # The instruction is modified to request a bulleted list for a clear output format.
        instruction="""Read the provided research findings: {research_findings}
    Create a concise summary as a bulleted list with 3-5 key points.""",
        output_key="final_summary",
    )

    print("✅ summarizer_agent created.")
    # Root Coordinator: Orchestrates the workflow by calling the sub-agents as tools.
    root_agent = Agent(
        name="ResearchCoordinator",
        model="gemini-2.5-flash-lite",
        # This instruction tells the root agent HOW to use its tools (which are the other agents).
        instruction="""You are a research coordinator. Your goal is to answer the user's query by orchestrating a workflow.
    1. First, you MUST call the `ResearchAgent` tool to find relevant information on the topic provided by the user.
    2. Next, after receiving the research findings, you MUST call the `SummarizerAgent` tool to create a concise summary.
    3. Finally, present the final summary clearly to the user as your response.""",
        # We wrap the sub-agents in `AgentTool` to make them callable tools for the root agent.
        tools=[
            AgentTool(research_agent),
            AgentTool(summarizer_agent)
        ],
    )

    print("✅ root_agent created.")
    runner = InMemoryRunner(agent=root_agent)
    response = await runner.run_debug("""What are the latest news for the AAPL ticker and the company associated?""")
    return jsonify({
        "question": question,
        "answer": response
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
