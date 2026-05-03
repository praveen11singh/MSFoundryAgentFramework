import os
import json
import time
import logging
from pathlib import Path

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import HttpResponseError, ServiceRequestError

from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    Agent,
    AgentThread,
    ThreadMessage,
    ThreadRun,
    ThreadMessageOptions,
    MessageRole,
    MessageAttachment,
    CodeInterpreterToolDefinition,
    FileSearchToolDefinition,
    FunctionTool,
    CodeInterpreterTool,
    FileSearchTool,
    ToolSet,
    ConnectedAgentTool,
    ConnectedAgentDetails,
    AgentEventHandler,
    AgentStreamEvent,
    MessageDeltaChunk,
    MessageDeltaTextContent,
    RunStepDeltaChunk,
    MessageTextContent,
    MessageImageFileContent,
    RunStatus,
    FilePurpose,
    AgentsResponseFormat,
)

load_dotenv()
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
# )
# logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CLIENT INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def get_client() -> AgentsClient:
    """
    Create an AgentsClient from FOUNDRY_PROJECT_ENDPOINT environment variable.

    Endpoint format:
        https://<aiservices-id>.services.ai.azure.com/api/projects/<project-name>
    """
    endpoint = os.environ.get("FOUNDRY_PROJECT_ENDPOINT")
    if not endpoint:
        raise EnvironmentError(
            "FOUNDRY_PROJECT_ENDPOINT is not set.\n"
            "Format: https://<aiservices-id>.services.ai.azure.com/api/projects/<project-name>"
        )
    client = AgentsClient(endpoint=endpoint, credential=DefaultAzureCredential())
   
    return client


# ─────────────────────────────────────────────────────────────────────────────
# 2. BASIC AGENT
# ─────────────────────────────────────────────────────────────────────────────

def pk_basic_agent(client: AgentsClient) -> None:
    """Create agent → thread → message → run → read response → cleanup."""
    model = os.environ["MODEL_DEPLOYMENT_NAME"]

    agent: Agent = client.create_agent(
        model=model,
        name="basic-assistant",
        instructions="You are a helpful assistant. Answer concisely and accurately.",
    )
    # logger.info("Agent created: %s", agent.id)

    thread: AgentThread = client.threads.create()

    client.messages.create(
        thread_id=thread.id,
        role=MessageRole.USER,
        content="What is UNESCO in two sentences?",
    )

    run: ThreadRun = client.runs.create_and_process(
        thread_id=thread.id,
        agent_id=agent.id,
    )

    if run.status == RunStatus.COMPLETED:
        last = client.messages.get_last_message_text_by_role(
            thread_id=thread.id,
            role=MessageRole.AGENT,
        )
        print("\n[Basic Agent]\n", last.text.value if last else "(no response)")
    else:
        # logger.error("Run failed: %s | %s", run.status, run.last_error)
        pass            

    client.threads.delete(thread.id)
    client.delete_agent(agent.id)
    # logger.info("Basic agent demo complete.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. FUNCTION TOOLS
# ─────────────────────────────────────────────────────────────────────────────

def get_weather(location: str, unit: str = "celsius") -> str:
    """
    Get the current weather for a given location.

    Args:
        location: City and country, e.g. 'London, UK'.
        unit: Temperature unit — 'celsius' or 'fahrenheit'.

    Returns:
        JSON string with temperature, condition, and humidity.
    """
    data = {
        "location":    location,
        "temperature": 18 if unit == "celsius" else 64,
        "unit":        unit,
        "condition":   "Partly cloudy",
        "humidity":    "62%",
    }
    return json.dumps(data)


def search_knowledge_base(query: str, top_k: int = 3) -> str:
    """
    Search an internal knowledge base for relevant documents.

    Args:
        query: Natural language search query.
        top_k: Number of results to return (1 to 10).

    Returns:
        JSON string with a list of matching document snippets.
    """
    results = [
        {"id": f"doc_{i}", "snippet": f"Result {i} for: {query}", "score": round(1.0 - i * 0.1, 2)}
        for i in range(1, top_k + 1)
    ]
    return json.dumps({"results": results})


def send_email(to: str, subject: str, body: str) -> str:
    """
    Send an email to a recipient.

    Args:
        to: Recipient email address.
        subject: Email subject line.
        body: Plain-text email body.

    Returns:
        JSON string with send status and message ID.
    """
    # logger.info("Sending email to %s: %s", to, subject)
    return json.dumps({"status": "sent", "message_id": f"msg_{int(time.time())}"})


def pk_function_tools(client: AgentsClient) -> None:
    """Agent with FunctionTool — auto-executes tool calls via create_and_process."""
    model = os.environ["MODEL_DEPLOYMENT_NAME"]

    fn_tool = FunctionTool(functions={get_weather, search_knowledge_base, send_email})
    toolset = ToolSet()
    toolset.add(fn_tool)

    # Register so the client auto-executes tool calls
    client.enable_auto_function_calls(toolset)

    agent = client.create_agent(
        model=model,
        name="tool-agent",
        instructions=(
            "You are a helpful assistant with weather, knowledge base, and email tools. "
            "Use tools when relevant."
        ),
        tools=fn_tool.definitions,
    )

    thread = client.threads.create()
    client.messages.create(
        thread_id=thread.id,
        role=MessageRole.USER,
        content="What is the weather in Tokyo? Also search the knowledge base for 'Azure pricing'.",
    )

    # create_and_process() DOES accept toolset=
    run = client.runs.create_and_process(
        thread_id=thread.id,
        agent_id=agent.id,
        toolset=toolset,
    )

    if run.status == RunStatus.COMPLETED:
        last = client.messages.get_last_message_text_by_role(
            thread_id=thread.id, role=MessageRole.AGENT
        )
        print("\n[Function Tools]\n", last.text.value if last else "(no response)")

    client.threads.delete(thread.id)
    client.delete_agent(agent.id)
    # logger.info("Function tools demo complete.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. CODE INTERPRETER
# ─────────────────────────────────────────────────────────────────────────────

def pk_code_interpreter(client: AgentsClient, csv_path: str | None = None) -> None:
    """Agent with CodeInterpreterTool. Saves any generated images locally."""
    model = os.environ["MODEL_DEPLOYMENT_NAME"]

    ci_tool = CodeInterpreterTool()
    file_ids: list[str] = []

    if csv_path and Path(csv_path).exists():
        with open(csv_path, "rb") as f:
            uploaded = client.files.upload(
                file=f,
                purpose="assistants"
            )
        file_ids.append(uploaded.id)
        # logger.info("Uploaded file: %s -> %s", csv_path, uploaded.id)

    agent = client.create_agent(
        model=model,
        name="data-analyst",
        instructions="You are a data analyst. Use Python to analyse data and create visualisations.",
        tools=ci_tool.definitions,
        tool_resources=ci_tool.resources,
    )

    thread = client.threads.create()

    # FIX 1: use file_ids[0] not the undefined name 'fid'
    attachments = (
        [MessageAttachment(file_id=file_ids[0], tools=[CodeInterpreterToolDefinition()])]
        if file_ids else None
    )

    client.messages.create(
        thread_id=thread.id,
        role=MessageRole.USER,
        content=(
            "Generate a bar chart of [10, 25, 17, 38, 22] labelled Mon-Fri and return the image."
            if not file_ids
            else "Summarise this CSV and plot a trend line for the first numeric column."
        ),
        attachments=attachments,
    )

    run = client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

    if run.status == RunStatus.COMPLETED:
        for msg in client.messages.list(thread_id=thread.id):
            if msg.role == MessageRole.AGENT:
                for block in msg.content:
                    if isinstance(block, MessageTextContent):
                        print("\n[Code Interpreter]\n", block.text.value)
                    elif isinstance(block, MessageImageFileContent):
                        # FIX 2: join chunks in case get_content returns an iterator
                        img_data = client.files.get_content(block.image_file.file_id)
                        img_bytes = b"".join(img_data)
                        out = f"output_{block.image_file.file_id}.png"
                        with open(out, "wb") as f:
                            f.write(img_bytes)
                        # logger.info("Image saved -> %s", out)
    else:
        # logger.warning("Run did not complete. Status: %s", run.status)
        if hasattr(run, "last_error"):
            # logger.error("Run error: %s", run.last_error)
            pass

    # Cleanup
    client.threads.delete(thread.id)
    client.delete_agent(agent.id)
    for fid in file_ids:
        client.files.delete(fid)
    # logger.info("Code interpreter demo complete.")


# ─────────────────────────────────────────────────────────────────────────────
# 5. FILE SEARCH / RAG
# ─────────────────────────────────────────────────────────────────────────────

def pk_file_search(client: AgentsClient, file_paths: list[str]) -> None:
    """Upload documents to a vector store and query them with FileSearchTool."""
    model = os.environ["MODEL_DEPLOYMENT_NAME"]

    file_ids: list[str] = []
    for path in file_paths:
        if not Path(path).exists():
            logger.warning("Skipping missing file: %s", path)
            continue
        uploaded = client.files.upload_and_poll(
            file_path=path,
            purpose=FilePurpose.ASSISTANTS,
        )
        file_ids.append(uploaded.id)

    if not file_ids:
        logger.error("No files uploaded. Skipping.")
        return

    vector_store = client.vector_stores.create_and_poll(
        name="project-docs",
        file_ids=file_ids,
    )
    logger.info("Vector store ready: %s", vector_store.id)

    fs_tool = FileSearchTool(vector_store_ids=[vector_store.id])

    agent = client.create_agent(
        model=model,
        name="docs-agent",
        instructions="Answer using only the uploaded documents. Always cite the source.",
        tools=fs_tool.definitions,
        tool_resources=fs_tool.resources,
    )

    thread = client.threads.create()
    client.messages.create(
        thread_id=thread.id,
        role=MessageRole.USER,
        content="Summarise the key points from the uploaded documents.",
    )

    run = client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

    if run.status == RunStatus.COMPLETED:
        for msg in client.messages.list(thread_id=thread.id):
            if msg.role == MessageRole.AGENT:
                for block in msg.content:
                    if isinstance(block, MessageTextContent):
                        print("\n[File Search]\n", block.text.value)
                        for ann in block.text.annotations:
                            if hasattr(ann, "file_citation"):
                                print(f"  Citation: {ann.file_citation.file_id}")

    client.threads.delete(thread.id)
    client.delete_agent(agent.id)
    client.vector_stores.delete(vector_store.id)
    for fid in file_ids:
        client.files.delete(fid)
    # logger.info("File search demo complete.")


# ─────────────────────────────────────────────────────────────────────────────
# 6. STREAMING
# ─────────────────────────────────────────────────────────────────────────────

class PrintingEventHandler(AgentEventHandler):
    """Streams tokens to stdout as they arrive."""

    def on_message_delta(self, delta: MessageDeltaChunk):
        for block in delta.delta.content:
            if isinstance(block, MessageDeltaTextContent) and block.text:
                print(block.text.value, end="", flush=True)

    def on_thread_run(self, run: ThreadRun):
        if run.status == RunStatus.FAILED:
            print(f"\n[FAILED] {run.last_error}")
        elif run.status == RunStatus.COMPLETED:
            print("\n[Complete]")

    def on_error(self, data: str):
        print(f"\n[Stream error] {data}")

    def on_done(self):
        pass


def pk_streaming(client: AgentsClient) -> None:
    """
    Stream tokens in real-time.

    RULE: runs.stream() does NOT accept toolset=.
    For tool use with streaming, call client.enable_auto_function_calls(toolset) first.
    """
    model = os.environ["MODEL_DEPLOYMENT_NAME"]

    agent = client.create_agent(
        model=model,
        name="streaming-agent",
        instructions="You are a creative storyteller. Write vivid, engaging responses.",
    )

    thread = client.threads.create()
    client.messages.create(
        thread_id=thread.id,
        role=MessageRole.USER,
        content="Write a short paragraph about the future of UNESCO.",
    )

    print("\n[Streaming Response]")
    # No toolset= here — stream() does not accept it
    with client.runs.stream(
        thread_id=thread.id,
        agent_id=agent.id,
        event_handler=PrintingEventHandler(),
    ) as stream:
        stream.until_done()

    client.threads.delete(thread.id)
    client.delete_agent(agent.id)
    # logger.info("Streaming demo complete.")


# ─────────────────────────────────────────────────────────────────────────────
# 7. MULTI-AGENT ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────

def pk_multi_agent(client: AgentsClient) -> None:
    """Orchestrator delegates to researcher + writer via ConnectedAgentTool."""
    model = os.environ["MODEL_DEPLOYMENT_NAME"]

    researcher = client.create_agent(
        model=model,
        name="researcher",
        instructions="You are a research specialist. Find facts and summarise information accurately.",
    )

    writer = client.create_agent(
        model=model,
        name="writer",
        instructions="You are a technical writer. Turn research into clear, structured prose.",
    )

    research_tool = ConnectedAgentTool(
        id=researcher.id,
        name="researcher",
        description="Use this agent to research facts and gather information on any topic.",
    )
    writer_tool = ConnectedAgentTool(
        id=writer.id,
        name="writer",
        description="Use this agent to write polished documents from research notes.",
    )

    orchestrator = client.create_agent(
        model=model,
        name="orchestrator",
        instructions=(
            "Coordinate tasks:\n"
            "1. Use 'researcher' to gather information.\n"
            "2. Use 'writer' to produce the final document.\n"
            "3. Return the writer's polished text."
        ),
        tools=research_tool.definitions + writer_tool.definitions,
    )

    thread = client.threads.create()
    client.messages.create(
        thread_id=thread.id,
        role=MessageRole.USER,
        content="Write a two-paragraph briefing on the benefits of  AI in Retail sector.",
    )

    run = client.runs.create_and_process(
        thread_id=thread.id,
        agent_id=orchestrator.id,
    )

    if run.status == RunStatus.COMPLETED:
        last = client.messages.get_last_message_text_by_role(
            thread_id=thread.id, role=MessageRole.AGENT
        )
        print("\n[Multi-Agent]\n", last.text.value if last else "(no response)")

    for aid in [orchestrator.id, researcher.id, writer.id]:
        client.delete_agent(aid)
    client.threads.delete(thread.id)
    # logger.info("Multi-agent demo complete.")


# ─────────────────────────────────────────────────────────────────────────────
# 8. MULTI-TURN CONVERSATION
# ─────────────────────────────────────────────────────────────────────────────

def pk_multi_turn(client: AgentsClient) -> None:
    """Stateful conversation across multiple user messages in one thread."""
    model = os.environ["MODEL_DEPLOYMENT_NAME"]

    agent = client.create_agent(
        model=model,
        name="conversation-agent",
        instructions="You have an excellent memory. Remember everything the user tells you.",
    )

    thread = client.threads.create(
        messages=[
            ThreadMessageOptions(
                role=MessageRole.USER,
                content="My name is Praveen and I work on the Azure product team.",
            )
        ]
    )

    turns = [
        "What do you know about me so far?",
        "I'm interested in building AI agents. What would you recommend?",
        "Summarise our conversation in one sentence.",
    ]

    for i, user_msg in enumerate(turns, 1):
        client.messages.create(thread_id=thread.id, role=MessageRole.USER, content=user_msg)
        run = client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

        if run.status == RunStatus.COMPLETED:
            last = client.messages.get_last_message_text_by_role(
                thread_id=thread.id, role=MessageRole.AGENT
            )
            print(f"\n[Turn {i}] User:  {user_msg}")
            print(f"[Turn {i}] Agent: {last.text.value if last else '(no response)'}")

    client.threads.delete(thread.id)
    client.delete_agent(agent.id)
    # logger.info("Multi-turn demo complete.")


# ─────────────────────────────────────────────────────────────────────────────
# 9. STRUCTURED OUTPUT — JSON mode
# ─────────────────────────────────────────────────────────────────────────────

def pk_structured_output(client: AgentsClient) -> None:
    """Agent that always returns valid JSON, parsed and pretty-printed."""
    model = os.environ["MODEL_DEPLOYMENT_NAME"]

    agent = client.create_agent(
        model=model,
        name="json-agent",
        instructions=(
            'Always respond with valid JSON only. No prose, no markdown fences. '
            'Schema: {"answer": string, "confidence": float (0-1), "sources": [string]}'
        ),
        response_format=AgentsResponseFormat(type="json_object"),
    )

    thread = client.threads.create()
    client.messages.create(
        thread_id=thread.id,
        role=MessageRole.USER,
        content="What is the capital of France?",
    )

    run = client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

    if run.status == RunStatus.COMPLETED:
        last = client.messages.get_last_message_text_by_role(
            thread_id=thread.id, role=MessageRole.AGENT
        )
        raw = last.text.value if last else ""
        try:
            print("\n[Structured Output]")
            print(json.dumps(json.loads(raw), indent=2))
        except json.JSONDecodeError as exc:
            logger.error("JSON parse error: %s\nRaw: %s", exc, raw)

    client.threads.delete(thread.id)
    client.delete_agent(agent.id)
    # logger.info("Structured output demo complete.")


# ─────────────────────────────────────────────────────────────────────────────
# 10. FULL PIPELINE — FunctionTool + CodeInterpreter + Streaming
#
# KEY: enable_auto_function_calls() registers the toolset on the client.
#      Then stream() picks it up internally. NEVER pass toolset= to stream().
# ─────────────────────────────────────────────────────────────────────────────

def pk_full_pipeline(client: AgentsClient) -> None:
    """FunctionTool + CodeInterpreter combined, run with real-time streaming."""
    model = os.environ["MODEL_DEPLOYMENT_NAME"]

    fn_tool = FunctionTool(functions={get_weather})
    ci_tool = CodeInterpreterTool()

    toolset = ToolSet()
    toolset.add(fn_tool)
    toolset.add(ci_tool)

    # Register tool auto-execution on the client BEFORE calling stream()
    # This is the ONLY way to use tools with streaming — do NOT pass toolset= to stream()
    client.enable_auto_function_calls(toolset)

    agent = client.create_agent(
        model=model,
        name="pipeline-agent",
        instructions=(
            "You have access to weather data and a Python code sandbox. "
            "Use the right tool for each part of a request."
        ),
        tools=fn_tool.definitions + ci_tool.definitions,
        tool_resources=ci_tool.resources,
    )

    thread = client.threads.create()
    client.messages.create(
        thread_id=thread.id,
        role=MessageRole.USER,
        content=(
            "Get the weather in Paris, then write and run Python code "
            "to convert the temperature from Celsius to Fahrenheit."
        ),
    )

    print("\n[Full Pipeline - Streaming]")
    # ✓ No toolset= kwarg — stream() does not accept it
    # ✓ Tool calls are handled via enable_auto_function_calls() registered above
    with client.runs.stream(
        thread_id=thread.id,
        agent_id=agent.id,
        event_handler=PrintingEventHandler(),
    ) as stream:
        stream.until_done()

    client.threads.delete(thread.id)
    client.delete_agent(agent.id)
    #  logger.info("Full pipeline demo complete.")


# ─────────────────────────────────────────────────────────────────────────────
# 11. ERROR HANDLING — resilient run with exponential-backoff retry
# ─────────────────────────────────────────────────────────────────────────────

def safe_run(
    client: AgentsClient,
    thread_id: str,
    agent_id: str,
    toolset: ToolSet | None = None,
    max_retries: int = 3,
) -> ThreadRun | None:
    """
    Run with retry on rate limits and transient errors.
    Returns ThreadRun on success, None after exhausting retries.
    """
    for attempt in range(max_retries):
        try:
            run = client.runs.create_and_process(
                thread_id=thread_id,
                agent_id=agent_id,
                toolset=toolset,          # create_and_process DOES accept toolset
            )

            if run.status == RunStatus.COMPLETED:
                return run

            if run.status == RunStatus.FAILED:
                err = run.last_error
                # logger.error("Run failed [%s]: %s", err.code, err.message)
                if err.code == "rate_limit_exceeded":
                    time.sleep(2 ** attempt)
                    continue
                return None

            if run.status in (RunStatus.CANCELLED, RunStatus.EXPIRED):
                # logger.warning("Run ended: %s", run.status)
                return None

        except HttpResponseError as exc:
            # logger.error("HTTP %s: %s", exc.status_code, exc.message)
            if exc.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            raise

        except ServiceRequestError as exc:
            logger.warning("Network error (attempt %d): %s", attempt + 1, exc)
            time.sleep(1)
            continue

    # logger.error("Exhausted %d retries.", max_retries)
    return None


def cleanup(
    client: AgentsClient,
    *,
    agent_ids:        list[str] | None = None,
    thread_ids:       list[str] | None = None,
    file_ids:         list[str] | None = None,
    vector_store_ids: list[str] | None = None,
) -> None:
    """Bulk-delete resources. Logs errors but does not re-raise."""
    for aid in (agent_ids or []):
        try:
            client.delete_agent(aid)
        except HttpResponseError as e:
            # logger.warning("Could not delete agent %s: %s", aid, e.message)
            pass
    for tid in (thread_ids or []):
        try:
            client.threads.delete(tid)
        except HttpResponseError as e:
            # logger.warning("Could not delete thread %s: %s", tid, e.message)
            pass
    for fid in (file_ids or []):
        try:
            client.files.delete(fid)
        except HttpResponseError as e:
            # logger.warning("Could not delete file %s: %s", fid, e.message)
            pass
    for vid in (vector_store_ids or []):
        try:
            client.vector_stores.delete(vid)
        except HttpResponseError as e:
            # logger.warning("Could not delete vector store %s: %s", vid, e.message)
            pass


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    client = get_client()

        # bar = "=" * 56
        # print(f"\n{bar}")
        # print("  Microsoft Agent Framework  |  azure-ai-agents 1.1.0")
        # print(bar)

    print("\n-- 1. Basic Agent --")
    pk_basic_agent(client)

    print("\n-- 2. Function Tools --")
    pk_function_tools(client)

    print("\n-- 3. Code Interpreter --")
    pk_code_interpreter(client, csv_path="synthetic_500_quarterly_results.csv")  # Provide a real CSV path or set to None to skip file upload    

    # Provide real file paths to test RAG:
    # print("\n-- 4. File Search / RAG --")
    # pk_file_search(client, file_paths=["manual.pdf", "faq.txt"])

    print("\n-- 5. Streaming --")
    pk_streaming(client)

    print("\n-- 6. Multi-Agent Orchestration --")
    pk_multi_agent(client)

    print("\n-- 7. Multi-Turn Conversation --")
    pk_multi_turn(client)

    print("\n-- 8. Structured Output (JSON mode) --")
    pk_structured_output(client)

    print("\n-- 9. Full Pipeline (function + code interpreter + streaming) --")
    pk_full_pipeline(client)

    # print(f"\n{bar}")
    print("  All demos complete.")
    # print(f"{bar}\n")


if __name__ == "__main__":
    main()
