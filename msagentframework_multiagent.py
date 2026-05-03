import asyncio
import os
from agent_framework.foundry import FoundryChatClient
from agent_framework.orchestrations import SequentialBuilder
from azure.identity import AzureCliCredential
from dotenv import load_dotenv

load_dotenv()

async def main():
    client = FoundryChatClient(
        project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
        model=os.environ["MODEL_DEPLOYMENT_NAME"],
        credential=AzureCliCredential(),
    )

    researcher = client.as_agent(
        name="Researcher",
        instructions="Research the given topic and provide key facts. Be concise.",
    )

    writer = client.as_agent(
        name="Writer",
        instructions="Take research notes and turn them into a polished 3-sentence summary.",
    )

    # Build a sequential workflow: researcher output feeds into writer
    workflow = SequentialBuilder(participants=[researcher, writer]).build()
    workflow_agent = workflow.as_agent(name="ResearchPipeline")

    async for update in workflow_agent.run(
        "The impact of AI in Retail sector",
        stream=True,
    ):
        if update.text:
            print(update.text, end="", flush=True)
    print()

asyncio.run(main())