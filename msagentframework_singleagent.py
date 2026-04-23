# simple_agent.py
import asyncio
import os
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from azure.identity import AzureCliCredential
from dotenv import load_dotenv

load_dotenv()


async def main():
    client = FoundryChatClient(
        project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
        model=os.environ["FOUNDRY_MODEL"],
        credential=AzureCliCredential(),
    )

    agent = client.as_agent(
        name="AssistantAgent",
        instructions="You are a helpful assistant.",
    )

    result = await agent.run("Explain what Microsoft Foundry does in 2 sentences.")
    print(result)

asyncio.run(main())