import asyncio
import os
from agent_framework.foundry import FoundryChatClient
from azure.identity import AzureCliCredential
from dotenv import load_dotenv

load_dotenv()
async def main():
    client = FoundryChatClient(
        project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
        model=os.environ["MODEL_DEPLOYMENT_NAME"],
        credential=AzureCliCredential(),
    )

    agent = client.as_agent(
        name="ChatAgent",
        instructions="You are a friendly conversational assistant.",
    )

    # create_session() tracks conversation history across turns
    session = agent.create_session()

    questions = [
        "My name is Praveen. Remember that.",
        "What is the capital of Bihar?",
        "What is my name?",  # tests memory
    ]

    for question in questions:
        print(f"User: {question}")
        response = await agent.run(question, session=session)
        print(f"Agent: {response}\n")

asyncio.run(main())