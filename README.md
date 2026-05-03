# MSFoundryMSAFramework

A collection of Microsoft Foundry / Azure AI agent examples demonstrating single-agent chat, multi-turn sessions, multi-agent workflows, and a full end-to-end implementation.

## Repository Contents

- `msagentframework_all_implementation.py`: Full demonstration of the Azure AI Agents SDK, including agent creation, threading, function tools, file uploads, code interpreter, vector store creation, and cleanup.
- `msagentframework_multiagent.py`: Multi-agent workflow example using `FoundryChatClient` and a sequential pipeline of agents.
- `msagentframework_multiturn_conversation.py`: Session-based multi-turn conversation example showing stateful interaction across multiple user prompts.
- `msagentframework_singleagent.py`: Minimal single-agent chat example for quick testing.
- `synthetic_500_quarterly_results.csv`: Sample CSV used by the code interpreter example in the full implementation.

## Prerequisites

- Python 3.8 or higher
- Azure account with Foundry project access
- Registered model deployment available in the Foundry project

## Environment Setup

Create a `.env` file in the project root with the following values:

```env
FOUNDRY_PROJECT_ENDPOINT=https://<aiservices-id>.services.ai.azure.com/api/projects/<project-name>
MODEL_DEPLOYMENT_NAME=gpt-4o
```

## Install Dependencies

Install the required Python packages:

```powershell
pip install azure-ai-agents azure-ai-projects azure-identity python-dotenv agent-framework
```

> If you already have a `requirements.txt` file, you can also install dependencies with `pip install -r requirements.txt`.

## Running Examples

Run any demo script directly from the project root.

- Single agent example:
  ```powershell
  python msagentframework_singleagent.py
  ```

- Multi-turn conversation example:
  ```powershell
  python msagentframework_multiturn_conversation.py
  ```

- Multi-agent workflow example:
  ```powershell
  python msagentframework_multiagent.py
  ```

- Full implementation demo:
  ```powershell
  python msagentframework_all_implementation.py
  ```

## Notes

- The full implementation script uses the sample file `synthetic_500_quarterly_results.csv` for data analysis and code interpreter demos.
- The project relies on Azure Foundry credentials via `DefaultAzureCredential` or `AzureCliCredential` depending on the example.
- Remove or ignore local virtual environment folders such as `.pkvenv` if needed.

## Contributing

Contributions are welcome. Please follow standard GitHub workflow:

1. Fork the repository
2. Create a feature branch
3. Open a pull request with a clear description of your changes

## License

This repository is provided as-is for demonstration and experimentation purposes.

