# SEARCH-API

A Flask-based RAG (Retrieval-Augmented Generation) API service that combines vector search with LLM-powered responses.

## Overview

The Search API provides a bridge between:

1. User queries submitted via REST API
2. External vector search service that retrieves relevant documents
3. LLM integration that generates context-aware responses

The service supports two LLM providers:

- **Ollama**: For local development and testing using open-source models
- **Azure OpenAI**: For production deployments using Azure's managed 


OpenAI service

For detailed documentation, see [DOCUMENTATION.md](./DOCUMENTATION.md).

## Getting Started

### Development Environment

* Install the following:
  * [Python](https://www.python.org/)
  * [Docker](https://www.docker.com/)
  * [Docker-Compose](https://docs.docker.com/compose/install/)
* Install Dependencies
  * Run `make setup` in the root of the project (search-api)
* Configure your environment variables (see below)
* Run the application
  * Run `make run` in the root of the project

## LLM Integration Options

### Option 1: Local Development with Ollama

For local development and testing, you can use Ollama to run open-source models locally:

1. Install Ollama following instructions at [Ollama.ai](https://ollama.ai)
2. Configure the following environment variables:

```shell
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:0.5b  # or your preferred model
LLM_HOST=http://localhost:11434
```

3. Pull your chosen model using Ollama (e.g., `ollama pull qwen2.5:0.5b`)

### Option 2: Azure OpenAI Integration

For production deployments, you can use Azure OpenAI Service through a private endpoint:

1. Ensure your application is deployed in a VNet with access to the Azure OpenAI private endpoint
2. Configure the following environment variables:

```shell
LLM_PROVIDER=openai
AZURE_OPENAI_API_KEY=[your-api-key]
AZURE_OPENAI_ENDPOINT=[your-private-endpoint-url]  # e.g., https://{your-resource-name}.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=[your-model-deployment-name]  # e.g., gpt-4, gpt-35-turbo
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

3. Verify network connectivity to the private endpoint

For detailed configuration options for either integration, see [DOCUMENTATION.md](./DOCUMENTATION.md).

## Environment Variables

The development scripts for this application allow customization via an environment file in the root directory called `.env`. See an example of the environment variables that can be overridden in `sample.env`.

Key environment variables include:

Common settings:
* `VECTOR_SEARCH_API_URL`: URL for the external vector search service
* `LLM_PROVIDER`: Choice of LLM provider ('ollama' or 'openai')
* `LLM_TEMPERATURE`: Temperature parameter for LLM generation (default: 0.3)
* `LLM_MAX_TOKENS`: Maximum tokens for LLM response (default: 1000)
* `LLM_MAX_CONTEXT_LENGTH`: Maximum context length for LLM (default: 8192)

Ollama-specific settings:
* `LLM_MODEL`: Ollama model to use (e.g., 'qwen2.5:0.5b')
* `LLM_HOST`: Ollama API host URL (default: http://localhost:11434)

Azure OpenAI settings:
* `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key
* `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI endpoint URL
* `AZURE_OPENAI_DEPLOYMENT`: Model deployment name
* `AZURE_OPENAI_API_VERSION`: API version (default: 2024-02-15-preview)

## Commands

### Development

The following commands support various development scenarios and needs.
Before running the following commands run `. venv/bin/activate` to enter into the virtual env.

> `make run`
>
> Runs the python application and runs database migrations.  
Open [http://localhost:5000/api](http://localhost:5000/api) to view it in the browser.
> The page will reload if you make edits.
> You will also see any lint errors in the console.
>
> `make test`
>
> Runs the application unit tests
>
> `make lint`
>
> Lints the application code.

## Debugging in the Editor

### Visual Studio Code

Ensure the latest version of [VS Code](https://code.visualstudio.com) is installed.

The [`launch.json`](.vscode/launch.json) is already configured with a launch task (SEARCH-API Launch) that allows you to launch chrome in a debugging capacity and debug through code within the editor.
