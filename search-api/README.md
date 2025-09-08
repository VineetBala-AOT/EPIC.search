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

## API Coverage Status

🎉 **COMPLETE PARITY ACHIEVED!** The Search API now exposes **13/13 Vector API endpoints (100% coverage)** with full MCP server integration support.

### Vector API Endpoint Coverage: ✅ 13/13 (100%)

- **Search Operations**: 2/2 (100%) - Query, Document similarity
- **Discovery Operations**: 6/6 (100%) - Projects, Document types, Search strategies, Inference options, API capabilities
- **Statistics Operations**: 3/3 (100%) - Processing stats, Project details, System summary
- **Health Operations**: 2/2 (100%) - Health check, Readiness check

### Key Endpoints

- `POST /api/search/query` - Main search with LLM synthesis
- `POST /api/search/document-similarity` - Document-level similarity
- `GET /api/tools/*` - Discovery endpoints for UI metadata and MCP integration
- `GET /api/stats/*` - Processing statistics and health monitoring
- `GET /healthz`, `GET /readyz` - Public health endpoints

### MCP Server Integration Ready

Our VectorSearchClient provides complete Vector API coverage with **5 essential MCP tools** for agentic search functionality.

**Key Features:**

- 🛡️ **Query Relevance Validation** - LLM-powered EAO scope validation
- 🔍 **Smart Filter Extraction** - AI-powered project/document type detection
- ⚡ **Search Strategy Optimization** - Intelligent search approach recommendations
- 📊 **Dynamic Metadata Discovery** - Real-time project and document type lookup

For complete MCP tool details, see: [`src/mcp_server/COMPLETE_TOOLS_REFERENCE.md`](src/mcp_server/COMPLETE_TOOLS_REFERENCE.md)

See [DOCUMENTATION.md](./DOCUMENTATION.md) for complete endpoint details, implementation status, and MCP integration information.

## Getting Started

### Development Environment

#### Install the following

- [Python](https://www.python.org/)
- [Docker](https://www.docker.com/)
- [Docker-Compose](https://docs.docker.com/compose/install/)
- Install Dependencies
- Run `make setup` in the root of the project (search-api)
- Configure your environment variables (see below)
- Run the application
- Run `make run` in the root of the project

### Deployment

The Search API uses **environment-aware MCP integration** that automatically adapts to deployment context:

- **Local Development**: Uses subprocess MCP server for easy debugging
- **Container/Azure**: Uses direct integration for better reliability and performance
- **Auto-detection**: No manual configuration required

For detailed deployment instructions including Azure App Service, container deployment, and environment configuration, see [DOCUMENTATION.md](./DOCUMENTATION.md#deployment-guide).

## LLM Integration Options

### Option 1: Local Development with Ollama

For local development and testing, you can use Ollama to run open-source models locally:

1.1. Install Ollama following instructions at [Ollama.ai](https://ollama.ai)
1.2. Configure the following environment variables:

```shell
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:0.5b  # or your preferred model
LLM_HOST=http://localhost:11434
```

1.3. Pull your chosen model using Ollama (e.g., `ollama pull qwen2.5:0.5b`)

### Option 2: Azure OpenAI Integration

For production deployments, you can use Azure OpenAI Service through a private endpoint:

2.1. Ensure your application is deployed in a VNet with access to the Azure OpenAI private endpoint
2.2. Configure the following environment variables:

```shell
LLM_PROVIDER=openai
AZURE_OPENAI_API_KEY=[your-api-key]
AZURE_OPENAI_ENDPOINT=[your-private-endpoint-url]  # e.g., https://{your-resource-name}.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=[your-model-deployment-name]  # e.g., gpt-4, gpt-35-turbo
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

2.3. Verify network connectivity to the private endpoint

For detailed configuration options for either integration, see [DOCUMENTATION.md](./DOCUMENTATION.md).

## Environment Variables

The development scripts for this application allow customization via an environment file in the root directory called `.env`. See an example of the environment variables that can be overridden in `sample.env`.

Key environment variables include:

Common settings:

- `VECTOR_SEARCH_API_URL`: URL for the external vector search service
- `LLM_PROVIDER`: Choice of LLM provider ('ollama' or 'openai')
- `LLM_TEMPERATURE`: Temperature parameter for LLM generation (default: 0.3)
- `LLM_MAX_TOKENS`: Maximum tokens for LLM response (default: 1000)
- `LLM_MAX_CONTEXT_LENGTH`: Maximum context length for LLM (default: 8192)

Ollama-specific settings:

- `LLM_MODEL`: Ollama model to use (e.g., 'qwen2.5:0.5b')
- `LLM_HOST`: Ollama API host URL (default: [http://localhost:11434](http://localhost:11434))

Azure OpenAI settings:

- `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key
- `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI endpoint URL
- `AZURE_OPENAI_DEPLOYMENT`: Model deployment name
- `AZURE_OPENAI_API_VERSION`: API version (default: 2024-02-15-preview)

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
