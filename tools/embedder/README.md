# EPIC.search - Embedder

The EPIC.search Embedder is a document processing system that converts PDF documents into vector embeddings stored in a PostgreSQL database with pgvector. This enables semantic search capabilities across your document corpus.

## Overview

The Embedder performs the following operations:

- Retrieves documents from S3 storage
- Converts PDF content to searchable text
- Splits documents into manageable chunks
- Creates vector embeddings for each chunk
- Stores embeddings in a vector database
- Extracts and indexes document tags
- Tracks processing status for each document

## Environment Variables

To run this project, you will need to add the following environment variables to your `.env` file:

### Required Environment Variables

- `DOCUMENT_SEARCH_URL` - Base URL for the EPIC.search API
- `S3_ENDPOINT_URI` - Endpoint URL for S3 storage
- `S3_BUCKET_NAME` - Name of the S3 bucket containing documents
- `S3_ACCESS_KEY_ID` - Access key for S3
- `S3_SECRET_ACCESS_KEY` - Secret key for S3
- `S3_REGION` - AWS region for S3
- `VECTOR_DB_URL` - Connection URL for the vector database
- `LOGS_DATABASE_URL` - Connection URL for the processing logs database

### Optional Environment Variables

- `DOC_TAGS_TABLE_NAME` - Table name for the document chunks with tags index (default: "document_tags")
- `DOC_CHUNKS_TABLE_NAME` - Table name for the untagged document chunks (default: "document_chunks")
- `EMBEDDING_DIMENSIONS` - Dimensions of the embedding vectors (default: 768)
- `FILES_CONCURRENCY_SIZE` - Number of files to process in parallel (default: 4)
- `CHUNK_SIZE` - Size of text chunks in characters (default: 1000)
- `CHUNK_OVERLAP` - Number of characters to overlap between chunks (default: 200)

### Model Configuration

The embedder uses two separate models that can be configured independently:

- `EMBEDDING_MODEL_NAME` - The model to use for document embedding (default: "all-mpnet-base-v2")
- `KEYWORD_MODEL_NAME` - The model to use for keyword extraction (default: "all-mpnet-base-v2")

A sample environment file is provided in `sample.env`. Copy this file to `.env` and update the values.

## Installation

### Prerequisites

- Python 3.9+
- Docker and Docker Compose (for database deployment)
- Access to an S3-compatible storage service
- PostgreSQL with pgvector extension

### Step 1: Start the Database

Run the following command in the `src/database` folder to start the database using Docker:

```bash
cd src/database
docker-compose up -d
```

### Step 2: Install Dependencies

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Run as a Python Application

You can run the embedder as a standard Python application:

```bash
# Process a specific project
python main.py --project_id <project_id>

# Process all available projects
python main.py
```

### Run as a Docker Container

You can build and run the embedder as a Docker container with several different configuration approaches:

#### Option 1: Standard Build (Models downloaded at runtime)

With this approach, both models will be downloaded when first needed during processing:

```bash
# Build the Docker image
docker build -t epic-search-embedder .

# Run the container with environment variables
docker run --env-file .env epic-search-embedder
```

#### Option 2: Preloaded Models Build (Models embedded in image)

For faster startup in production environments, you can preload one or both models into the Docker image:

```bash
# Build with both models preloaded (same model for both)
docker build -t epic-search-embedder \
  --build-arg PRELOAD_EMBEDDING_MODEL="all-mpnet-base-v2" \
  --build-arg PRELOAD_KEYWORD_MODEL="all-mpnet-base-v2" .

# Or build with different models for embedding and keyword extraction
docker build -t epic-search-embedder \
  --build-arg PRELOAD_EMBEDDING_MODEL="all-mpnet-base-v2" \
  --build-arg PRELOAD_KEYWORD_MODEL="distilbert-base-nli-stsb-mean-tokens" .
```

The preloaded model approach is recommended for production deployments as it eliminates the download delay when containers start.

## Architecture

The Embedder follows a modular architecture with the following key components:

- **Main Processor (`main.py`)**: Entry point and workflow orchestrator
- **Processor Service (`processor.py`)**: Manages parallel document processing
- **Loader Service (`loader.py`)**: Handles document loading and embedding
- **Logger Service (`logger.py`)**: Tracks document processing status

For detailed technical documentation, see [DOCUMENTATION.md](DOCUMENTATION.md).

## Monitoring

Processing progress is logged to the console during execution. Each document's processing status is also recorded in the database.

For detailed troubleshooting, check the console output for error messages, which include specific document IDs and failure reasons.

## Troubleshooting

### Known Issues

#### ProcessPoolExecutor Shutdown Error

You may occasionally see this error when the application exits:

```code
Exception ignored in: <function _ExecutorManagerThread.__init__.<locals>.weakref_cb at 0x...>
Traceback (most recent call last):
  File "...\Lib\concurrent\futures\process.py", line 310, in weakref_cb
AttributeError: 'NoneType' object has no attribute 'util'
```

This is a harmless error related to Python's multiprocessing module during interpreter shutdown and can be safely ignored. This error has been suppressed in the application but may still appear in certain environments.

## Development

### Running Tests

Run unit tests with:

```bash
pytest
```

### Preloading Models Manually

For testing purposes, you can manually preload the embedding models:

```bash
# Set the model names
export EMBEDDING_MODEL_NAME="all-mpnet-base-v2"
export KEYWORD_MODEL_NAME="all-mpnet-base-v2"

# Run the preloader
python preload_models.py
```

## License

This project is proprietary and confidential.
