# EPIC.search - Embedder

The EPIC.search Embedder is a document processing system that converts PDF documents into vector embeddings stored in a PostgreSQL database with pgvector. This enables semantic search capabilities across your document corpus.

## Installation

### Virtual Environment Setup (Recommended)

It's recommended to run this project in a Python virtual environment:

1. Create a virtual environment in the `.venv` directory:

   ```powershell
   python -m venv .venv
   ```

2. Activate the virtual environment:

   ```powershell
   cd .venv\Scripts
   .\activate
   cd ..\..
   ```

3. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

Note: Always ensure you're in the virtual environment when running scripts. You'll see `(.venv)` in your terminal prompt when it's active.

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

```powershell
# Process a specific project
python main.py --project_id <project_id>

# Process all available projects
python main.py

# Skip creation of HNSW vector indexes (faster startup, less resource usage)
python main.py --skip-hnsw-indexes

# Combine with other options
python main.py --project_id <project_id> --skip-hnsw-indexes

# High-performance server runs (examples)
# 32-core server optimized for maximum throughput
FILES_CONCURRENCY_SIZE=32 KEYWORD_EXTRACTION_WORKERS=6 python main.py

# 64-core server with extreme parallelism  
FILES_CONCURRENCY_SIZE=64 KEYWORD_EXTRACTION_WORKERS=8 python main.py

# Background processing with nohup (server deployment)
nohup python -u main.py --project_id <project_id> > output.log 2>&1 &
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

## Overview

The Embedder performs the following operations:

- Retrieves documents from S3 storage
- **Validates PDF content** and intelligently skips scanned/image-based documents
- Converts PDF content to searchable text  
- Splits documents into manageable chunks
- Creates vector embeddings for each chunk
- Stores embeddings in a vector database with rich metadata (including S3 keys)
- Extracts and indexes document tags and keywords (5 per chunk for focused results)
- Stores complete project metadata as JSONB for analytics
- **Comprehensive failure tracking**: Captures complete document metadata (PDF title, author, creator, creation date, page count, file size) even for failed processing
- Tracks processing status and detailed metrics for each document

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
- `FILES_CONCURRENCY_SIZE` - Number of documents to process in parallel (default: auto-detects CPU cores)
- `KEYWORD_EXTRACTION_WORKERS` - Number of threads per document for keyword extraction (default: 8)
- `CHUNK_INSERT_BATCH_SIZE` - Number of chunks to insert per database batch for stability (default: 25)
- `CHUNK_SIZE` - Size of text chunks in characters (default: 1000)
- `CHUNK_OVERLAP` - Number of characters to overlap between chunks (default: 200)
- `AUTO_CREATE_PGVECTOR_EXTENSION` - Whether to automatically create the pgvector extension (default: True)
- `GET_PROJECT_PAGE` - Number of projects to fetch per API call (default: 1)
- `GET_DOCS_PAGE` - Number of documents to fetch per API call (default: 1000)

### Model Configuration

The embedder uses two separate models that can be configured independently:

- `EMBEDDING_MODEL_NAME` - The model to use for document embedding (default: "all-mpnet-base-v2")
- `KEYWORD_MODEL_NAME` - The model to use for keyword extraction (default: "all-mpnet-base-v2")

A sample environment file is provided in `sample.env`. Copy this file to `.env` and update the values.

### High-Performance Server Configuration

For dedicated embedding servers with many CPU cores (16+), the application automatically scales to use all available cores for maximum throughput:

```env
# For 32-core server
FILES_CONCURRENCY_SIZE=32            # Process 32 documents simultaneously
KEYWORD_EXTRACTION_WORKERS=6         # 6 threads per document for keyword extraction
CHUNK_INSERT_BATCH_SIZE=20          # Smaller batches for database stability

# For 64-core server  
FILES_CONCURRENCY_SIZE=64            # Process 64 documents simultaneously
KEYWORD_EXTRACTION_WORKERS=8         # 8 threads per document for keyword extraction
CHUNK_INSERT_BATCH_SIZE=15          # Even smaller batches for high concurrency

# For local development (8-core)
FILES_CONCURRENCY_SIZE=4             # Conservative for development
KEYWORD_EXTRACTION_WORKERS=4         # Fewer threads per document
CHUNK_INSERT_BATCH_SIZE=50          # Larger batches for fewer connections
```

**Performance scaling:**

- **32-core server:** ~8x faster than 4-core setup
- **64-core server:** ~16x faster than 4-core setup
- **Database connections:** Pool size automatically scales with concurrency

## Architecture

The Embedder follows a modular architecture with the following key components:

- **Main Processor (`main.py`)**: Entry point and workflow orchestrator
- **Processor Service (`processor.py`)**: Manages parallel document processing
- **Loader Service (`loader.py`)**: Handles document loading and embedding
- **Logger Service (`logger.py`)**: Tracks document processing status

For detailed technical documentation, see [DOCUMENTATION.md](DOCUMENTATION.md).

## Monitoring

Processing progress is logged to the console during execution. Each document's processing status is also recorded in the database with comprehensive metrics.

### Enhanced Failure Analysis

The system captures detailed document metadata even for failed processing attempts, including:

- Complete PDF metadata (title, author, creator, creation date, format info)
- Page count and file size
- Validation status and specific failure reasons (e.g., scanned PDFs detected by content/producer analysis, corrupted files)
- Full exception details for runtime errors

This enables detailed analysis of processing patterns and identification of problematic document types.

For detailed troubleshooting, check the console output for error messages, which include specific document IDs and failure reasons. Query the `processing_logs` table for comprehensive failure analytics.

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

#### Database Connection Issues on Servers

If you encounter SSL SYSCALL errors or connection timeouts when running on Ubuntu servers:

```text
(psycopg.OperationalError) consuming input failed: SSL SYSCALL error: EOF detected
```

This indicates database connection issues during bulk operations. The application now includes:

- **Batched chunk insertions** - Large documents are split into smaller database transactions
- **Connection retry logic** - Automatic retry with exponential backoff for connection failures  
- **Improved connection pooling** - Better connection management and timeouts
- **Configurable batch sizes** - Adjust `CHUNK_INSERT_BATCH_SIZE` environment variable (default: 50)

To resolve persistent connection issues:

1. Reduce batch size: `CHUNK_INSERT_BATCH_SIZE=25`
2. Check database connection stability
3. Ensure adequate database connection limits
4. Verify network connectivity between application and database

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
