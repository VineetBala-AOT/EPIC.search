# EPIC.search - Embedder

The EPIC.search Embedder is a robust, production-grade document processing system that converts PDF documents into vector embeddings stored in a PostgreSQL database with pgvector. This enables powerful semantic search capabilities across your document corpus.

## ✨ Key Features

- **📄 Advanced PDF Processing**: Handles both regular and scanned PDF documents
- **🔍 OCR Support**: Automatic text extraction from scanned PDFs using Tesseract or Azure Document Intelligence
- **🧠 Semantic Search**: Vector embeddings for intelligent document search
- **⚡ High Performance**: Intelligent auto-configuration with parallel processing
- **🏷️ Smart Tagging**: AI-powered keyword and tag extraction
- **📊 Rich Analytics**: Comprehensive processing metrics and failure analysis
- **🔧 Production Ready**: Docker support, robust error handling, and monitoring

## 🆕 OCR Support for Scanned PDFs

The embedder now supports **Optical Character Recognition (OCR)** for scanned PDF documents that would otherwise be skipped due to lack of extractable text. Choose between:

- **🏠 Local Tesseract OCR**: Free, private, good accuracy
- **☁️ Azure Document Intelligence**: Cloud-based, excellent accuracy for complex documents

### Quick OCR Setup

```env
# Enable OCR processing
OCR_ENABLED=true

# Choose provider (tesseract or azure)
OCR_PROVIDER=tesseract

# For Azure Document Intelligence (optional)
# AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://yourresource.cognitiveservices.azure.com/
# AZURE_DOCUMENT_INTELLIGENCE_KEY=your_api_key_here
```

See the [OCR Documentation](#ocr-processing) section for detailed setup instructions.

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

# Retry documents that previously failed processing
python main.py --retry-failed

# Retry documents that were previously skipped (unsupported formats, missing OCR)
python main.py --retry-skipped

# Retry failed/skipped documents for specific project(s)
python main.py --retry-failed --project_id <project_id>
python main.py --retry-skipped --project_id <project_id>

# Shallow mode: process limited number of documents per project
python main.py --shallow 10 --project_id <project_id>
python main.py --retry-failed --shallow 5

# High-performance server runs with intelligent auto-configuration
# The embedder automatically detects hardware and optimizes settings
python main.py --project_id <project_id>

# Manual override examples (auto-configuration is usually better)
# 32-core server with manual settings
FILES_CONCURRENCY_SIZE=16 KEYWORD_EXTRACTION_WORKERS=2 python main.py

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
- **Validates PDF content** and processes scanned/image-based documents with OCR
- Converts PDF content to searchable text using standard extraction or Tesseract OCR
- Splits documents into manageable chunks
- Creates vector embeddings for each chunk
- Stores embeddings in a vector database with rich metadata (including S3 keys)
- Extracts and indexes document tags and keywords (5 per chunk for focused results)
- Stores complete project metadata as JSONB for analytics
- **Comprehensive failure tracking**: Captures complete document metadata (PDF title, author, creator, creation date, page count, file size) even for failed processing
- Tracks processing status and detailed metrics for each document
- **OCR Support**: Automatically detects and processes scanned PDFs using Tesseract OCR when available

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
- `FILES_CONCURRENCY_SIZE` - Number of documents to process in parallel (default: auto - intelligent CPU-based)
- `KEYWORD_EXTRACTION_WORKERS` - Number of threads per document for keyword extraction (default: auto - optimized for KeyBERT)
- `CHUNK_INSERT_BATCH_SIZE` - Number of chunks to insert per database batch for stability (default: 25)
- `CHUNK_SIZE` - Size of text chunks in characters (default: 1000)
- `CHUNK_OVERLAP` - Number of characters to overlap between chunks (default: 200)
- `AUTO_CREATE_PGVECTOR_EXTENSION` - Whether to automatically create the pgvector extension (default: True)
- `GET_PROJECT_PAGE` - Number of projects to fetch per API call (default: 1)
- `GET_DOCS_PAGE` - Number of documents to fetch per API call (default: 1000)

## OCR Processing

The embedder includes advanced **Optical Character Recognition (OCR)** support to process scanned PDF documents that would otherwise be skipped. Choose between two powerful OCR providers:

### 🏠 Tesseract (Local Processing)

- **Free and open source**
- **Complete privacy** - processing happens locally
- **Good accuracy** for most documents
- **No API costs** or internet required
- **100+ languages** supported

### ☁️ Azure Document Intelligence (Cloud Processing)

- **Excellent accuracy** for complex documents
- **Specialized for documents** - superior to general OCR
- **Advanced layout understanding** - preserves structure
- **High-quality text extraction** from scanned PDFs
- **Confidence scores** and metadata

### Provider Configuration

Set your preferred OCR provider via environment variables:

```env
# Core OCR Settings
OCR_ENABLED=true              # Enable/disable OCR processing
OCR_PROVIDER=tesseract        # Choose: 'tesseract' or 'azure'
OCR_DPI=300                   # Image quality (higher = better but slower)
OCR_LANGUAGE=eng              # Language code for OCR

# Tesseract Settings (when OCR_PROVIDER=tesseract)
# TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe  # Auto-detected if not set

# Azure Document Intelligence Settings (when OCR_PROVIDER=azure)
# AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://yourresource.cognitiveservices.azure.com/
# AZURE_DOCUMENT_INTELLIGENCE_KEY=your_api_key_here
```

### Tesseract Installation

#### Windows

1. Download installer from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)
2. Run installer (recommended path: `C:\Program Files\Tesseract-OCR`)
3. Add to PATH or set `TESSERACT_PATH` environment variable

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt-get install tesseract-ocr
# Install additional languages if needed:
sudo apt-get install tesseract-ocr-fra  # French
sudo apt-get install tesseract-ocr-deu  # German
```

#### macOS

```bash
brew install tesseract
# Install additional languages:
brew install tesseract-lang
```

### Azure Document Intelligence Setup

1. **Create Azure Resource:**
   - Go to [Azure Portal](https://portal.azure.com)
   - Create a "Document Intelligence" resource
   - Choose pricing tier (Free tier available)

2. **Get Credentials:**
   - Copy the **Endpoint URL**
   - Copy the **API Key** from Keys and Endpoint section

3. **Configure Environment:**

   ```env
   OCR_PROVIDER=azure
   AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://yourresource.cognitiveservices.azure.com/
   AZURE_DOCUMENT_INTELLIGENCE_KEY=your_api_key_here
   ```

### OCR Behavior

**Automatic Detection:**

- Automatically detects scanned PDFs with minimal extractable text
- Falls back to OCR processing when standard PDF extraction fails
- Logs detailed progress for multi-page documents
- Maintains document structure and metadata consistency

**Provider Comparison:**

| Feature | Tesseract | Azure Document Intelligence |
|---------|-----------|----------------------------|
| **Cost** | Free | Pay-per-use API calls |
| **Privacy** | Complete privacy | Data sent to Azure |
| **Accuracy** | Good | Excellent |
| **Speed** | Moderate | Fast (cloud processing) |
| **Setup** | Install software | Azure account + API key |
| **Internet** | Not required | Required |
| **Complex Docs** | Basic | Advanced layout understanding |

**Dependencies:**

OCR functionality requires additional packages (included in `requirements.txt`):

```txt
# For Tesseract OCR
pytesseract==0.3.13
Pillow==11.1.0

# For Azure Document Intelligence  
azure-ai-formrecognizer==3.3.0
requests==2.32.3
```

### Intelligent Auto-Configuration

The embedder features intelligent auto-configuration that optimizes performance based on your hardware:

**FILES_CONCURRENCY_SIZE Options:**

- `auto` (default) - Uses half CPU cores for 16+ core systems, all cores for smaller systems
- `auto-full` - Uses all CPU cores (maximum parallelism)
- `auto-conservative` - Uses quarter CPU cores (resource-constrained environments)
- Integer value - Manual override

**KEYWORD_EXTRACTION_WORKERS Options:**

- `auto` (default) - Optimized for KeyBERT: 2 threads for 16+ cores, 3 for 8-15 cores, 4 for <8 cores
- `auto-aggressive` - 4 threads per process (maximum keyword parallelism)
- `auto-conservative` - 1 thread per process (minimal thread contention)
- Integer value - Manual override

### Model Configuration

The embedder uses two separate models that can be configured independently:

- `EMBEDDING_MODEL_NAME` - The model to use for document embedding (default: "all-mpnet-base-v2")
- `KEYWORD_MODEL_NAME` - The model to use for keyword extraction (default: "all-mpnet-base-v2")

A sample environment file is provided in `sample.env`. Copy this file to `.env` and update the values.

### High-Performance Server Configuration

The embedder now uses intelligent auto-configuration by default, eliminating the need for manual tuning in most cases:

```env
# Recommended: Let the embedder auto-configure (works for all hardware)
FILES_CONCURRENCY_SIZE=auto          # Automatically optimizes based on CPU count
KEYWORD_EXTRACTION_WORKERS=auto      # Automatically optimizes for KeyBERT bottleneck
CHUNK_INSERT_BATCH_SIZE=50           # Good for high-RAM systems (HC44-32rs)

# Alternative auto-modes for specific scenarios
FILES_CONCURRENCY_SIZE=auto-full           # Maximum parallelism (all CPU cores)
KEYWORD_EXTRACTION_WORKERS=auto-aggressive # Maximum keyword parallelism (4 threads)

FILES_CONCURRENCY_SIZE=auto-conservative      # Resource-constrained (quarter cores)
KEYWORD_EXTRACTION_WORKERS=auto-conservative  # Minimal contention (1 thread)
```

**Hardware-specific auto-configuration results:**

- **32-core server (HC44-32rs):** auto = 16 processes × 2 threads = 32 total threads (100% CPU utilization)
- **16-core server:** auto = 8 processes × 2 threads = 16 total threads
- **8-core development machine:** auto = 8 processes × 3 threads = 24 total threads
- **4-core laptop:** auto = 4 processes × 4 threads = 16 total threads

**Performance scaling with auto-configuration:**

- **Eliminates over-parallelization** that caused 100x slowdowns in previous versions
- **Optimizes for KeyBERT bottleneck** (the main performance constraint)
- **Automatically adjusts** database connection pooling based on process count

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

### OCR Issues

#### Tesseract Not Found

```text
FileNotFoundError: [WinError 2] The system cannot find the file specified
```

**Solutions:**

1. **Install Tesseract**: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)
2. **Add to PATH**: Ensure Tesseract is in your system PATH
3. **Set explicit path**: Use `TESSERACT_PATH` environment variable:

   ```env
   TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
   ```

#### Azure Document Intelligence Errors

```text
azure.core.exceptions.ClientAuthenticationError: Invalid API key
```

**Solutions:**

1. **Verify credentials**: Check endpoint URL and API key in Azure Portal
2. **Check resource status**: Ensure Document Intelligence resource is active
3. **Validate region**: Endpoint URL should match your resource region

#### OCR Quality Issues

- **Poor text quality**: Increase `OCR_DPI` (e.g., 400-600 for high-quality scans)
- **Wrong language**: Set `OCR_LANGUAGE` to correct language code
- **Complex layouts**: Consider switching to Azure Document Intelligence for better layout understanding

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

### Retrying Failed or Skipped Documents

If documents failed processing or were skipped (e.g., due to missing OCR), you can reprocess them using retry flags:

#### Retry Failed Documents

```bash
# Retry all failed documents across all projects
python main.py --retry-failed

# Retry failed documents for specific project(s)
python main.py --retry-failed --project_id <project_id>

# Retry failed documents with limited processing per project
python main.py --retry-failed --shallow 10
```

Use `--retry-failed` when:

- OCR processing was failing but is now working
- Database connection issues have been resolved
- Model loading or embedding generation was failing

#### Retry Skipped Documents

```bash
# Retry all skipped documents across all projects
python main.py --retry-skipped

# Retry skipped documents for specific project(s)
python main.py --retry-skipped --project_id <project_id>
```

Use `--retry-skipped` when:

- OCR was not available but is now enabled/configured
- Previously unsupported document formats are now supported
- Documents were skipped due to validation issues that have been fixed

> **Note**: You cannot use `--retry-failed` and `--retry-skipped` together. Choose the appropriate retry mode based on the status of documents you want to reprocess.

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
