![GPU Accelerated](https://img.shields.io/badge/GPU-Accelerated-green)
![CUDA 12.6](https://img.shields.io/badge/CUDA-12.6-blue)
![Python 3.12](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-blue)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

![Normalized Semantic Chunker](logo.png)

# Normalized Semantic Chunker

The Normalized Semantic Chunker is a cutting-edge tool that unlocks the full potential of semantic chunking in an expanded range of NLP applications processing text documents and splits them into semantically coherent segments while ensuring optimal chunk size for downstream NLP tasks.
This innovative solution builds upon concepts from [YouTube's Advanced Text Splitting for RAG](https://www.youtube.com/watch?v=8OJC21T2SL4&t=1930s) and implementation patterns from [LangChain's semantic chunker documentation](https://python.langchain.com/docs/how_to/semantic-chunker/).
Conventional semantic chunkers prioritize content coherence but often produce chunks with highly variable token counts. This leads to issues like context window overflow and inconsistent retrieval quality, significantly impacting token-sensitive applications such as retrieval-augmented generation (RAG).
The Normalized Semantic Chunker overcomes these challenges by combining semantic cohesion with statistical guarantees for token size compliance. It ensures chunks are not only semantically meaningful but also fall within an optimal size range in terms of token count. This enables more precise and efficient text preparation for embeddings, RAG pipelines, and other NLP applications.
Whether working with long documents, varied content structures, or token-sensitive NLP architectures, the Normalized Semantic Chunker provides a robust, adaptable solution for optimizing text segmentation.


## Key Features
-   **Adaptive Semantic Chunking**: Intelligently splits text based on semantic similarity between consecutive sentences.
-   **Precise Chunk Size Control**: Advanced algorithm statistically ensures compliance with maximum token limits.
-   **Parallel Multi-Percentile Optimization**: Efficiently searches for the optimal similarity percentile using parallel processing.
-   **Intelligent Small Chunk Management**: Automatically merges undersized chunks with their most semantically similar neighbors.
-   **Smart Oversized Chunk Handling**: Intelligently splits chunks that exceed token threshold limits while preserving semantic integrity.
-   **GPU Acceleration**: CUDA-enabled for fast embedding generation using PyTorch.
-   **Comprehensive Processing Pipeline**: From raw text to optimized chunks in a single workflow.
-   **Universal REST API with FastAPI**: Modern, high-performance API interface with automatic documentation, data validation, and seamless integration capabilities for any system or language.
-   **Optional Bearer Token Authentication**: Secure the API with a Bearer token via the `API_TOKEN` environment variable. When unset, authentication is disabled for easy local development.
-   **Docker Integration**: Easy deployment with Docker and docker-compose.
-   **Adaptive Processing**: Adjusts processing parameters based on document size for optimal resource usage.
-   **Model Caching**: Caches embedding models with timeout for improved performance.
-   **Format Support**: Handles text (.txt), markdown (.md), and structured JSON (.json) files.
-   **Resource Management**: Intelligently manages system resources based on available RAM and CPU cores.

## Table of Contents

- [How the Text Chunking Algorithm Works](#how-the-text-chunking-algorithm-works)
  - [The Pipeline](#the-pipeline)
  - [Statistical Control of Maximum Tokens Chunk Size](#statistical-control-of-maximum-tokens-chunk-size)
  - [Parallel Multi-Core Percentile Search Optimization](#parallel-multi-core-percentile-search-optimization)
  - [Comparison with Traditional Chunking](#comparison-with-traditional-chunking)
- [Advantages of the Solution](#advantages-of-the-solution)
  - [Optimal Preparation for RAG and Semantic Retrieval](#optimal-preparation-for-rag-and-semantic-retrieval)
  - [Superior Performance](#superior-performance)
  - [Flexibility and Customization](#flexibility-and-customization)
- [Installation and Deployment](#installation-and-deployment)
  - [Prerequisites](#prerequisites)
  - [Docker Image from GitHub Container Registry (Recommended)](#docker-image-from-github-container-registry-recommended)
  - [Getting the Code](#getting-the-code)
  - [Local Installation with Uvicorn](#local-installation-with-uvicorn)
  - [Building Docker Image Locally](#building-docker-image-locally)
- [Using the API](#using-the-api)
  - [API Endpoints](#api-endpoints)
  - [Authentication](#authentication)
  - [Environment Variables](#environment-variables)
  - [Input Constraints](#input-constraints)
  - [Example API Call](#example-api-call)
  - [Response Format](#response-format)
- [Contributing](#contributing)

## How the Text Chunking Algorithm Works

### The Pipeline

The core innovation of Normalized Semantic Chunker lies in its multi-step pipeline that combines NLP techniques with statistical optimization to ensure both semantic coherence and size consistency:

1. The application exposes a simple REST API endpoint where users can upload a text document and parameters for maximum token limits and embedding model selection. 
2. The text is initially split into sentences using sophisticated regex pattern matching.
3. Each sentence is transformed into a vector embedding using state-of-the-art transformer models (default: `sentence-transformers/all-MiniLM-L6-v2`).
4. The angular similarity between consecutive sentence vectors is calculated.
5. A parallel search algorithm identifies the optimal percentile of the similarity distribution that respects the specified size constraints.
6. Chunks are formed by grouping sentences across boundaries identified by the chosen percentile.
7. A post-processing step identifies and merges chunks too small with their most semantically similar neighbours, ensuring size constraints are met.
8. A final step splits any remaining chunks that exceed the maximum token limit, prioritizing sentence boundaries.
9. The application returns a well-structured JSON response containing the chunks, metadata, and performance statistics, ready for immediate integration into production environments.

### Statistical Control of Maximum Tokens Chunk Size

Unlike traditional approaches, Normalized Semantic Chunker uses a sophisticated statistical method to ensure that chunks generally stay below a maximum token limit.

During the percentile search, potential chunkings are evaluated based on an estimate of their 95th percentile token count:

```python
# Calculate the estimated 95th percentile using z-score of 1.645
estimated_95th_percentile = average_tokens + (1.645 * std_dev)
if estimated_95th_percentile <= max_tokens:
    # This percentile is considered valid
    return chunks_with_tokens, percentile, average_tokens
```

This approach ensures that approximately 95% of the generated chunks respect the specified token limit while automatically handling the few edge cases through a subsequent splitting step.

### Parallel Multi-Core Percentile Search Optimization

The algorithm leverages parallel processing to simultaneously test multiple percentiles, significantly speeding up the search for the optimal splitting point:

```python
with ProcessPoolExecutor(max_workers=max_workers) as executor:
    futures = [
        executor.submit(_process_percentile_range, args)
        for args in process_args
    ]
```

This parallel implementation allows for quickly finding the best balance between semantic cohesion and adherence to size constraints.

### Comparison with Traditional Chunking

| Feature | Traditional Chunking | Normalized Semantic Chunker |
|---------|----------------------|------------------------------|
| Boundary Determination | Fixed rules or token counts | Statistical analysis of semantic similarity distribution |
| Size Control | Often approximate or not guaranteed | Statistical guarantee (e.g., ~95%) + explicit splitting/merging |
| Semantic Cohesion | Can split related concepts | Preserves semantic cohesion via similarity analysis |
| Outlier Handling | Limited or absent | Intelligent merging of small chunks & splitting of large ones |
| Parallelization | Rarely implemented | Built-in parallel multi-core optimization |
| Adaptability | Requires manual parameter tuning | Automatically finds optimal parameters for each document type and size |

## Advantages of the Solution

### Optimal Preparation for RAG and Semantic Retrieval

Chunks generated by Normalized Semantic Chunker are ideal for Retrieval-Augmented Generation systems:

- **Semantic Coherence**: Each chunk contains semantically related information.
- **Balanced Sizes**: Chunks adhere to maximum size limits while avoiding excessively small fragments through merging.
- **Representativeness**: Each chunk aims to contain a complete and coherent unit of information.

### Superior Performance

The parallel implementation and statistical approach offer:

- **Processing Speed**: Parallel optimization on multi-core systems.
- **GPU Acceleration**: Fast embedding generation using CUDA-enabled PyTorch.
- **Scalability**: Efficient handling of large documents with adaptive processing based on document size.
- **Consistent Quality**: Predictable and reliable results regardless of text type.
- **Resource Management**: Intelligent allocation of CPU cores and memory based on document size and system resources.

### Flexibility and Customization

The algorithm adapts automatically to different types of content:

- **Adaptive Parameters**: Automatic identification of the best chunking parameters for each document.
- **Configurability**: Ability to specify custom maximum token limits (max_tokens) and control small chunk merging.
- **Extensibility**: Modular architecture easily extendable with new features.
- **Embedding Model Selection**: Switch between different transformer models based on your needs.

## Installation and Deployment

### Prerequisites

- Docker and Docker Compose (for Docker deployment)
- NVIDIA GPU with CUDA support (recommended)
- NVIDIA Container Toolkit (for GPU passthrough in Docker)
- Python 3.10-3.12 (Python 3.12 recommended for Docker deployment, Python 3.13+ not supported due to PyTorch and sentence-transformers compatibility)

### Docker Image from GitHub Container Registry (Recommended)

The easiest way to get started is using the pre-built Docker image from GitHub Container Registry. This image includes CUDA support and works on both GPU and CPU machines.

**Pull the image:**
```bash
# Latest version
docker pull ghcr.io/smart-models/normalized-semantic-chunker:latest

# Or a specific version
docker pull ghcr.io/smart-models/normalized-semantic-chunker:1.0.0
```

**Run with GPU (recommended for best performance):**
```bash
docker run --gpus all -p 8000:8000 ghcr.io/smart-models/normalized-semantic-chunker:latest
```

**Run on CPU (works on any machine):**
```bash
docker run -p 8000:8000 ghcr.io/smart-models/normalized-semantic-chunker:latest
```

**With persistent model cache (recommended):**
```bash
# Create directories for persistent storage
mkdir -p models logs

# Run with mounted volumes
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  ghcr.io/smart-models/normalized-semantic-chunker:latest
```

The API will be available at `http://localhost:8000`. Access the interactive documentation at `http://localhost:8000/docs`.

### Getting the Code

Before proceeding with any installation method, clone the repository:
```bash
git clone https://github.com/smart-models/Normalized-Semantic-Chunker.git
cd Normalized-Semantic-Chunker
```

### Local Installation with Uvicorn

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   ```
   
   **For Windows users:**
   
   * Using Command Prompt:
   ```cmd
   .venv\Scripts\activate.bat
   ```
   
   * Using PowerShell:
   ```powershell
   # If you encounter execution policy restrictions, run this once per session:
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
   
   # Then activate the virtual environment:
   .venv\Scripts\Activate.ps1
   ```
   > **Note:** PowerShell's default security settings may prevent script execution. The above command temporarily allows scripts for the current session only, which is safer than changing system-wide settings.

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Note: For GPU support, ensure you install the correct PyTorch version:
   ```bash
   pip install --extra-index-url https://download.pytorch.org/whl/cu126 torch==2.6.0+cu126
   ```

3. Run the FastAPI server:
   ```bash
   uvicorn normalized_semantic_chunker:app --reload
   ```

4. The API will be available at `http://localhost:8000`.
   
   Access the API documentation and interactive testing interface at `http://localhost:8000/docs`.

### Building Docker Image Locally

If you prefer to build the Docker image locally or need to customize it:

1. Create required directories for persistent storage:
   ```bash
   # Linux/macOS
   mkdir -p models logs
   
   # Windows CMD
   mkdir models
   mkdir logs
   
   # Windows PowerShell
   New-Item -ItemType Directory -Path models -Force
   New-Item -ItemType Directory -Path logs -Force
   # Or with PowerShell alias
   mkdir -Force models, logs
   ```

2. Deploy with Docker Compose:

   **CPU-only deployment** (default, works on all machines):
   ```bash
   cd docker
   docker compose --profile cpu up -d
   ```

   **GPU-accelerated deployment** (requires NVIDIA GPU and drivers):
   ```bash
   cd docker
   docker compose --profile gpu up -d
   ```

   **Stopping the service**:
   
   > **Important**: Always match the `--profile` flag between your `up` and `down` commands:
   ```bash
   # To stop CPU deployment
   docker compose --profile cpu down
   
   # To stop GPU deployment
   docker compose --profile gpu down
   ```
   > This ensures Docker Compose correctly identifies and manages the specific set of containers you intended to control.

   > **Note**: The GPU-accelerated deployment requires an NVIDIA GPU with appropriate drivers installed. If you don't have an NVIDIA GPU, use the CPU-only deployment.

3. The API will be available at `http://localhost:8001` (docker-compose default port).

   Access the API documentation and interactive testing interface at `http://localhost:8001/docs`.

   > **Note**: You can change the port by setting the `APP_PORT` environment variable before running docker-compose:
   > ```bash
   > APP_PORT=8000 docker compose --profile gpu up -d
   > ```

## Using the API

### API Endpoints

- **POST `/normalized_semantic_chunker/`**  
  Chunks a text document into semantically coherent segments while controlling token size.
  
  **Parameters:**
  - `file`: The text file to be chunked (supports .txt, .md, and .json formats). Must be UTF-8 encoded. Text files must contain at least 2 sentences.
  - `max_tokens`: Maximum token count per chunk (integer, required)
  - `model`: Embedding model to use for semantic analysis (string, default: `sentence-transformers/all-MiniLM-L6-v2`)
  - `merge_small_chunks`: Whether to merge undersized chunks (boolean, default: `true`)
  - `merge_passes`: Maximum number of merge passes for undersized chunks (integer, default: `3`, range: 1-5). Higher values reduce small chunks but increase processing time. Only used when `merge_small_chunks` is `true`.
  - `verbosity`: Show detailed logs (boolean, default: `false`)
  - `chunk_metadata_json`: Optional JSON string to merge into each output chunk. Useful for adding custom metadata like source document ID, category, or any other fields you need in your chunks (string, default: `null`)
  
  **Response:**
  Returns a JSON object containing:
  - `chunks`: Array of text segments with their token counts and IDs
  - `metadata`: Processing statistics including chunk count, token statistics, percentile used, model name, and processing time

  **JSON Input Format:**
  When using JSON files as input, the expected structure is:
  ```json
  {
    "chunks": [
      {
        "text": "First chunk of text content...",
        "metadata_field": "Additional metadata is allowed..."
      },
      {
        "text": "Second chunk of text content...",
        "id": 12345
      },
      ...
    ]
  }
  ```
  The service will process each text chunk individually, maintaining the chunk boundaries provided in your JSON file, then apply semantic chunking within those boundaries as needed. Additional metadata fields beyond `text` are allowed and will be ignored during processing, so you can include any extra information you need while still having the JSON process correctly.

- **GET `/`**
  Health check endpoint. Always accessible, no authentication required. Returns:
  - `status`: `"healthy"` when the service is running
  - `gpu_available`: whether a CUDA-capable GPU is detected
  - `version`: API version
  - `default_model`: name of the default embedding model
  - `model_loaded`: whether the default embedding model is loaded and ready
  - `models_in_cache`: number of models currently cached in memory

### Authentication

The API supports optional Bearer Token authentication controlled by the `API_TOKEN` environment variable.

- **Disabled (default)**: leave `API_TOKEN` unset or empty — all requests are accepted without any token.
- **Enabled**: set `API_TOKEN` to a secret value — all `POST` requests must include the header:
  ```
  Authorization: Bearer <your-token>
  ```
  The `GET /` health check remains publicly accessible regardless.

**Configuration:**
```bash
# Local (Linux/Mac)
export API_TOKEN=your-secret-token

# Local (Windows CMD)
set API_TOKEN=your-secret-token

# Docker: copy and edit docker/.env.example → docker/.env
API_TOKEN=your-secret-token
```

**Error response when token is missing or invalid (HTTP 403):**
```json
{
  "detail": "Invalid or missing API token"
}
```

### Environment Variables

The service behaviour can be tuned via environment variables (set in `docker/.env` or exported before running):

| Variable | Default | Description |
|----------|---------|-------------|
| `API_TOKEN` | *(empty)* | Bearer token for authentication. Leave empty to disable. |
| `MAX_FILE_SIZE` | `10485760` | Maximum upload size in bytes (10 MB). |
| `MAX_CHUNK_TEXT_SIZE` | `100000` | Maximum characters allowed per chunk text field in JSON input. Chunks exceeding this are rejected with HTTP 400. |
| `MAX_WORKERS` | auto | Maximum parallel worker processes for percentile search (default: CPU count − 1, max 4). |
| `WORKER_TIMEOUT` | `300` | Per-worker timeout in seconds. |
| `CACHE_TIMEOUT` | `3600` | Seconds before an unused model is evicted from the in-memory cache. |
| `EMBEDDER_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Default embedding model loaded at startup. |

Copy `docker/.env.example` to `docker/.env` and uncomment the variables you want to override.

### Input Constraints

- **Encoding**: text files (`.txt`, `.md`) must be **UTF-8 encoded**. Non-UTF-8 files are rejected with HTTP 400.
- **Minimum length**: text files must contain **at least 2 sentences**. Single-sentence documents return HTTP 400.
- **File size**: uploads are limited to `MAX_FILE_SIZE` (default 10 MB).
- **JSON chunks**: each `text` field in a JSON input file must not exceed `MAX_CHUNK_TEXT_SIZE` characters (default 100 000). The total JSON payload is also limited to `MAX_FILE_SIZE`.

### Example API Call using cURL

```bash
# Basic usage with required parameters
curl -X POST "http://localhost:8000/normalized_semantic_chunker/?max_tokens=512" \
  -F "file=@document.txt"

# With all parameters specified
curl -X POST "http://localhost:8000/normalized_semantic_chunker/?max_tokens=512&model=sentence-transformers/all-MiniLM-L6-v2&merge_small_chunks=true&merge_passes=3&verbosity=false" \
  -F "file=@document.txt" \
  -H "accept: application/json"

# With custom metadata to include in each chunk
curl -X POST 'http://localhost:8000/normalized_semantic_chunker/?max_tokens=512&chunk_metadata_json={"source_id":"doc123","category":"legal"}' \
  -F "file=@document.txt"

# With Bearer token authentication
curl -X POST "http://localhost:8000/normalized_semantic_chunker/?max_tokens=512" \
  -H "Authorization: Bearer your-secret-token" \
  -F "file=@document.txt"

# Health check endpoint (always public, no token needed)
curl http://localhost:8000/
```

### Example API Call using Python

```python
import requests
import json

# Replace with your actual API endpoint if hosted elsewhere
api_url = 'http://localhost:8000/normalized_semantic_chunker/'
file_path = 'document.txt' # Your input text file
max_tokens_per_chunk = 512
# model_name = "sentence-transformers/all-MiniLM-L6-v2" # Optional: specify a different model
merge_small_chunks = True  # Whether to merge undersized chunks with semantically similar neighbors
merge_passes = 3  # Number of merge passes (1-5, higher = fewer small chunks but slower)
verbosity = False  # Whether to show detailed logs
# Optional: add custom metadata to each chunk
chunk_metadata = {"source_id": "doc123", "category": "technical"}

try:
    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f, 'text/plain')}
        params = {
            'max_tokens': max_tokens_per_chunk,
            'merge_small_chunks': merge_small_chunks,
            'merge_passes': merge_passes,
            'verbosity': verbosity,
            # 'chunk_metadata_json': json.dumps(chunk_metadata)  # Uncomment to add metadata
        }
        # if model_name: # Uncomment to specify a model
        #     params['model'] = model_name

        response = requests.post(api_url, files=files, params=params)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        result = response.json()

        print(f"Successfully chunked document into {result['metadata']['n_chunks']} chunks.")
        # Save the response to a file
        output_file = 'response.json'
        # print("Metadata:", result['metadata'])
        # print("First chunk:", result['chunks'][0])
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(result, outfile, indent=4, ensure_ascii=False)
        print(f"Response saved to {output_file}")

except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except requests.exceptions.RequestException as e:
    print(f"API Request failed: {e}")
    if e.response is not None:
        print("Error details:", e.response.text)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
```


### Response Format

A successful chunking operation returns a `ChunkingResult` object:

```json
{
  "chunks": [
    {
      "text": "This is the first chunk of text...",
      "token_count": 480,
      "id": 1
    },
    {
      "text": "This is the second chunk...",
      "token_count": 505,
      "id": 2
    },
    {
      "text": "Additional chunks would appear here...",
      "token_count": 490,
      "id": 3
    }
  ],
  "metadata": {
    "n_chunks": 42,
    "avg_tokens": 495,
    "max_tokens": 510,
    "min_tokens": 150,
    "percentile": 85,
    "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
    "source": "your-document-source.txt",
    "processing_time": 15.78
  }
}
```

**With custom metadata** (when using `chunk_metadata_json` parameter):

```json
{
  "chunks": [
    {
      "text": "This is the first chunk of text...",
      "token_count": 480,
      "id": 1,
      "source_id": "doc123",
      "category": "legal"
    }
  ],
  "metadata": { ... }
}
```


## Contributing

The Normalized Semantic Chunker is an open-source project that thrives on community contributions. Your involvement is not just welcome, it's fundamental to the project's growth, innovation, and long-term success.

Whether you're fixing bugs, improving documentation, adding new features, or sharing ideas, every contribution helps build a better tool for everyone. We believe in the power of collaborative development and welcome participants of all skill levels.

If you're interested in contributing:

1. Fork the repository
2. Create a development environment with all dependencies
3. Make your changes
4. Add tests if applicable
5. Ensure all tests pass
6. Submit a pull request

Happy Semantic Chunking!

---
