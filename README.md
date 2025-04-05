![GPU Accelerated](https://img.shields.io/badge/GPU-Accelerated-green)
![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-blue)
![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)
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
-   **Docker Integration**: Easy deployment with Docker and docker-compose.

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
  - [Getting the Code](#getting-the-code)
  - [Local Installation with Uvicorn](#local-installation-with-uvicorn)
  - [Docker Deployment (Recommended)](#docker-deployment-recommended)
- [Using the API](#using-the-api)
  - [API Endpoints](#api-endpoints)
  - [Example API Call](#example-api-call)
  - [Response Format](#response-format)
- [Contributing](#contributing)

## How the Text Chunking Algorithm Works

### The Pipeline

The core innovation of Normalized Semantic Chunker lies in its multi-step pipeline that combines NLP techniques with statistical optimization to ensure both semantic coherence and size consistency:

1. The application exposes a simple REST API endpoint where users can upload a text document and parameters for maximum token limits and embedding model selection. 
2. The text is initially split into sentences using sophisticated regex pattern matching.
3. Each sentence is transformed into a vector embedding using state-of-the-art transformer models (default: `BAAI/bge-m3`).
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
| Adaptability | Requires manual parameter tuning | Automatically finds optimal parameters for each document |

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
- **Scalability**: Efficient handling of large documents.
- **Consistent Quality**: Predictable and reliable results regardless of text type.

### Flexibility and Customization

The algorithm adapts automatically to different types of content:

- **Adaptive Parameters**: Automatic identification of the best chunking parameters for each document.
- **Configurability**: Ability to specify custom maximum token limits (max_tokens).
- **Extensibility**: Modular architecture easily extendable with new features.
- **Embedding Model Selection**: Switch between different transformer models based on your needs.

## Installation and Deployment

### Prerequisites

- Docker and Docker Compose (for Docker deployment)
- NVIDIA GPU with CUDA support (recommended)
- NVIDIA Container Toolkit (for GPU passthrough in Docker)
- Python 3.10-3.12 (Python 3.11 recommended, Python 3.13 not supported due to dependency compatibility issues)

### Getting the Code

Before proceeding with any installation method, clone the repository:
```bash
git clone https://github.com/smart-models/Normalized-Semantic-Chunking.git
cd Normalized-Semantic-Chunking
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
   pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.1.1+cu121
   ```

3. Run the FastAPI server:
   ```bash
   uvicorn normalized_semantic_chunker:app --reload
   ```

4. The API will be available at `http://localhost:8000`.
   
   Access the API documentation and interactive testing interface at `http://localhost:8000/docs`.

### Docker Deployment (Recommended)

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

3. The API will be available at `http://localhost:8000`.
   
   Access the API documentation and interactive testing interface at `http://localhost:8000/docs`.

## Using the API

### API Endpoints

- **POST `/normalized_semantic_chunker/`**  
  Chunks a text document into semantically coherent segments while controlling token size.
  
  **Parameters:**
  - `file`: The text file to be chunked (supports .txt and .md formats)
  - `max_tokens`: Maximum token count per chunk (integer, required)
  - `model`: Embedding model to use for semantic analysis (string, default: `BAAI/bge-m3`)
  
  **Response:**
  Returns a JSON object containing:
  - `chunks`: Array of text segments with their token counts and IDs
  - `metadata`: Processing statistics including chunk count, token statistics, percentile used, model name, and processing time

- **GET `/`**  
  Health check endpoint that returns service status, GPU availability, and API version.

### Example API Call using cURL

```bash
# Basic usage with required parameters
curl -X POST "http://localhost:8000/normalized_semantic_chunker/?max_tokens=512" \
  -F "file=@document.txt" 

# With all parameters specified
curl -X POST "http://localhost:8000/normalized_semantic_chunker/?max_tokens=512&model=BAAI/bge-m3" \
  -F "file=@document.txt" \
  -H "accept: application/json"

# Health check endpoint
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
# model_name = "BAAI/bge-m3" # Optional: specify a different model

try:
    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f, 'text/plain')}
        params = {'max_tokens': max_tokens_per_chunk}
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
    "embedder_model": "BAAI/bge-m3",
    "processing_time": 15.78
  }
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
