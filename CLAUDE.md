# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Run the server locally
```bash
uvicorn normalized_semantic_chunker:app --reload
```

### Install dependencies
```bash
pip install -r requirements.txt
```

For GPU support (CUDA 12.6):
```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu126 torch==2.6.0+cu126
```

### Run tests
```bash
pytest test/test_api.py
```

Run a single test:
```bash
pytest test/test_api.py::test_alice_file_processing
pytest test/test_api.py::TestMergePasses::test_merge_passes_parameter_accepted
```

### Docker (from the `docker/` directory)
```bash
# CPU
docker compose --profile cpu up -d

# GPU
docker compose --profile gpu up -d

# Stop (must match the profile used to start)
docker compose --profile cpu down
```

## Architecture

The entire application lives in a single file: `normalized_semantic_chunker.py`. It is a FastAPI app that exposes two endpoints:
- `POST /normalized_semantic_chunker/` — the main chunking endpoint
- `GET /` — health check (always public, no auth)

### Chunking pipeline (all in `normalized_semantic_chunker.py`)

1. **Text ingestion**: Accepts `.txt`, `.md`, or `.json` files. JSON must follow `{"chunks": [{"text": "..."}]}` format — each `text` field is processed independently.
2. **Sentence splitting**: Regex-based sentence tokenization.
3. **Embedding**: Uses `SentenceTransformer` (default: `sentence-transformers/all-MiniLM-L6-v2`). Models are cached in-memory with a timeout (`CACHE_TIMEOUT`, default 1 hour). Models are stored on disk in `models/`.
4. **Percentile search**: Parallel (`ProcessPoolExecutor`) search across similarity percentiles to find the optimal split threshold. The target is that the estimated 95th percentile of chunk token counts stays ≤ `max_tokens` (using `mean + 1.645 * std_dev`).
5. **Post-processing**:
   - Small chunk merging: undersized chunks are merged with their most semantically similar neighbor (up to `merge_passes` times, range 1–5).
   - Oversized chunk splitting: any chunk still over `max_tokens` is split at sentence boundaries.
6. **Response**: Returns `ChunkingResult` (Pydantic model) with `chunks` array and `metadata`.

### Resource management constants (top of file)
Key tunables: `MIN_SENTENCES_FOR_PARALLEL`, `WORKERS_VERY_LARGE_DOC`, `WORKERS_LARGE_DOC`, `STEP_SIZE_*` — these control adaptive behavior based on document size.

### Authentication
Controlled by the `API_TOKEN` environment variable. When set, all POST requests require `Authorization: Bearer <token>`. Token comparison uses `hmac.compare_digest` to prevent timing attacks. See `docker/.env.example` for Docker configuration.

### Environment variables
| Variable | Default | Description |
|---|---|---|
| `EMBEDDER_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Default embedding model |
| `MAX_FILE_SIZE` | `10485760` (10MB) | Max upload file size in bytes |
| `MAX_WORKERS` | `min(cpu_count-1, 4)` | Max parallel workers |
| `CACHE_TIMEOUT` | `3600` | Model cache TTL in seconds |
| `API_TOKEN` | `""` (disabled) | Bearer token for auth |

### Tests
Tests use `fastapi.testclient.TestClient` against the real app (no mocks). The test suite requires `test/test_data/alice_in_wonderland.txt` to exist. The `alice_in_wonderland.json` file in the same directory is a pre-chunked JSON version used for JSON input format testing.
