import os
import hmac
import time
import json
import logging
import re
import psutil
import tiktoken
import torch
import asyncio
import multiprocessing
import numpy as np
from logging.handlers import RotatingFileHandler
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Dict, Union, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, ConfigDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from contextlib import asynccontextmanager


RESERVED_SYSTEM_MEMORY_GB = 4.0  # GB of RAM to keep free for the OS and other processes
BASE_MEMORY_PER_WORKER_GB = 1.0  # Base GB of RAM allocated per worker
MEMORY_PER_SENTENCE_GB = 0.0004  # Additional GB of RAM per sentence per worker
DOC_SIZE_FACTOR_SCALER = (
    3000  # Scaler used in document size factor for CPU-based worker calculation
)
VERY_LARGE_DOC_SENTENCE_THRESHOLD = (
    20000  # Sentences to be considered a very large document
)
LARGE_DOC_SENTENCE_THRESHOLD = 10000  # Sentences to be considered a large document
MIN_SENTENCES_FOR_PARALLEL = (
    100  # Minimum sentences to use parallel processing instead of sequential
)
WORKERS_VERY_LARGE_DOC = 1  # Max workers for very large documents
WORKERS_LARGE_DOC = 2  # Max workers for large documents
STEP_SIZE_VERY_LARGE_DOC_THRESHOLD = (
    15000  # Sentence count threshold for using smallest step size
)
STEP_SIZE_LARGE_DOC_THRESHOLD = (
    5000  # Sentence count threshold for using medium step size
)
STEP_SIZE_DEFAULT = 10  # Default step size for smaller documents
STEP_SIZE_LARGE_DOC = 5  # Step size for large documents
STEP_SIZE_VERY_LARGE_DOC = 3  # Step size for very large documents

ALLOWED_EXTENSIONS = {"txt", "md", "json"}
EMBEDDER_MODEL = os.environ.get(
    "EMBEDDER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
MAX_FILE_SIZE = int(os.environ.get("MAX_FILE_SIZE", 10 * 1024 * 1024))  # 10MB default
MAX_WORKERS = max(
    1, int(os.environ.get("MAX_WORKERS", min(multiprocessing.cpu_count() - 1, 4)))
)
CACHE_TIMEOUT = int(os.environ.get("CACHE_TIMEOUT", 3600))  # 1 hour in seconds
WORKER_TIMEOUT = int(os.environ.get("WORKER_TIMEOUT", 300))  # 5 minutes default


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model_lock, _gpu_lock

    # Inizializza i lock nell'event loop attivo (non a livello di modulo)
    _model_lock = asyncio.Lock()
    _gpu_lock = asyncio.Lock()

    logger.info(
        f"Loading embedding model {EMBEDDER_MODEL} during application startup..."
    )
    try:
        await _get_model(EMBEDDER_MODEL)
        async with _gpu_lock:
            await _load_model_to_gpu(EMBEDDER_MODEL)
        logger.info("Embedding model loaded and ready on device.")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {str(e)}")
        raise RuntimeError(f"Cannot start: embedding model failed to load: {e}") from e

    yield

    # Cleanup
    logger.info("Application shutting down, cleaning up resources...")
    try:
        # Cleanup model cache
        async with _model_lock:
            for model_name in list(_model_cache.keys()):
                if model_name in _model_cache:
                    del _model_cache[model_name]
                    logger.info(f"Removed model {model_name} from cache")

        # Cleanup GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory cleared during shutdown")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")


app = FastAPI(
    title="Normalized Semantic Chunker",
    description="API for processing and chunking text documents into smaller, semantically coherent segments",
    version="1.0.0",
    lifespan=lifespan,
)

API_TOKEN = os.environ.get("API_TOKEN", "").strip()
_security = HTTPBearer(auto_error=False)


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(_security),
):
    """Verify Bearer token if API_TOKEN is configured. No-op when API_TOKEN is unset."""
    if not API_TOKEN:
        return  # Auth disabled
    if credentials is None or not hmac.compare_digest(credentials.credentials, API_TOKEN):
        raise HTTPException(status_code=403, detail="Invalid or missing API token")

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Configure logging (this sets up root logger with a console handler)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Get the logger
logger = logging.getLogger(__name__)

# Create a file handler for error logs
error_log_path = logs_dir / "errors.log"
file_handler = RotatingFileHandler(
    error_log_path,
    maxBytes=10485760,  # 10 MB
    backupCount=5,  # Keep 5 backup logs
    encoding="utf-8",
)

# Set the file handler to only log errors and critical messages
file_handler.setLevel(logging.ERROR)

# Create a formatter
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s - %(pathname)s:%(lineno)d"
)
file_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(file_handler)


# Create a singleton for model caching with expiration
_model_cache: dict = {}
_model_last_used: dict = {}
_model_lock: Optional[asyncio.Lock] = None        # inizializzato nel lifespan
_gpu_lock: Optional[asyncio.Lock] = None          # inizializzato nel lifespan — un solo modello in VRAM alla volta
_current_gpu_model: Optional[str] = None          # nome del modello attualmente in VRAM


async def _get_model(model_name: str) -> SentenceTransformer:
    """Get model from cache or load it into RAM with cache expiration.

    Args:
        model_name (str): Name or path of the model to use.

    Returns:
        SentenceTransformer: The loaded model instance.
    """
    global _model_cache, _model_last_used

    current_time = time.time()

    async with _model_lock:
        # Check for expired models first (solo RAM, non tocca VRAM — gestita da _evict_gpu_model)
        expired_models = [
            name
            for name, last_used in _model_last_used.items()
            if current_time - last_used > CACHE_TIMEOUT
        ]

        # Remove expired models from RAM
        for name in expired_models:
            if name in _model_cache and name != model_name:
                logger.info(f"Removing expired model {name} from cache")
                del _model_cache[name]
                del _model_last_used[name]

        # Update or load the requested model
        if model_name not in _model_cache:
            # Create models directory if it doesn't exist
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)

            # Local path for the model
            local_model_path = models_dir / model_name.replace("/", "_")

            try:
                if local_model_path.exists():
                    # Load from local storage
                    logger.info(f"Loading model from local storage: {local_model_path}")
                    _model_cache[model_name] = SentenceTransformer(
                        str(local_model_path)
                    )
                else:
                    # Download and save model
                    logger.info(
                        f"Downloading model {model_name} and saving to {local_model_path}"
                    )
                    _model_cache[model_name] = SentenceTransformer(model_name)
                    _model_cache[model_name].save(str(local_model_path))
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {str(e)}")
                raise

        # Update last used timestamp
        _model_last_used[model_name] = current_time

        return _model_cache[model_name]


async def _evict_gpu_model() -> None:
    """Rimuove il modello corrente dalla VRAM e lo mantiene in RAM.

    DEVE essere chiamata mentre si detiene _gpu_lock.
    """
    global _current_gpu_model

    if _current_gpu_model is None or not torch.cuda.is_available():
        return

    model = _model_cache.get(_current_gpu_model)
    if model is not None:
        _model_cache[_current_gpu_model] = model.cpu()  # assegnazione corretta — evita il leak
        logger.info(f"Evicted model '{_current_gpu_model}' from VRAM, kept in RAM")

    torch.cuda.empty_cache()
    _current_gpu_model = None


async def _load_model_to_gpu(model_name: str) -> SentenceTransformer:
    """Carica il modello richiesto in VRAM, evictando quello precedente se necessario.

    DEVE essere chiamata mentre si detiene _gpu_lock.
    Il modello deve essere già in _model_cache (caricato da _get_model).
    """
    global _current_gpu_model

    if not torch.cuda.is_available():
        # Nessuna GPU: ritorna il modello in RAM così com'è
        return _model_cache[model_name]

    if _current_gpu_model == model_name:
        # Modello già in VRAM — nessun overhead
        return _model_cache[model_name]

    # Evict modello precedente dalla VRAM
    if _current_gpu_model is not None:
        await _evict_gpu_model()

    # Carica il nuovo modello in VRAM
    device = torch.device("cuda")
    _model_cache[model_name] = _model_cache[model_name].to(device)
    _current_gpu_model = model_name
    logger.info(f"Loaded model '{model_name}' to VRAM")

    return _model_cache[model_name]


def split_into_sentences(doc: str) -> List[str]:
    """Split a document into sentences using regex pattern matching.

    Args:
        doc (str): The input document text to be split into sentences.

    Returns:
        List[str]: A list of sentences, with each sentence stripped of leading/trailing whitespace.

    Note:
        The function handles common edge cases like:
        - Titles (Mr., Mrs., Dr., etc.)
        - Common abbreviations (i.e., e.g., etc.)
        - Decimal numbers
        - Ellipsis
        - Quotes and brackets
    """
    # Define a pattern that looks for sentence boundaries but doesn't include them in the split
    # Instead of splitting directly at punctuation, we'll look for patterns that indicate sentence endings
    pattern = r"""
        # Match sentence ending punctuation followed by space and capital letter
        # Negative lookbehind for common titles and abbreviations
        (?<![A-Z][a-z]\.)                                                  # Not an abbreviation like U.S.
        (?<!Mr\.)(?<!Mrs\.)(?<!Dr\.)(?<!Prof\.)(?<!Sr\.)(?<!Jr\.)(?<!Ms\.) # Not a title
        (?<!i\.e\.)(?<!e\.g\.)(?<!vs\.)(?<!etc\.)                          # Not a common abbreviation
        (?<!\d\.)(?<!\.\d)                                                 # Not a decimal or numbered list
        (?<!\.\.\.)                                                        # Not an ellipsis
        [\.!\?]                                                            # Sentence ending punctuation
        \s+                                                                # One or more whitespace
        (?=[A-Z])                                                          # Followed by capital letter
    """

    # Find all positions where we should split
    split_positions = []
    for match in re.finditer(pattern, doc, re.VERBOSE):
        # Split after the punctuation and space
        split_positions.append(match.end())

    # Use the positions to extract sentences
    sentences = []
    start = 0
    for pos in split_positions:
        if pos > start:
            sentences.append(doc[start:pos].strip())
            start = pos

    # Add the last sentence if there's remaining text
    if start < len(doc):
        sentences.append(doc[start:].strip())

    # Filter out empty sentences
    return [s for s in sentences if s]


async def get_embeddings(
    doc: List[str],
    model: str = EMBEDDER_MODEL,
    batch_size: int = 8,
    verbosity: bool = False,
    convert_to_numpy: bool = True,
    normalize_embeddings: bool = True,
) -> dict[str, List[float]]:
    """Generate embeddings for a list of text strings using a Sentence Transformer model.

    Args:
        doc (List[str]): List of text strings to generate embeddings for.
        model (str, optional): Name or path of the model to use.
        batch_size (int, optional): Batch size for embedding generation.
        verbosity (bool, optional): If True, shows all log messages and progress bars.
        convert_to_numpy (bool, optional): Whether to convert output to numpy array.
        normalize_embeddings (bool, optional): Whether to normalize embeddings.

    Returns:
        dict[str, List[float]]: Dictionary mapping input strings to their embeddings.

    Raises:
        HTTPException: If there's an error during the embedding process.
    """
    try:
        # Carica modello in RAM se non presente
        await _get_model(model)

        # Aggiusta batch size per documenti grandi
        effective_batch_size = batch_size
        if len(doc) > 1000:
            effective_batch_size = max(1, batch_size // 2)
            if verbosity:
                logger.info(
                    f"Large document detected, reducing batch size to {effective_batch_size}"
                )

        # Garantisce che il modello corretto sia in VRAM (serializza lo switch tra modelli)
        async with _gpu_lock:
            active_model = await _load_model_to_gpu(model)

            # Esegui encode in un thread separato per non bloccare l'event loop
            embeddings = await asyncio.to_thread(
                active_model.encode,
                doc,
                batch_size=effective_batch_size,
                show_progress_bar=verbosity,
                convert_to_numpy=convert_to_numpy,
                normalize_embeddings=normalize_embeddings,
            )

        # Costruisce il dizionario fuori dal lock (operazione CPU pura)
        result = {
            sentence: embedding.tolist()
            for sentence, embedding in zip(doc, embeddings)
        }

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}",
        )


def calculate_similarity(
    embeddings_dict: dict[str, List[float]], sentences: List[str]
) -> List[float]:
    """Calculate similarity scores between consecutive vectors in sentence order.

    Args:
        embeddings_dict (dict[str, List[float]]): Dictionary mapping sentences to vectors
        sentences (List[str]): Sentences in original order

    Returns:
        List[float]: List of similarity scores between consecutive vector pairs
    """
    if len(sentences) <= 1:
        return []

    try:
        # Extract vectors in sentence order
        vectors = [embeddings_dict[sentence] for sentence in sentences]

        # Process in batches for large documents to prevent OOM errors
        batch_size = 5000  # Adjust based on memory constraints
        if len(vectors) > batch_size:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            similarities = []

            for i in range(0, len(vectors) - 1, batch_size):
                end_idx = min(i + batch_size, len(vectors) - 1)
                batch_vectors1 = torch.tensor(
                    vectors[i:end_idx], dtype=torch.float32, device=device
                )
                batch_vectors2 = torch.tensor(
                    vectors[i + 1 : end_idx + 1], dtype=torch.float32, device=device
                )

                # Calculate norms
                norms1 = torch.linalg.norm(batch_vectors1, dim=1)
                norms2 = torch.linalg.norm(batch_vectors2, dim=1)

                # Calculate dot products
                dot_products = torch.sum(batch_vectors1 * batch_vectors2, dim=1)

                # Calculate similarities
                batch_similarities = 1 - (dot_products / (norms1 * norms2))
                similarities.extend(
                    [round(float(sim), 5) for sim in batch_similarities.cpu()]
                )

                # Clean up batch memory
                del (
                    batch_vectors1,
                    batch_vectors2,
                    norms1,
                    norms2,
                    dot_products,
                    batch_similarities,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            return similarities
        else:
            # For smaller documents, process all at once
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            vectors_tensor = torch.tensor(vectors, dtype=torch.float32, device=device)

            # Get consecutive pairs
            vectors1 = vectors_tensor[:-1]
            vectors2 = vectors_tensor[1:]

            # Calculate norms
            norms = torch.linalg.norm(vectors_tensor, dim=1)
            norms1 = norms[:-1]
            norms2 = norms[1:]

            # Calculate dot products
            dot_products = torch.sum(vectors1 * vectors2, dim=1)

            # Calculate similarities (using 1 - cosine similarity for angular distance)
            cosine_similarities = dot_products / (norms1 * norms2)
            angular_similarities = 1 - cosine_similarities

            # Move to CPU and round
            result = [round(float(sim), 5) for sim in angular_similarities.cpu()]

            # Cleanup GPU memory
            if torch.cuda.is_available():
                del vectors_tensor, vectors1, vectors2, norms, norms1, norms2
                del dot_products, cosine_similarities, angular_similarities
                torch.cuda.empty_cache()

            return result

    except Exception as e:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.error(f"Error calculating similarities: {str(e)}")
        raise HTTPException(status_code=500, detail="Error calculating similarities")


def _count_tokens_for_text(args: tuple[str, str]) -> int:
    """Count tokens in a text string using the specified encoding.

    Args:
        args (tuple): Tuple containing:
            - text (str): Text to count tokens for
            - encoding_name (str): Name of the tiktoken encoding to use

    Returns:
        int: Number of tokens in the text

    Raises:
        Exception: If tiktoken fails. Caller (ProcessPoolExecutor) handles the exception.
    """
    text, encoding_name = args
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def _group_chunks_by_similarity(
    sentences: List[str], distance: List[float], percentile: int
) -> tuple[dict[str, int], int, float, float]:
    """Group sentences into chunks based on similarity distances and a percentile threshold.

    Args:
        sentences (List[str]): List of sentences to group into chunks.
        distance (List[float]): List of similarity distances between consecutive sentences.
        percentile (int): Percentile value to use as threshold for chunk boundaries.

    Returns:
        tuple[dict[str, int], int, float, float]: A tuple containing:
            - Dictionary mapping text chunks to their token counts
            - Maximum token count across all chunks
            - Average token count across all chunks
            - Standard deviation of token counts
    """
    try:
        breakpoint = np.percentile(distance, percentile)
        indices_above_th = [i for i, x in enumerate(distance) if x > breakpoint]

        chunks = []
        start_index = 0
        for index in indices_above_th:
            combined_text = " ".join(sentences[start_index : index + 1])
            chunks.append(combined_text)
            start_index = index + 1

        if start_index < len(sentences):
            chunks.append(" ".join(sentences[start_index:]))

        # Calculate token counts with multiprocessing for large documents
        if len(chunks) > 100:
            with ProcessPoolExecutor(
                max_workers=min(MAX_WORKERS, len(chunks) // 10 + 1)
            ) as executor:
                token_args = [(chunk_text, "cl100k_base") for chunk_text in chunks]
                token_counts = list(executor.map(_count_tokens_for_text, token_args))
        else:
            # For smaller sets, avoid the overhead of multiprocessing
            token_counts = [
                _count_tokens_for_text((chunk_text, "cl100k_base"))
                for chunk_text in chunks
            ]

        # Create dictionary mapping chunks to token counts
        chunks_with_tokens = {
            chunk: count for chunk, count in zip(chunks, token_counts)
        }

        max_tokens = max(token_counts) if token_counts else 0
        average_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
        std_dev = np.std(token_counts) if len(token_counts) > 1 else 0

        return chunks_with_tokens, max_tokens, average_tokens, std_dev

    except Exception as e:
        logger.error(f"Error in _group_chunks_by_similarity: {str(e)}")
        # Return safe fallback in case of error
        if not sentences:
            return {}, 0, 0, 0

        # Single chunk fallback
        combined_text = " ".join(sentences)
        token_count = _count_tokens_for_text((combined_text, "cl100k_base"))
        return {combined_text: token_count}, token_count, token_count, 0


def _process_percentile_range(
    args: tuple[List[str], List[float], int, int],
) -> tuple[Optional[dict[str, int]], Optional[int], Optional[float]]:
    """Process a single percentile and check if it produces valid chunks.

    Args:
        args (tuple): Tuple containing:
            - sentences (List[str]): List of sentences to process
            - distance (List[float]): List of similarity distances
            - max_tokens (int): Maximum tokens allowed per chunk
            - percentile (int): Percentile to check

    Returns:
        tuple[Optional[dict[str, int]], Optional[int], Optional[float]]:
            - Dictionary mapping chunks to token counts (if valid)
            - Percentile used (if valid)
            - Average tokens (if valid)
            - Or (None, None, None) if invalid
    """
    try:
        sentences, distance, max_tokens, percentile = args
        chunks_with_tokens, threshold_tokens, average_tokens, std_dev = (
            _group_chunks_by_similarity(sentences, distance, percentile)
        )

        # Calculate 95th percentile value using z-score of 1.645
        estimated_95th_percentile = average_tokens + (1.645 * std_dev)

        if estimated_95th_percentile <= max_tokens:
            return chunks_with_tokens, percentile, average_tokens

        return None, None, None

    except Exception as e:
        logger.error(f"Error processing percentile {percentile}: {str(e)}")
        return None, None, None


def _find_optimal_chunks(
    sentences: List[str], distance: List[float], max_tokens: int
) -> tuple[dict[str, int], int, float]:
    """Find optimal chunk groupings that fit within a token limit using statistical approach.

    Args:
        sentences (List[str]): List of sentences to group into chunks.
        distance (List[float]): List of similarity distances between consecutive sentences.
        max_tokens (int): Maximum number of tokens allowed per chunk.

    Returns:
        tuple[dict[str, int], int, float]: A tuple containing:
            - Dictionary mapping text chunks to their token counts
            - Percentile value used for grouping (0 if no suitable grouping found)
            - Average token count across all chunks

    Note:
        Uses a statistical approach by calculating the estimated 95th percentile
        of token counts to ensure most chunks stay below the token limit.
    """
    # Use a targeted percentile range for more efficient exploration
    percentile_steps = 5
    for percentile in range(95, 0, -percentile_steps):
        chunks_with_tokens, max_token_val, average_tokens, std_dev = (
            _group_chunks_by_similarity(sentences, distance, percentile)
        )

        # Calculate 95th percentile value using z-score of 1.645
        estimated_95th_percentile = average_tokens + (1.645 * std_dev)

        if estimated_95th_percentile <= max_tokens:
            logger.info(
                f"Sequential fallback found valid percentile {percentile} with 95th percentile estimate {estimated_95th_percentile:.2f} <= {max_tokens}"
            )
            return chunks_with_tokens, percentile, average_tokens

    # If no valid percentile found, return the entire text as a single chunk
    logger.info(
        "No valid chunking found using sequential approach - returning single chunk"
    )
    fallback_chunks = {
        " ".join(sentences): _count_tokens_for_text(
            (" ".join(sentences), "cl100k_base")
        )
    }
    return fallback_chunks, 0, 0


def parallel_find_optimal_chunks(
    sentences: List[str],
    distance: List[float],
    max_tokens: int,
    start_percentile: int = 99,
    verbosity: bool = True,
) -> tuple[dict[str, int], int, float]:
    """Find optimal chunks using parallel batch processing starting from highest percentiles.

    Uses available CPU cores to process multiple percentiles simultaneously, collecting all
    valid results and selecting the highest percentile that satisfies the constraints.

    Args:
        sentences: List of sentences to group
        distance: List of similarity distances
        max_tokens: Maximum tokens per chunk
        start_percentile: Starting percentile for search (default: 99)

    Returns:
        tuple[dict[str, int], int, float]: Dictionary mapping chunks to token counts,
        percentile used, and average tokens
    """
    # Get available memory in GB and estimate memory needs based on document size
    available_memory_gb = psutil.virtual_memory().available / (1024**3)

    if verbosity:
        logger.info(f"Available system memory: {available_memory_gb:.2f} GB")

    # Estimate memory needed per worker (rough heuristic based on sentence count)
    estimated_mem_per_worker_gb = max(
        BASE_MEMORY_PER_WORKER_GB, len(sentences) * MEMORY_PER_SENTENCE_GB
    )
    if verbosity:
        logger.info(
            f"Estimated memory per worker: {estimated_mem_per_worker_gb:.2f} GB"
        )

    # Calculate max workers based on available memory (leave RESERVED_SYSTEM_MEMORY_GB for system)
    # Ensure estimated_mem_per_worker_gb is not zero to prevent DivisionByZeroError
    if estimated_mem_per_worker_gb <= 0:
        logger.warning(
            "Estimated memory per worker is zero or negative, defaulting memory_based_max_workers to 1."
        )
        memory_based_max_workers = 1
    else:
        memory_based_max_workers = max(
            1,
            int(
                (available_memory_gb - RESERVED_SYSTEM_MEMORY_GB)
                / estimated_mem_per_worker_gb
            ),
        )
    if verbosity:
        logger.info(f"Memory-based worker limit: {memory_based_max_workers}")

    # Also consider document size for worker scaling
    doc_size_factor = min(1.0, DOC_SIZE_FACTOR_SCALER / max(1, len(sentences)))
    cpu_based_max_workers = max(
        1, min(MAX_WORKERS, int(multiprocessing.cpu_count() * doc_size_factor))
    )
    if verbosity:
        logger.info(f"CPU-based worker limit: {cpu_based_max_workers}")

    # Take the minimum of memory-based and CPU-based worker counts
    max_workers = min(memory_based_max_workers, cpu_based_max_workers)

    # Hard cap based on sentence count for very large documents as extra safety
    if len(sentences) > VERY_LARGE_DOC_SENTENCE_THRESHOLD:
        max_workers = min(max_workers, WORKERS_VERY_LARGE_DOC)
        if verbosity:
            logger.info(
                f"Very large document (>{VERY_LARGE_DOC_SENTENCE_THRESHOLD} sentences), capping at {WORKERS_VERY_LARGE_DOC} worker(s)"
            )
    elif len(sentences) > LARGE_DOC_SENTENCE_THRESHOLD:
        max_workers = min(max_workers, WORKERS_LARGE_DOC)
        if verbosity:
            logger.info(
                f"Large document (>{LARGE_DOC_SENTENCE_THRESHOLD} sentences), capping at {WORKERS_LARGE_DOC} worker(s)"
            )

    # For very small documents, skip parallel processing
    if len(sentences) < MIN_SENTENCES_FOR_PARALLEL:
        if verbosity:
            logger.info(
                f"Small document (<{MIN_SENTENCES_FOR_PARALLEL} sentences), using sequential processing"
            )
        return _find_optimal_chunks(sentences, distance, max_tokens)

    if verbosity:
        logger.info(f"Using {max_workers} workers for parallel processing")

    try:
        # Scale step size inversely with document size
        if len(sentences) > STEP_SIZE_VERY_LARGE_DOC_THRESHOLD:
            initial_step_size = STEP_SIZE_VERY_LARGE_DOC
        elif len(sentences) > STEP_SIZE_LARGE_DOC_THRESHOLD:
            initial_step_size = STEP_SIZE_LARGE_DOC
        else:
            initial_step_size = STEP_SIZE_DEFAULT

        if verbosity:
            logger.info(f"Using initial step size of {initial_step_size}")

        for batch_start in range(start_percentile, 0, -initial_step_size * max_workers):
            batch_end = max(0, batch_start - initial_step_size * max_workers)
            current_percentiles = range(batch_start, batch_end, -initial_step_size)

            process_args = [
                (sentences, distance, max_tokens, p) for p in current_percentiles
            ]

            valid_results = []

            # Use a timeout for worker processes to prevent hung processes
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_process_percentile_range, args)
                    for args in process_args
                ]

                try:
                    for future in as_completed(futures, timeout=WORKER_TIMEOUT):
                        try:
                            chunks_with_tokens, percentile, average_tokens = future.result()
                            if chunks_with_tokens is not None:
                                valid_results.append(
                                    (chunks_with_tokens, percentile, average_tokens)
                                )
                        except Exception as e:
                            logger.error(f"Error processing future: {str(e)}")
                            continue
                except TimeoutError:
                    logger.error(
                        f"Percentile search timed out after {WORKER_TIMEOUT}s. "
                        f"Proceeding with results collected so far ({len(valid_results)} valid)."
                    )

            if valid_results:
                # We found at least one valid percentile, now refine with finer grain
                valid_results.sort(key=lambda x: x[1], reverse=True)
                best_valid_percentile = valid_results[0][1]

                # Perform a refined search around the best valid percentile
                refined_start = min(99, best_valid_percentile + initial_step_size)
                refined_end = max(1, best_valid_percentile - initial_step_size)
                refined_percentiles = range(refined_start, refined_end, -1)

                refined_args = [
                    (sentences, distance, max_tokens, p) for p in refined_percentiles
                ]
                refined_results = []

                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(_process_percentile_range, args)
                        for args in refined_args
                    ]

                    try:
                        for future in as_completed(futures, timeout=WORKER_TIMEOUT):
                            try:
                                chunks_with_tokens, percentile, average_tokens = (
                                    future.result()
                                )
                                if chunks_with_tokens is not None:
                                    refined_results.append(
                                        (chunks_with_tokens, percentile, average_tokens)
                                    )
                            except Exception as e:
                                logger.error(
                                    f"Error processing future in refinement: {str(e)}"
                                )
                                continue
                    except TimeoutError:
                        logger.error(
                            f"Refined percentile search timed out after {WORKER_TIMEOUT}s. "
                            f"Proceeding with results collected so far ({len(refined_results)} valid)."
                        )

                # Combine results and select the best one
                all_results = valid_results + refined_results
                all_results.sort(key=lambda x: x[1], reverse=True)
                best_chunks_with_tokens, best_percentile, best_average_tokens = (
                    all_results[0]
                )

                if verbosity:
                    logger.info(
                        f"Selected the highest valid percentile: {best_percentile}"
                    )
                return best_chunks_with_tokens, best_percentile, best_average_tokens

        if verbosity:
            logger.info(
                "No valid chunking found in parallel processing, falling back to sequential"
            )
        return _find_optimal_chunks(sentences, distance, max_tokens)

    except Exception as e:
        logger.error(
            f"Error in parallel processing: {str(e)}, falling back to sequential approach"
        )
        return _find_optimal_chunks(sentences, distance, max_tokens)


async def merge_undersized_chunks(
    chunks: List[dict],
    min_token_threshold: float,
    max_tokens: int,
    model: str = EMBEDDER_MODEL,
    verbosity: bool = False,
    max_passes: int = 3,
) -> List[dict]:
    """
    Merge chunks that are below a minimum token threshold with semantically similar neighbors.
    Uses multiple passes to maximize the number of small chunks that get merged.

    Args:
        chunks (List[dict]): List of chunks with 'text' and 'token_count' keys
        min_token_threshold (float): Minimum token threshold (e.g., 5th percentile).
            This threshold remains FIXED across all passes to ensure convergence.
        max_tokens (int): Maximum allowed tokens for a chunk
        model (str): Embedding model to use
        verbosity (bool): If True, shows all log messages and progress bars
        max_passes (int): Maximum number of merge passes (default: 3, range: 1-5)

    Returns:
        List[dict]: Updated list of chunks after merging small ones
    """
    # Track initial state for final logging
    initial_undersized = sum(
        1 for chunk in chunks if chunk["token_count"] < min_token_threshold
    )

    if initial_undersized == 0:
        return chunks  # No small chunks to merge

    # If most chunks are undersized, adjust the threshold (only once, before passes)
    adjusted_threshold = min_token_threshold
    if initial_undersized > len(chunks) * 0.5:
        logger.info("Too many undersized chunks, adjusting threshold to 80%")
        adjusted_threshold = min_token_threshold * 0.8
        initial_undersized = sum(
            1 for chunk in chunks if chunk["token_count"] < adjusted_threshold
        )

    # Working copy of chunks that will be modified across passes
    current_chunks = list(chunks)
    total_merged = 0

    # Multi-pass merge loop
    for pass_num in range(1, max_passes + 1):
        # Identify undersized chunks for this pass (using fixed threshold)
        undersized_indices = [
            i
            for i, chunk in enumerate(current_chunks)
            if chunk["token_count"] < adjusted_threshold
        ]

        if not undersized_indices:
            if verbosity:
                logger.info(
                    f"Pass {pass_num}: No undersized chunks remaining, stopping early"
                )
            break

        # Sort by token count (ascending) to process smallest first
        undersized_indices.sort(key=lambda i: current_chunks[i]["token_count"])

        # Calculate embeddings for all current chunks
        chunk_texts = [chunk["text"] for chunk in current_chunks]
        embeddings_dict = await get_embeddings(
            doc=chunk_texts,
            model=model,
            batch_size=8,
            verbosity=False,  # Suppress per-pass embedding logs
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embeddings = np.array([embeddings_dict[text] for text in chunk_texts])

        # Track merges for this pass
        merged_indices = set()
        merged_count = 0

        for idx in undersized_indices:
            if idx in merged_indices:
                continue

            current_chunk = current_chunks[idx]
            if current_chunk is None:
                continue

            candidates = []

            # Check previous chunk
            if idx > 0 and idx - 1 not in merged_indices:
                prev_chunk = current_chunks[idx - 1]
                if prev_chunk is not None:
                    combined_tokens = (
                        current_chunk["token_count"] + prev_chunk["token_count"]
                    )
                    if combined_tokens <= max_tokens:
                        similarity = float(np.dot(embeddings[idx], embeddings[idx - 1]))
                        candidates.append((idx - 1, similarity, combined_tokens))

            # Check next chunk
            if idx < len(current_chunks) - 1 and idx + 1 not in merged_indices:
                next_chunk = current_chunks[idx + 1]
                if next_chunk is not None:
                    combined_tokens = (
                        current_chunk["token_count"] + next_chunk["token_count"]
                    )
                    if combined_tokens <= max_tokens:
                        similarity = float(np.dot(embeddings[idx], embeddings[idx + 1]))
                        candidates.append((idx + 1, similarity, combined_tokens))

            if not candidates:
                continue

            # Select best candidate by similarity
            best_candidate = max(candidates, key=lambda x: x[1])
            merge_idx, similarity, combined_tokens = best_candidate

            # Merge maintaining document order
            if merge_idx < idx:
                merged_text = (
                    current_chunks[merge_idx]["text"] + " " + current_chunk["text"]
                )
                target_idx = merge_idx
                removed_idx = idx
            else:
                merged_text = (
                    current_chunk["text"] + " " + current_chunks[merge_idx]["text"]
                )
                target_idx = idx
                removed_idx = merge_idx

            current_chunks[target_idx] = {
                "text": merged_text,
                "token_count": combined_tokens,
            }
            current_chunks[removed_idx] = None
            merged_indices.add(removed_idx)
            merged_count += 1

        # Compact the list (remove None values) for next pass
        current_chunks = [chunk for chunk in current_chunks if chunk is not None]
        total_merged += merged_count

        if verbosity:
            remaining = sum(
                1
                for chunk in current_chunks
                if chunk["token_count"] < adjusted_threshold
            )
            logger.info(
                f"Pass {pass_num}/{max_passes}: {len(undersized_indices)} undersized, "
                f"{merged_count} merged, {remaining} remaining"
            )

        # Stop if no merges were possible in this pass
        if merged_count == 0:
            if verbosity:
                logger.info(f"Pass {pass_num}: No merges possible, stopping early")
            break

    # Final statistics
    final_undersized = sum(
        1 for chunk in current_chunks if chunk["token_count"] < adjusted_threshold
    )

    if verbosity:
        logger.info(
            f"Merge completed: {len(chunks)} → {len(current_chunks)} chunks "
            f"({total_merged} merged), undersized: {initial_undersized} → {final_undersized}"
        )

    return current_chunks


def _split_large_sentence(
    sentence: str, max_tokens: int
) -> List[Dict[str, Union[str, int]]]:
    """Split an oversized sentence at token boundaries to fit within token limits.

    Args:
        sentence: The sentence text to split
        max_tokens: Maximum number of tokens per chunk

    Returns:
        List of dictionaries containing text and token_count

    Raises:
        Exception: If tiktoken fails to encode/decode the sentence.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    all_tokens = encoding.encode(sentence)

    chunks = []
    for i in range(0, len(all_tokens), max_tokens):
        chunk_tokens = all_tokens[i : min(i + max_tokens, len(all_tokens))]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append({"text": chunk_text, "token_count": len(chunk_tokens)})

    return chunks


def split_oversized_chunk(
    chunk_text: str, max_tokens: int
) -> List[Dict[str, Union[str, int]]]:
    """Split an oversized chunk using a sentence-based approach with semantic prioritization.

    Args:
        chunk_text: The text to be split into chunks
        max_tokens: Maximum number of tokens allowed per chunk

    Returns:
        List of dictionaries containing text and token_count
    """
    try:
        # Split text into sentences
        sentences = split_into_sentences(chunk_text)

        # Calculate token count for each sentence
        sentence_tokens = [
            (s, _count_tokens_for_text((s, "cl100k_base"))) for s in sentences
        ]

        # Create chunks respecting sentence boundaries
        chunks = []
        current_chunk = []
        current_tokens = 0

        # Optimal target is 70-80% of max_tokens for better balance
        target_size = int(max_tokens * 0.75)

        for sentence, tokens in sentence_tokens:
            # If the sentence is already too large (rare), split at token level
            if tokens > max_tokens:
                chunks.extend(_split_large_sentence(sentence, max_tokens))
                continue

            # If adding this sentence would exceed max_tokens, finalize current chunk
            if current_tokens + tokens > max_tokens:
                chunks.append(
                    {"text": " ".join(current_chunk), "token_count": current_tokens}
                )
                current_chunk = []
                current_tokens = 0

            # If we've reached optimal size and are at a "natural" sentence boundary
            elif (
                current_tokens >= target_size
                and sentence[-1] in ".!?"
                and len(current_chunk) > 0
            ):
                chunks.append(
                    {"text": " ".join(current_chunk), "token_count": current_tokens}
                )
                current_chunk = []
                current_tokens = 0

            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_tokens += tokens

        # Add final chunk if there's anything remaining
        if current_chunk:
            chunks.append(
                {"text": " ".join(current_chunk), "token_count": current_tokens}
            )

        return chunks
    except Exception as e:
        logger.error(f"Error splitting oversized chunk: {str(e)}")
        # Fallback to simpler splitting method
        return _split_large_sentence(chunk_text, max_tokens)


def validate_file_extension(filename: str) -> None:
    """Validate that the file has an allowed extension.

    Args:
        filename: Name of the file to validate

    Raises:
        HTTPException: If the file extension is not allowed
    """
    if not filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    extension = filename.split(".")[-1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File extension '.{extension}' is not allowed. Only {', '.join(ALLOWED_EXTENSIONS)} files are accepted",
        )


def parse_json_content(content: bytes) -> List[str]:
    """Parse JSON content and extract text chunks.

    Args:
        content: Raw bytes from the uploaded file

    Returns:
        List of text strings from the JSON chunks

    Raises:
        ValueError: With specific messages for different validation failures
    """
    try:
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("JSON root must be an object")

        if "chunks" not in data:
            raise ValueError("JSON must contain a 'chunks' key")

        if not isinstance(data["chunks"], list):
            raise ValueError("JSON 'chunks' must be an array")

        # Protect against excessive chunks
        MAX_CHUNKS = 10000  # Set a reasonable limit
        if len(data["chunks"]) > MAX_CHUNKS:
            raise ValueError(f"JSON contains too many chunks (max: {MAX_CHUNKS})")

        chunks = []
        for i, item in enumerate(data["chunks"], 1):
            if not isinstance(item, dict):
                raise ValueError(f"Chunk {i} must be an object")

            if "text" not in item:
                raise ValueError(f"Chunk {i} is missing required 'text' field")

            if not isinstance(item["text"], str):
                raise ValueError(f"Chunk {i} 'text' must be a string")

            # Skip empty strings after stripping
            text = item["text"].strip()
            if text:
                chunks.append(text)

        if not chunks:
            raise ValueError("No valid text chunks found in JSON")

        # Detailed logging is handled by the caller based on verbosity setting
        return chunks

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {str(e)}")
        raise ValueError(f"Invalid JSON format: {str(e)}")
    except Exception as e:
        logger.error(f"Error parsing JSON: {str(e)}")
        raise ValueError(f"Error parsing JSON: {str(e)}")


def parse_chunk_metadata(metadata_json: Optional[str]) -> Optional[Any]:
    """Parse and validate the chunk_metadata_json parameter."""
    if metadata_json is None:
        return None

    if not metadata_json.strip():
        return None

    try:
        return json.loads(metadata_json)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid chunk_metadata_json: {str(e)}")
        raise HTTPException(
            status_code=400, detail=f"Invalid chunk_metadata_json: {str(e)}"
        )


def validate_file_size(content_length: int) -> None:
    """Validate that the file size is within allowed limits.

    Args:
        content_length: Length of the file content in bytes

    Raises:
        HTTPException: If the file size exceeds the maximum allowed
    """
    if content_length > MAX_FILE_SIZE:
        max_size_mb = MAX_FILE_SIZE / (1024 * 1024)
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds the maximum allowed size of {max_size_mb:.1f} MB",
        )


class ChunkingInput(BaseModel):
    max_tokens: int = Field(..., gt=0, description="Maximum number of tokens per chunk")
    model: str = Field(
        default=EMBEDDER_MODEL,
        description=f"Embedding model to use. Default is {EMBEDDER_MODEL}",
        json_schema_extra={"example": EMBEDDER_MODEL},
    )
    merge_small_chunks: bool = Field(
        default=True,
        description="Whether to merge undersized chunks for more balanced output",
    )
    merge_passes: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum number of merge passes for undersized chunks (higher = fewer small chunks, but slower). Only used when merge_small_chunks is True.",
    )
    verbosity: bool = Field(
        default=False,
        description="If True, shows all log messages. If False, only shows step headers and final statistics.",
    )
    chunk_metadata_json: Optional[str] = Field(
        default=None,
        description="Optional JSON string to merge into each output chunk.",
    )


class ChunkingMetadata(BaseModel):
    n_chunks: int
    avg_tokens: float
    max_tokens: int
    min_tokens: int
    percentile: int
    embedder_model: str
    source: str
    processing_time: float


class Chunk(BaseModel):
    model_config = ConfigDict(extra="allow")
    text: str
    token_count: int
    id: int


class ChunkingResult(BaseModel):
    chunks: List[Chunk]
    metadata: ChunkingMetadata


@app.get("/")
async def health_check():
    """Check the health status of the API service.

    Returns:
        dict: A dictionary containing:
            - status: Current health status of the service
            - gpu_available: Boolean indicating if GPU is available
            - version: Current API version
    """
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "version": app.version,
        "default_model": EMBEDDER_MODEL,
    }


@app.post("/normalized_semantic_chunker/", response_model=ChunkingResult)
async def Normalized_Semantic_Chunker(
    file: UploadFile = File(...),
    input_data: ChunkingInput = Depends(),
    background_tasks: BackgroundTasks = None,
    _auth=Depends(verify_token),
):
    """
    Process and chunk a text document into smaller, semantically coherent segments using advanced NLP techniques.
    The function performs the following steps:
    1. Validates the input file format and size
    2. Processes the content based on file type
    3. Splits text into sentences using regex pattern matching
    4. Generates embeddings for semantic analysis
    5. Groups sentences into chunks based on semantic similarity
    6. Handles edge cases like oversized chunks and merges small chunks when needed

    Args:
        file (UploadFile): The uploaded file to process. Supported formats:
            - Text files (.txt, .md): Plain text content that will be split into sentences
            - JSON files (.json): Expected format: {"chunks": [{"text": "..."}, ...]}
        input_data (ChunkingInput): Configuration parameters for the chunking process:
            - max_tokens (int): Maximum number of tokens allowed per chunk. Must be > 0.
            - model (str, optional): Name or path of the embedding model to use. Defaults to 'sentence-transformers/all-MiniLM-L6-v2'.
            - merge_small_chunks (bool, optional): If True, merges undersized chunks with semantically similar neighbors. Defaults to True.
            - show_progress_bar (bool, optional): If True, displays progress bars during long-running operations. Defaults to False.

    Returns:
        ChunkingResult: Object containing the chunking results with the following structure:
            - chunks (List[Chunk]): List of processed chunks, each containing:
                - text (str): The chunk content
                - token_count (int): Number of tokens in the chunk
                - id (int): Unique identifier for the chunk
            - metadata (ChunkingMetadata): Processing statistics and metadata:
                - n_chunks (int): Total number of chunks created
                - avg_tokens (float): Average token count across all chunks
                - max_tokens (int): Maximum token count in any chunk
                - min_tokens (int): Minimum token count in any chunk
                - percentile (int): Optimal percentile used for semantic chunking
                - embedder_model (str): Name of the model used for embeddings
                - processing_time (float): Total processing time in seconds

    Raises:
        HTTPException: With appropriate status codes for various error conditions:
            - 400: Invalid file format, empty content, or invalid parameters
            - 413: File size exceeds maximum allowed limit
            - 500: Internal server error during processing
            - 422: Validation error for input parameters
    """
    start_time = time.time()

    try:
        parsed_metadata = parse_chunk_metadata(input_data.chunk_metadata_json)
        # Validate file extension
        validate_file_extension(file.filename)

        # Read and validate file size
        content = await file.read()
        validate_file_size(len(content))

        # Check if file is JSON
        if file.filename.lower().endswith(".json"):
            try:
                # Parse JSON content directly
                logger.info("Step 1 - JSON file processing")
                sentences = parse_json_content(content)
                if input_data.verbosity:
                    logger.info(f"Extracted {len(sentences)} text chunks from JSON")
            except ValueError as e:
                raise HTTPException(
                    status_code=400, detail=f"Invalid JSON format: {str(e)}"
                )
        else:
            # Handle text files (txt, md)
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                # Try with different encoding if UTF-8 fails
                text = content.decode("latin-1")

            # Step 1: Split the document into sentences
            logger.info("Step 1 - Text Splitting")
            try:
                if input_data.verbosity:
                    logger.info("Using regex-based text splitting")
                sentences = split_into_sentences(text)
                if input_data.verbosity:
                    logger.info(f"Number of sentences: {len(sentences)}")
            except ValueError as ve:
                # Handle validation errors (like None inputs or wrong types)
                logger.error(f"Validation error during sentence splitting: {str(ve)}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid input for sentence splitting: {str(ve)}",
                )
            except RuntimeError as re:
                # Handle runtime errors from the sentence splitting functions
                logger.error(f"Runtime error during sentence splitting: {str(re)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing document with sentence splitting: {str(re)}",
                )
            except Exception as e:
                # Catch any other unexpected errors
                logger.error(f"Unexpected error during sentence splitting: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Unexpected error during sentence splitting: {str(e)}",
                )

        if not sentences:
            raise HTTPException(
                status_code=400,
                detail="No valid sentences found in the document. Please check the content.",
            )

        # Step 2: Vector embedding
        logger.info("Step 2 - Vector Embedding")
        embeddings_dict = await get_embeddings(
            sentences,
            model=input_data.model,
            verbosity=input_data.verbosity,
        )

        # Step 3: Calculate sentence vector distances
        logger.info("Step 3 - Sentence Vector Distances Calculation")
        sentence_vector_distance = calculate_similarity(embeddings_dict, sentences)

        # Step 4: Semantic chunking - now returns dictionary with token counts
        logger.info("Step 4 - Semantic Chunking")
        chunks_with_tokens, percentile, average_tokens = parallel_find_optimal_chunks(
            sentences,
            sentence_vector_distance,
            input_data.max_tokens,
            verbosity=input_data.verbosity,
        )

        # Check if chunking produced valid results
        # Note: percentile=0 is a valid fallback (single chunk), so we check chunks instead
        if not chunks_with_tokens:
            logger.warning("No valid chunks produced")
            raise HTTPException(
                status_code=400,
                detail="No valid chunks produced. Try increasing max_tokens parameter.",
            )
        if percentile == 0:
            logger.warning("Using fallback single-chunk mode (percentile=0)")

        # Dict text->token_count
        final_chunks = []

        for chunk_text, token_count in chunks_with_tokens.items():
            final_chunks.append({"text": chunk_text, "token_count": token_count})

        # Extract token counts for statistical analysis
        token_counts = [chunk["token_count"] for chunk in final_chunks]
        total_chunks = len(token_counts)

        if not token_counts:
            raise HTTPException(
                status_code=500,
                detail="Chunking failed to produce any valid chunks. Please try with different parameters.",
            )

        # Calculate statistics
        mean_tokens = np.mean(token_counts)
        percentile_5th = np.percentile(token_counts, 5)
        percentile_95th = np.percentile(token_counts, 95)

        # Count chunks in each category
        chunks_under_5th = sum(1 for count in token_counts if count < percentile_5th)
        chunks_over_95th = sum(1 for count in token_counts if count > percentile_95th)
        chunks_in_range = total_chunks - chunks_under_5th - chunks_over_95th

        # STEP 5: Add to log
        logger.info("Step 5 - Distribution Statistics Calculation")

        # Detailed statistics shown only in verbose mode
        if input_data.verbosity:
            logger.info(f"Mean tokens: {mean_tokens:.2f}")
            logger.info(f"Total chunks: {total_chunks}")
            logger.info(f"5th percentile threshold: {percentile_5th:.2f} tokens")
            logger.info(f"95th percentile threshold: {percentile_95th:.2f} tokens")
            logger.info(
                f"Number of chunks under 5th percentile: {chunks_under_5th} ({chunks_under_5th / total_chunks * 100:.2f}%)"
            )
            logger.info(
                f"Number of chunks in 5-95th percentile: {chunks_in_range} ({chunks_in_range / total_chunks * 100:.2f}%)"
            )
            logger.info(
                f"Number of chunks over 95th percentile: {chunks_over_95th} ({chunks_over_95th / total_chunks * 100:.2f}%)"
            )

        # STEP 6: Merge undersized chunks if enabled
        if input_data.merge_small_chunks and chunks_under_5th > 0:
            logger.info("Step 6 - Merge Undersized Chunks")
            if input_data.verbosity:
                logger.info(
                    f"Merge process for chunks under 5th percentile (threshold: {percentile_5th:.2f} tokens)"
                )
            final_chunks = await merge_undersized_chunks(
                chunks=final_chunks,
                min_token_threshold=percentile_5th,
                max_tokens=input_data.max_tokens // 2,
                model=input_data.model,
                verbosity=input_data.verbosity,
                max_passes=input_data.merge_passes,
            )

        # STEP 7: Split oversized chunks
        logger.info("Step 7 - Split Oversized Chunks")
        oversized_indices = [
            i
            for i, chunk in enumerate(final_chunks)
            if chunk["token_count"] > input_data.max_tokens
        ]

        if oversized_indices:
            if input_data.verbosity:
                logger.info(
                    f"Starting split process for oversized chunks (95th percentile: {percentile_95th:.2f} tokens)"
                )
                logger.info(
                    f"Found {len(oversized_indices)} chunks over threshold of {input_data.max_tokens} tokens"
                )

        normalized_chunks = []

        for i, chunk in enumerate(final_chunks):
            if i in oversized_indices:
                sub_chunks = split_oversized_chunk(chunk["text"], input_data.max_tokens)
                normalized_chunks.extend(sub_chunks)
            else:
                normalized_chunks.append(chunk)

        chunks_before_split = len(final_chunks)
        final_chunks = normalized_chunks

        token_counts_after_split = [chunk["token_count"] for chunk in final_chunks]

        if not token_counts_after_split:
            raise HTTPException(
                status_code=500,
                detail="Chunking failed after splitting process. Please try with different parameters.",
            )

        mean_tokens_after_split = np.mean(token_counts_after_split)
        max_tokens_after_split = max(token_counts_after_split)

        if input_data.verbosity:
            logger.info(
                f"After splitting: {len(final_chunks)} total chunks (increased from {chunks_before_split} to {len(final_chunks)}, +{len(final_chunks) - chunks_before_split})"
            )

        logger.info("Step 8 - Semantic Chunking Statistics after processing")

        # Final statistics always shown regardless of verbosity
        logger.info(f"Mean tokens: {mean_tokens_after_split:.2f}")
        logger.info(f"Total chunks: {len(final_chunks)}")
        logger.info(f"Smallest chunk: {min(token_counts_after_split)} tokens")
        logger.info(f"Largest chunk: {max_tokens_after_split} tokens")

        metadata_payload = None
        # Check if metadata was explicitly provided (even if JSON null)
        has_explicit_metadata = (
            input_data.chunk_metadata_json is not None
            and input_data.chunk_metadata_json.strip()
        )
        if has_explicit_metadata:
            if isinstance(parsed_metadata, dict):
                metadata_payload = {
                    key: value
                    for key, value in parsed_metadata.items()
                    if key not in {"text", "token_count", "id"}
                }
            else:
                # Wrap non-dict values (including None from JSON null) in chunk_metadata
                metadata_payload = {"chunk_metadata": parsed_metadata}

        chunk_items = []
        for i, chunk in enumerate(final_chunks):
            chunk_payload = {
                "text": chunk["text"],
                "token_count": chunk["token_count"],
                "id": i + 1,
            }
            if metadata_payload:
                chunk_payload.update(metadata_payload)
            chunk_items.append(Chunk(**chunk_payload))

        processing_time = time.time() - start_time
        result = ChunkingResult(
            chunks=chunk_items,
            metadata=ChunkingMetadata(
                n_chunks=len(final_chunks),
                avg_tokens=float(
                    sum(chunk["token_count"] for chunk in final_chunks)
                    / len(final_chunks)
                ),
                max_tokens=max(chunk["token_count"] for chunk in final_chunks),
                min_tokens=min(chunk["token_count"] for chunk in final_chunks),
                percentile=percentile,
                embedder_model=input_data.model,
                source=file.filename,  # Use filename as the source
                processing_time=processing_time,
            ),
        )

        # Schedule cleanup after response is sent
        if background_tasks:
            background_tasks.add_task(
                lambda: torch.cuda.empty_cache() if torch.cuda.is_available() else None
            )

        end_time = time.time()
        logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")
        return result

    except HTTPException:
        # Re-raise HTTP exceptions as is
        raise
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        logger.error(f"Error processing request: {error_type} - {error_msg}")

        # Clean GPU memory in case of error too
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Return a safe error message without exposing system details
        raise HTTPException(
            status_code=500,
            detail=f"Error processing the chunking request: {error_type}. Please check the logs for more information.",
        )
