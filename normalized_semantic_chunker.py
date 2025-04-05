import time
import logging
import numpy as np
import re
import tiktoken
import torch
import multiprocessing
from logging.handlers import RotatingFileHandler
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Dict, Union
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from pydantic import BaseModel, Field
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from contextlib import asynccontextmanager

ALLOWED_EXTENSIONS = {"txt", "md"}
EMBEDDER_MODEL = "BAAI/bge-m3"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading embedding model during application startup...")
    _get_model(EMBEDDER_MODEL)  # Assuming EMBEDDER_MODEL is defined
    logger.info("Embedding model loaded.")
    yield
    # Optional cleanup
    logger.info("Application shutting down.")
    if EMBEDDER_MODEL in _model_cache:  # Assuming _model_cache is defined
        del _model_cache[EMBEDDER_MODEL]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


app = FastAPI(
    title="Normalized Semantic Chunker",
    description="API for processing and chunking text documents into smaller, semantically coherent segments",
    version="0.5.0",
    lifespan=lifespan,
)

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Configure logging
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

# Create a singleton for model caching
_model_cache = {}


def _get_model(model_name: str) -> SentenceTransformer:
    """Get model from cache or load it into RAM.

    Args:
        model_name (str): Name or path of the model to use.

    Returns:
        SentenceTransformer: The loaded model instance.
    """
    if model_name not in _model_cache:
        # Create models directory if it doesn't exist
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        # Local path for the model
        local_model_path = models_dir / model_name.replace("/", "_")

        if local_model_path.exists():
            # Load from local storage
            logger.info(f"Loading model from local storage: {local_model_path}")
            _model_cache[model_name] = SentenceTransformer(str(local_model_path))
        else:
            # Download and save model
            logger.info(
                f"Downloading model {model_name} and saving to {local_model_path}"
            )
            _model_cache[model_name] = SentenceTransformer(model_name)
            _model_cache[model_name].save(str(local_model_path))

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


def get_embeddings(
    doc: List[str],
    model: str = EMBEDDER_MODEL,
    batch_size: int = 4,
    show_progress_bar: bool = True,
    convert_to_numpy: bool = True,
    normalize_embeddings: bool = True,
) -> dict[str, List[float]]:
    """Generate embeddings for a list of text strings using a Sentence Transformer model.

    Args:
        doc (List[str]): List of text strings to generate embeddings for.
        model (str, optional): Name or path of the model to use.
            Defaults to "BAAI/bge-m3".
        batch_size (int, optional): Batch size for embedding generation.
            Defaults to 8.
        show_progress_bar (bool, optional): Whether to show progress bar.
            Defaults to True.
        convert_to_numpy (bool, optional): Whether to convert output to numpy array.
            Defaults to True.
        normalize_embeddings (bool, optional): Whether to normalize embeddings.
            Defaults to True.

    Returns:
        dict[str, List[float]]: Dictionary mapping input strings to their embeddings.

    Raises:
        HTTPException: If there's an error during the embedding process.
    """
    try:
        # Get model from cache (loads from disk if not in RAM)
        model_instance = _get_model(model)

        # Move to GPU if available
        if torch.cuda.is_available():
            model_instance = model_instance.to("cuda")

        # Get embeddings
        embeddings = model_instance.encode(
            doc,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize_embeddings,
        )

        # Create dictionary mapping sentences to embeddings
        result = {
            sentence: embedding.tolist() for sentence, embedding in zip(doc, embeddings)
        }

        # Cleanup GPU memory but keep model in RAM
        if torch.cuda.is_available():
            model_instance.cpu()
            del embeddings
            torch.cuda.empty_cache()
            # logger.info("GPU memory cleared after embeddings generation")

        return result

    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating embeddings: {str(e)}",
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

        # Convert to tensor and move to GPU
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

        # Calculate similarities
        cosine_similarities = dot_products / (norms1 * norms2)
        angular_similarities = 1 - cosine_similarities

        # Move to CPU and round
        result = [round(float(sim), 5) for sim in angular_similarities.cpu()]

        # Cleanup GPU memory
        if torch.cuda.is_available():
            del vectors_tensor, vectors1, vectors2, norms, norms1, norms2
            del dot_products, cosine_similarities, angular_similarities
            torch.cuda.empty_cache()
            logger.info("GPU memory cleared after similarity calculation")

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

    # Calculate token counts once
    token_counts = [
        _count_tokens_for_text((chunk_text, "cl100k_base")) for chunk_text in chunks
    ]

    # Create dictionary mapping chunks to token counts
    chunks_with_tokens = {chunk: count for chunk, count in zip(chunks, token_counts)}

    max_tokens = max(token_counts) if token_counts else 0
    average_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
    std_dev = np.std(token_counts) if len(token_counts) > 1 else 0

    return chunks_with_tokens, max_tokens, average_tokens, std_dev


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

    for percentile in range(99, 0, -1):
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
    available_cores = multiprocessing.cpu_count()
    max_workers = max(1, min(available_cores - 1, int(available_cores * 0.75)))

    logger.info(f"Using {max_workers} workers for parallel processing")

    try:
        for batch_start in range(start_percentile, 0, -max_workers):
            batch_end = max(0, batch_start - max_workers)
            current_percentiles = range(batch_start, batch_end, -1)

            process_args = [
                (sentences, distance, max_tokens, p) for p in current_percentiles
            ]

            valid_results = []

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_process_percentile_range, args)
                    for args in process_args
                ]

                for future in as_completed(futures):
                    try:
                        chunks_with_tokens, percentile, average_tokens = future.result()
                        if chunks_with_tokens is not None:
                            valid_results.append(
                                (chunks_with_tokens, percentile, average_tokens)
                            )
                    except Exception as e:
                        logger.error(f"Error processing future: {str(e)}")
                        continue

            if valid_results:
                valid_results.sort(key=lambda x: x[1], reverse=True)
                best_chunks_with_tokens, best_percentile, best_average_tokens = (
                    valid_results[0]
                )
                logger.info(f"Selected the highest valid percentile: {best_percentile}")
                return best_chunks_with_tokens, best_percentile, best_average_tokens

        logger.info("No valid chunking found in any batch")
        fallback_chunks = {
            " ".join(sentences): _count_tokens_for_text(
                (" ".join(sentences), "cl100k_base")
            )
        }
        return fallback_chunks, 0, 0

    except Exception as e:
        logger.error(f"Error in parallel processing: {str(e)}")
        return _find_optimal_chunks(sentences, distance, max_tokens)


def merge_undersized_chunks(
    chunks: List[dict],
    min_token_threshold: float,
    max_tokens: int,
    model: str = EMBEDDER_MODEL,
) -> List[dict]:
    """
    Merge chunks that are below a minimum token threshold with semantically similar neighbors.

    Args:
        chunks (List[dict]): List of chunks with 'text' and 'token_count' keys
        min_token_threshold (float): Minimum token threshold (e.g., 5th percentile)
        max_tokens (int): Maximum allowed tokens for a chunk
        model (str): Embedding model to use

    Returns:
        List[dict]: Updated list of chunks after merging small ones
    """
    # Step 1: Identify undersized chunks
    undersized_indices = [
        i
        for i, chunk in enumerate(chunks)
        if chunk["token_count"] < min_token_threshold
    ]

    if not undersized_indices:
        return chunks  # No small chunks to merge

    total_undersized = len(undersized_indices)

    # Step 2: Sort undersized chunks by token count (ascending) to process smallest first
    undersized_indices.sort(key=lambda i: chunks[i]["token_count"])

    # Step 3: Calculate embeddings for all chunks at once for efficiency
    chunk_texts = [chunk["text"] for chunk in chunks]

    embeddings_dict = get_embeddings(
        doc=chunk_texts,
        model=model,
        batch_size=2,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Convert dictionary result to numpy array in the same order as chunk_texts
    embeddings = np.array([embeddings_dict[text] for text in chunk_texts])

    # Step 4: Create working copies for processing
    result_chunks = list(chunks)
    merged_indices = set()  # Track indices that have been merged

    # Step 5: Process each undersized chunk
    for idx in undersized_indices:
        if idx in merged_indices:
            continue  # Skip if this chunk has already been merged

        current_chunk = result_chunks[idx]
        candidates = []

        # Check previous chunk if available
        if idx > 0 and idx - 1 not in merged_indices:
            prev_chunk = result_chunks[idx - 1]
            combined_tokens = current_chunk["token_count"] + prev_chunk["token_count"]

            if combined_tokens <= max_tokens:
                # Calculate cosine similarity using dot product (vectors are already normalized)
                similarity = float(np.dot(embeddings[idx], embeddings[idx - 1]))
                candidates.append((idx - 1, similarity, combined_tokens))

        # Check next chunk if available
        if idx < len(result_chunks) - 1 and idx + 1 not in merged_indices:
            next_chunk = result_chunks[idx + 1]
            combined_tokens = current_chunk["token_count"] + next_chunk["token_count"]

            if combined_tokens <= max_tokens:
                # Calculate cosine similarity
                similarity = float(np.dot(embeddings[idx], embeddings[idx + 1]))
                candidates.append((idx + 1, similarity, combined_tokens))

        # If no valid candidates, continue to next chunk
        if not candidates:
            # logger.info(f"No suitable merge candidates found for chunk {idx}")
            continue

        # Find best candidate based on similarity
        best_candidate = max(candidates, key=lambda x: x[1])
        merge_idx, similarity, combined_tokens = best_candidate

        # logger.info(
        #     f"Merging chunk {idx} with chunk {merge_idx}, similarity: {similarity:.4f}"
        # )

        # Determine merge order (maintain document order)
        if merge_idx < idx:
            merged_text = result_chunks[merge_idx]["text"] + " " + current_chunk["text"]
            target_idx = merge_idx
            removed_idx = idx
        else:
            merged_text = current_chunk["text"] + " " + result_chunks[merge_idx]["text"]
            target_idx = idx
            removed_idx = merge_idx

        # Update chunk at target index
        result_chunks[target_idx] = {
            "text": merged_text,
            "token_count": combined_tokens,
        }

        # Mark the other chunk as None (to be filtered later)
        result_chunks[removed_idx] = None

        # Mark indices as merged
        merged_indices.add(removed_idx)

        # Step 6: Update embedding for the merged chunk
        new_embedding_dict = get_embeddings(
            doc=[merged_text],
            model=model,
            batch_size=2,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embeddings[target_idx] = new_embedding_dict[merged_text]

    # Step 7: Filter out None values (merged chunks)
    final_chunks = [chunk for chunk in result_chunks if chunk is not None]

    # Report merging statistics
    logger.info(
        f"After merging: {len(chunks)} → {len(final_chunks)} chunks ({len(chunks) - len(final_chunks)} merged), {total_undersized} originally undersized"
    )

    return final_chunks


def _split_large_sentence(
    sentence: str, max_tokens: int
) -> List[Dict[str, Union[str, int]]]:
    """Split an oversized sentence at token boundaries to fit within token limits.

    Args:
        sentence: The sentence text to split
        max_tokens: Maximum number of tokens per chunk

    Returns:
        List of dictionaries containing text and token_count
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
        chunks.append({"text": " ".join(current_chunk), "token_count": current_tokens})

    return chunks


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
            detail=f"File extension '.{extension}' is not allowed. Only .txt and .md files are accepted",
        )


class ChunkingInput(BaseModel):
    max_tokens: int = Field(..., gt=0, description="Maximum number of tokens per chunk")
    model: str = Field(
        default=EMBEDDER_MODEL,
        description=f"Embedding model to use. Default is {EMBEDDER_MODEL}",
        json_schema_extra={"example": EMBEDDER_MODEL},
    )


class ChunkingMetadata(BaseModel):
    n_chunks: int
    avg_tokens: int
    max_tokens: int
    min_tokens: int
    percentile: int
    embedder_model: str
    processing_time: float


class Chunk(BaseModel):
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
    }


@app.post("/normalized_semantic_chunker/", response_model=ChunkingResult)
async def Normalized_Semantic_Chunker(
    file: UploadFile = File(...), input_data: ChunkingInput = Depends()
):
    """
    Args:
        file (UploadFile): The uploaded text file to process (.txt or .md format).
        input_data (ChunkingInput): Input parameters including:
            - max_tokens (int): Maximum token count per chunk
            - model (str): Embedding model name to use for semantic analysis

    Returns:
        ChunkingResult: Object containing:
            - chunks (List[Chunk]): List of text chunks with their token counts
            - metadata (ChunkingMetadata): Processing statistics including:
                - n_chunks: Total number of chunks created
                - avg_tokens: Average token count per chunk
                - max_tokens: Maximum tokens in any chunk
                - min_tokens: Minimum tokens in any chunk
                - percentile: Optimal percentile used for chunking
                - embedder_model: Model used for embeddings
                - processing_time: Total processing time in seconds

    Raises:
        HTTPException: 400 if file format is invalid or processing fails

    Note:
        The chunking algorithm:
        1. Splits text into sentences using NLP-based sentence boundary detection
        2. Generates vector embeddings for each sentence using the specified model
        3. Calculates semantic similarity between adjacent sentences
        4. Groups similar sentences into coherent chunks while respecting token limits
        5. Optimizes chunk boundaries using parallel percentile analysis
    """
    start_time = time.time()

    try:
        # Validate file extension
        validate_file_extension(file.filename)

        # Read and process the file
        content = await file.read()
        text = content.decode("utf-8")

        # Step 1: Split the document into sentences
        logger.info("Step 1 - Sentence Splitting")
        sentences = split_into_sentences(text)
        logger.info(f"Number of sentences: {len(sentences)}")

        # Step 2: Vector embedding
        logger.info("Step 2 - Vector Embedding")
        embeddings_dict = get_embeddings(sentences, input_data.model)

        # Step 3: Calculate sentence vector distances
        logger.info("Step 3 - Sentence Vector Distances Calculation")
        sentence_vector_distance = calculate_similarity(embeddings_dict, sentences)

        # Step 4: Semantic chunking - now returns dictionary with token counts
        logger.info("Step 4 - Semantic Chunking")
        chunks_with_tokens, percentile, average_tokens = parallel_find_optimal_chunks(
            sentences, sentence_vector_distance, input_data.max_tokens
        )
        # Check if valid percentile was found, otherwise raise exception
        if not percentile:
            logger.warning("No valid percentile found for chunking")
            raise HTTPException(
                status_code=400,
                detail="No valid percentile found for chunking. Try increasing max_tokens parameter.",
            )

        # Dict text->token_count
        final_chunks = []

        for chunk_text, token_count in chunks_with_tokens.items():
            final_chunks.append({"text": chunk_text, "token_count": token_count})

        # Extract token counts for statistical analysis
        token_counts = [chunk["token_count"] for chunk in final_chunks]
        total_chunks = len(token_counts)

        # Calculate statistics
        mean_tokens = np.mean(token_counts)
        percentile_5th = np.percentile(token_counts, 5)
        percentile_95th = np.percentile(token_counts, 95)

        # Count chunks in each category
        chunks_under_5th = sum(1 for count in token_counts if count < percentile_5th)
        chunks_over_95th = sum(1 for count in token_counts if count > percentile_95th)
        chunks_in_range = total_chunks - chunks_under_5th - chunks_over_95th

        # Add to log
        logger.info("Step 5 - Distribution Statistics Calculation")
        logger.info(f"Mean tokens: {mean_tokens:.2f}")
        logger.info(f"5th percentile threshold: {percentile_5th:.2f} tokens")
        logger.info(f"95th percentile threshold: {percentile_95th:.2f} tokens")
        logger.info(f"Total chunks: {total_chunks}")
        logger.info(
            f"Number of chunks under 5th percentile: {chunks_under_5th} ({chunks_under_5th / total_chunks * 100:.2f}%)"
        )
        logger.info(
            f"Number of chunks in 5-95th percentile: {chunks_in_range} ({chunks_in_range / total_chunks * 100:.2f}%)"
        )
        logger.info(
            f"Number of chunks over 95th percentile: {chunks_over_95th} ({chunks_over_95th / total_chunks * 100:.2f}%)"
        )

        # STEP 6: Merge undersized chunks
        if chunks_under_5th > 0:
            logger.info("Step 6 - Merge Undersized Chunks")
            logger.info("Merge process for chunks under 5th percentile")
            final_chunks = merge_undersized_chunks(
                chunks=final_chunks,
                min_token_threshold=percentile_5th,
                max_tokens=input_data.max_tokens,
                model=input_data.model,
            )

        # STEP 7: Split oversized chunks
        logger.info("Step 7 - Split Oversized Chunks")
        oversized_indices = [
            i
            for i, chunk in enumerate(final_chunks)
            if chunk["token_count"] > input_data.max_tokens
        ]

        if oversized_indices:
            logger.info("Starting split process for oversized chunks")
            logger.info(
                f"Found {len(oversized_indices)} chunks over threshold of {input_data.max_tokens} tokens"
            )

        normalized_chunks = []

        for i, chunk in enumerate(final_chunks):
            if i in oversized_indices:
                # logger.info(f"Splitting chunk {i} with {chunk['token_count']} tokens")
                sub_chunks = split_oversized_chunk(chunk["text"], input_data.max_tokens)
                normalized_chunks.extend(sub_chunks)
                # logger.info(f"Split into {len(sub_chunks)} sub-chunks")
            else:
                normalized_chunks.append(chunk)

        chunks_before_split = len(final_chunks)
        final_chunks = normalized_chunks

        token_counts_after_split = [chunk["token_count"] for chunk in final_chunks]
        mean_tokens_after_split = np.mean(token_counts_after_split)
        max_tokens_after_split = max(token_counts_after_split)

        logger.info(
            f"After splitting: {len(final_chunks)} total chunks (increased from {chunks_before_split} to {len(final_chunks)}, +{len(final_chunks) - chunks_before_split})"
        )

        logger.info("Step 8 - Semantic Chunking Statistics after processing")

        logger.info(f"Mean tokens: {mean_tokens_after_split:.2f}")
        logger.info(f"Total chunks: {len(final_chunks)}")
        logger.info(f"Smallest chunk: {min(token_counts_after_split)} tokens")
        logger.info(f"Largest chunk: {max_tokens_after_split} tokens")
        logger.info(
            f"Number of chunks under 5th percentile: {len([c for c in final_chunks if c['token_count'] < percentile_5th])} ({len([c for c in final_chunks if c['token_count'] < percentile_5th]) / len(final_chunks) * 100:.2f}%)"
        )
        logger.info(
            f"Number of chunks in 5-95th percentile: {len([c for c in final_chunks if percentile_5th <= c['token_count'] <= percentile_95th])} ({len([c for c in final_chunks if percentile_5th <= c['token_count'] <= percentile_95th]) / len(final_chunks) * 100:.2f}%)"
        )
        logger.info(
            f"Number of chunks over 95th percentile: {len([c for c in final_chunks if c['token_count'] > percentile_95th])} ({len([c for c in final_chunks if c['token_count'] > percentile_95th]) / len(final_chunks) * 100:.2f}%)"
        )

        processing_time = time.time() - start_time
        result = ChunkingResult(
            chunks=[
                Chunk(text=chunk["text"], token_count=chunk["token_count"], id=i + 1)
                for i, chunk in enumerate(final_chunks)
            ],
            metadata=ChunkingMetadata(
                n_chunks=len(final_chunks),
                avg_tokens=int(
                    sum(chunk["token_count"] for chunk in final_chunks)
                    / len(final_chunks)
                ),
                max_tokens=max(chunk["token_count"] for chunk in final_chunks),
                min_tokens=min(chunk["token_count"] for chunk in final_chunks),
                percentile=percentile,
                embedder_model=input_data.model,
                processing_time=processing_time,
            ),
        )

        end_time = time.time()
        logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")
        return result
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
