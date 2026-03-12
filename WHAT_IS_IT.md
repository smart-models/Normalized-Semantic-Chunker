
![What is it?](what-is-it.jpg)

**1. Purpose of the Code**

Imagine you have a long text document, like an article or a book chapter. This code provides a web service (an API) that can intelligently break that long text down into smaller, meaningful pieces, called "chunks". The goal isn't just to chop the text randomly, but to create chunks that:

*   Are **semantically coherent**: Each chunk contains sentences that are related in meaning or topic.
*   Are **normalized in size**: Each chunk tries to stay below a maximum size limit you specify (measured in "tokens," which are like words or parts of words).

Essentially, it's a smart text splitter designed to keep related ideas together while respecting size constraints. This is often useful for preparing text to be processed by other AI models that might have limits on how much text they can handle at once.

**2. Input(s) it Takes**

To use this service, you need to send it a request containing:

*   **A Text File:** You must upload a file containing the text you want to chunk. The code allows files ending in `.txt`, `.md` (Markdown), or `.json` (JSON). Text files must be **UTF-8 encoded** (non-UTF-8 files are rejected) and must contain **at least 2 sentences**.
*   **Maximum Tokens:** You need to tell the code the maximum number of tokens (words/word pieces) allowed in each chunk. This is a required number and must be greater than zero.
*   **(Optional) Model Name:** The code uses a specific AI model (called an "embedding model") to understand the meaning of sentences. By default, it uses one called `"sentence-transformers/all-MiniLM-L6-v2"`, but you can optionally specify a different compatible model name if needed.
*   **(Optional) Merge Small Chunks:** You can specify whether to merge undersized chunks with semantically similar neighbors (default: `true`).
*   **(Optional) Merge Passes:** When merging small chunks is enabled, you can control how many passes the algorithm makes to merge undersized chunks. Higher values (1-5) result in fewer small chunks but increase processing time (default: `3`).
*   **(Optional) Verbosity:** You can specify whether to show detailed logs during processing (default: `false`).
*   **(Optional) Chunk Metadata JSON:** You can provide a JSON string containing custom metadata fields that will be merged into each output chunk. This is useful for adding source document IDs, categories, or any other fields you need to track (default: `null`).

**JSON Input Format:**
If you are using a JSON file as input, it must follow this structure:
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
    }
  ]
}
```
The service will process each text chunk individually, maintaining the chunk boundaries provided in your JSON file. Additional metadata fields beyond `text` are allowed and will be ignored during processing.

**3. Output(s) it Produces**

When the code successfully processes your file, it sends back a result containing:

*   **A List of Chunks:** This is the main output. It's a list where each item represents a chunk of text. For each chunk, you get:
    *   `text`: The actual text content of the chunk.
    *   `token_count`: How many tokens are in this specific chunk.
    *   `id`: A simple number (1, 2, 3...) to identify the chunk's order.
    *   Any **custom metadata fields** you provided via the `chunk_metadata_json` parameter (e.g., `source_id`, `category`, etc.).
*   **Metadata:** This provides summary information about the chunking process:
    *   `n_chunks`: The total number of chunks created.
    *   `avg_tokens`: The average number of tokens per chunk.
    *   `max_tokens`: The token count of the largest chunk created.
    *   `min_tokens`: The token count of the smallest chunk created.
    *   `percentile`: A number indicating how aggressively the code split the text based on meaning changes (explained more below).
    *   `embedder_model`: The name of the AI model used to understand sentence meaning.
    *   `source`: The name of the file that was uploaded.
    *   `processing_time`: How long the whole process took in seconds.

The code also includes a simple "health check" address (`/`) that confirms the service is running, whether it can use a GPU, whether the default embedding model is loaded and ready (`model_loaded`), and how many models are currently held in memory (`models_in_cache`).

**4. How it Achieves its Purpose (Logic and Algorithms)**

The code follows a multi-step process to create these smart chunks:

*   **Step 1: Sentence Splitting:** It first reads the text from your file and uses a clever pattern-matching technique (`split_into_sentences` function with Regular Expressions) to break the entire document down into individual sentences. It tries to be smart about handling things like abbreviations (Mr., Dr.), decimals, and quotes so it doesn't split sentences incorrectly.
*   **Step 2: Understanding Meaning (Embeddings):** For each sentence, it uses the specified AI model (`SentenceTransformer`) to generate a list of numbers called an "embedding" or "vector" (`get_embeddings` function). This vector represents the *meaning* of the sentence in a way the computer can understand. Sentences with similar meanings will have similar vectors. This step uses a GPU if available to speed it up. The models are downloaded and saved locally if not already present (`_get_model` function).
*   **Step 3: Measuring Similarity:** It then compares the meaning vectors of sentences that appear next to each other in the original text (`calculate_similarity` function). It calculates a "similarity score" (actually, a distance score - higher means *less* similar) between each adjacent pair of sentences. A high score suggests the topic might be changing between those two sentences.
*   **Step 4: Initial Semantic Chunking:** This is the core chunking step (`parallel_find_optimal_chunks` function). It looks at all the similarity scores calculated in the previous step. The idea is to group consecutive sentences together into chunks, *unless* the similarity score between two sentences is very high (meaning they are dissimilar).
    *   It tries to find the best "cutoff point" (a "percentile" of the similarity scores) to decide where to split. A higher percentile means it only splits chunks where there's a *very* strong change in meaning between sentences.
    *   Crucially, while doing this, it constantly checks if the chunks being formed are likely to stay *below* the `max_tokens` limit you provided (using a statistical estimate).
    *   It searches for the best percentile cutoff that balances meaningful splits and the size limit. It uses multiple CPU cores (`ProcessPoolExecutor`) to test different percentiles simultaneously to find the best one faster.
    *   This step produces an initial set of chunks, each with its text and token count.
*   **Step 5: Merging Small Chunks:** Sometimes, the initial chunking might create very small, possibly less useful chunks. This step (`merge_undersized_chunks`) identifies chunks that are significantly smaller than average (below the 5th percentile size). It then tries to merge such a small chunk with its adjacent neighbor (previous or next) *if* they are semantically similar (using the meaning vectors again) *and* the merged chunk doesn't exceed the `max_tokens` limit. This process runs in multiple passes (configurable via `merge_passes`) to maximize the number of small chunks that get merged.
*   **Step 6: Splitting Large Chunks:** After merging, some chunks might still be larger than the allowed `max_tokens`. This step (`split_oversized_chunk`) finds these oversized chunks and splits them further. It tries to split them along sentence boundaries first, aiming for reasonably sized sub-chunks. If a single sentence itself is too large, it will split that sentence based purely on the token limit.
*   **Step 7: Formatting Output:** Finally, it takes the finalized list of chunks (after merging and splitting), calculates the summary metadata, assigns IDs to the chunks, merges any custom metadata provided via `chunk_metadata_json`, and formats everything into the `ChunkingResult` structure to be sent back to the user.

**5. Important Logic Flows or Data Transformations**

*   **Text to Sentences:** The raw text string is transformed into a list of sentence strings.
*   **Sentences to Embeddings:** Each sentence string is transformed into a numerical vector representing its meaning.
*   **Embeddings to Similarity Scores:** Pairs of adjacent sentence vectors are transformed into a single number indicating their semantic distance.
*   **Sentences + Scores to Chunks:** The list of sentences and the list of similarity scores are combined, using the percentile logic and token limits, to form an initial list of text chunks (each with text and token count).
*   **Chunk Refinement:** The initial list of chunks undergoes two transformations: merging based on size and similarity, and splitting based on size limits.
*   **Model Caching:** The AI models needed for embeddings can be large. The code uses a cache (`_model_cache` and `_get_model`) to keep a loaded model in memory (RAM) so it doesn't have to be reloaded from disk or re-downloaded every time. It also saves downloaded models locally in a `models` directory.
*   **Parallel Processing:** The search for the optimal percentile cutoff (`parallel_find_optimal_chunks`) is computationally intensive. The code uses parallel processing to speed this up by distributing the work across multiple CPU cores.
*   **Error Handling & Logging:** The code includes `try...except` blocks to catch errors during processing and uses a `logging` system to record information about its steps and any errors encountered (saving errors to a file in a `logs` directory). This helps in debugging if something goes wrong.

In summary, this code defines a sophisticated API that takes text, understands its meaning sentence by sentence, and then groups related sentences into chunks that respect a user-defined size limit, performing adjustments to handle chunks that are too small or too large.