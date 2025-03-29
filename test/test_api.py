import pytest
import sys
import os
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from normalized_semantic_chunker import app

# Directory for test data files
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")


@pytest.fixture(scope="session", autouse=True)
def setup_test_data():
    """Create test data directory and ensure test file exists."""
    # Create test data directory if it doesn't exist
    os.makedirs(TEST_DATA_DIR, exist_ok=True)

    # Path to the test file
    alice_path = os.path.join(TEST_DATA_DIR, "alice_in_wonderland.txt")

    # Check if test file exists, if not raise an error
    if not os.path.exists(alice_path):
        raise FileNotFoundError(
            f"Required test file not found: {alice_path}. Please ensure the Alice in Wonderland text file exists in the {TEST_DATA_DIR} directory."
        )

    yield  # Run the tests


@pytest.fixture
def client():
    """Create a test client with actual embedder."""
    with TestClient(app) as test_client:
        yield test_client


def test_alice_file_processing(client):
    """Test processing alice_in_wonderland.txt and validate response structure."""
    # Path to the test file
    alice_path = os.path.join(TEST_DATA_DIR, "alice_in_wonderland.txt")

    # Open the file for sending to the API
    with open(alice_path, "rb") as f:
        # Send request to the API
        response = client.post(
            "/normalized_semantic_chunker/",
            files={"file": ("alice_in_wonderland.txt", f, "text/plain")},
            params={"max_tokens": 800},  # Pass as query parameter
        )

    # Check response status code
    assert response.status_code == 200, f"API returned error: {response.text}"

    # Parse the response
    data = response.json()

    # Validate the top-level structure matches expected schema
    assert "chunks" in data, "Response missing 'chunks' key"
    assert "metadata" in data, "Response missing 'metadata' key"

    # Validate chunks structure
    chunks = data["chunks"]
    assert isinstance(chunks, list), "'chunks' should be a list"
    assert len(chunks) > 0, "No chunks were generated"

    # Validate structure of first chunk
    first_chunk = chunks[0]
    assert "text" in first_chunk, "Chunk missing 'text' field"
    assert "token_count" in first_chunk, "Chunk missing 'token_count' field"
    assert "id" in first_chunk, "Chunk missing 'id' field"

    # Validate metadata structure
    metadata = data["metadata"]
    assert "n_chunks" in metadata, "Metadata missing 'n_chunks' field"
    assert "avg_tokens" in metadata, "Metadata missing 'avg_tokens' field"
    assert "max_tokens" in metadata, "Metadata missing 'max_tokens' field"
    assert "min_tokens" in metadata, "Metadata missing 'min_tokens' field"
    assert "percentile" in metadata, "Metadata missing 'percentile' field"
    assert "embedder_model" in metadata, "Metadata missing 'embedder_model' field"
    assert "processing_time" in metadata, "Metadata missing 'processing_time' field"

    # Types validation
    assert isinstance(metadata["n_chunks"], int)
    assert isinstance(metadata["avg_tokens"], int)
    assert isinstance(metadata["max_tokens"], int)
    assert isinstance(metadata["min_tokens"], int)
    assert isinstance(metadata["percentile"], int)
    assert isinstance(metadata["embedder_model"], str)
    assert isinstance(metadata["processing_time"], (int, float))

    # Validate values are reasonable
    assert metadata["n_chunks"] > 0, "Should have at least one chunk"
    assert metadata["n_chunks"] == len(chunks), (
        "n_chunks should match length of chunks list"
    )
    assert metadata["max_tokens"] > 0, "max_tokens should be positive"
    assert metadata["min_tokens"] > 0, "min_tokens should be positive"
    assert metadata["max_tokens"] >= metadata["min_tokens"], (
        "max_tokens should be >= min_tokens"
    )

    # Validate chunk content and ids
    for i, chunk in enumerate(chunks):
        assert isinstance(chunk["text"], str), f"Chunk {i} text should be string"
        assert isinstance(chunk["token_count"], int), (
            f"Chunk {i} token_count should be int"
        )
        assert isinstance(chunk["id"], int), f"Chunk {i} id should be int"
        assert len(chunk["text"]) > 0, f"Chunk {i} text should not be empty"
        assert chunk["token_count"] > 0, f"Chunk {i} token_count should be positive"
        assert chunk["id"] > 0, f"Chunk {i} id should be positive"

        # Validate chunk IDs are sequential starting from 1
        assert chunk["id"] == i + 1, (
            f"Chunk IDs should be sequential starting from 1, got {chunk['id']} at position {i}"
        )

    print(f"Successfully validated response with {metadata['n_chunks']} chunks.")
    print(f"Average tokens per chunk: {metadata['avg_tokens']}")
    print(f"Chunk token range: {metadata['min_tokens']} - {metadata['max_tokens']}")
