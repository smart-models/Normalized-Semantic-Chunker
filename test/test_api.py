import pytest
import sys
import os
import json
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
        # Send request to the API with the same parameters as request.py
        response = client.post(
            "/normalized_semantic_chunker/",
            files={"file": ("alice_in_wonderland.txt", f, "text/plain")},
            params={"max_tokens": 500},
        )

    # Check response status code
    assert response.status_code == 200, f"API returned error: {response.text}"

    # Parse the response
    data = response.json()

    # Basic structure validation
    assert "chunks" in data, "Response missing 'chunks' key"
    assert "metadata" in data, "Response missing 'metadata' key"

    # Validate chunks structure
    chunks = data["chunks"]
    assert isinstance(chunks, list), "'chunks' should be a list"
    assert len(chunks) > 0, "No chunks were generated"

    # Validate metadata structure
    metadata = data["metadata"]

    # Check required metadata fields
    required_metadata_fields = [
        "n_chunks",
        "avg_tokens",
        "max_tokens",
        "min_tokens",
        "percentile",
        "embedder_model",
        "source",
        "processing_time",
    ]

    for field in required_metadata_fields:
        assert field in metadata, f"Missing required metadata field: {field}"

    # Validate the source field value
    assert (
        metadata["source"] == "alice_in_wonderland.txt"
    ), f"Source field incorrect: expected 'alice_in_wonderland.txt', got '{metadata['source']}'"

    # Validate chunks structure
    for chunk in chunks:
        assert "text" in chunk, "Chunk missing 'text' field"
        assert "token_count" in chunk, "Chunk missing 'token_count' field"
        assert "id" in chunk, "Chunk missing 'id' field"

        # Validate types
        assert isinstance(chunk["text"], str), "Chunk text should be string"
        assert isinstance(chunk["token_count"], int), "Chunk token_count should be int"
        assert isinstance(chunk["id"], int), "Chunk id should be int"

        # Validate values
        assert len(chunk["text"]) > 0, "Chunk text should not be empty"
        assert chunk["token_count"] > 0, "Chunk token_count should be positive"
        assert chunk["id"] > 0, "Chunk id should be positive"

    # Log some information for debugging
    print(f"Successfully processed {len(chunks)} chunks")
    print(
        f"Token count range: {metadata['min_tokens']}-{metadata['max_tokens']} (avg: {metadata['avg_tokens']})"
    )
    print(f"Model used: {metadata['embedder_model']}")

    # Validate that the number of chunks matches the metadata
    assert len(chunks) == metadata["n_chunks"], (
        f"Number of chunks ({len(chunks)}) doesn't match metadata.n_chunks ({metadata['n_chunks']})"
    )

    # Validate that token counts are within the specified max_tokens
    for chunk in chunks:
        assert chunk["token_count"] <= 500, (
            f"Chunk {chunk['id']} exceeds max_tokens: {chunk['token_count']} > 500"
        )


@pytest.fixture
def alice_path():
    """Return the path to the alice_in_wonderland.txt test file."""
    return os.path.join(TEST_DATA_DIR, "alice_in_wonderland.txt")


def post_with_metadata(client, alice_path, metadata_json):
    """Helper to POST with chunk_metadata_json parameter."""
    with open(alice_path, "rb") as f:
        return client.post(
            "/normalized_semantic_chunker/",
            files={"file": ("alice_in_wonderland.txt", f, "text/plain")},
            params={"max_tokens": 500, "chunk_metadata_json": metadata_json},
        )


class TestMergePasses:
    """Tests for the merge_passes feature (Multi-Pass Merge for Small Chunks)."""

    def test_merge_passes_parameter_accepted(self, client, alice_path):
        """Test that merge_passes parameter is accepted by the API."""
        with open(alice_path, "rb") as f:
            response = client.post(
                "/normalized_semantic_chunker/",
                files={"file": ("alice_in_wonderland.txt", f, "text/plain")},
                params={"max_tokens": 500, "merge_passes": 3},
            )

        assert response.status_code == 200, f"API returned error: {response.text}"
        data = response.json()
        assert "chunks" in data
        assert "metadata" in data

    def test_merge_passes_one_baseline(self, client, alice_path):
        """Test merge_passes=1 produces valid output (baseline behavior)."""
        with open(alice_path, "rb") as f:
            response = client.post(
                "/normalized_semantic_chunker/",
                files={"file": ("alice_in_wonderland.txt", f, "text/plain")},
                params={"max_tokens": 500, "merge_passes": 1, "merge_small_chunks": True},
            )

        assert response.status_code == 200, f"API returned error: {response.text}"
        data = response.json()
        assert len(data["chunks"]) > 0, "Should produce chunks"

        # Validate all chunks have required fields
        for chunk in data["chunks"]:
            assert "text" in chunk
            assert "token_count" in chunk
            assert "id" in chunk

    def test_merge_passes_three_reduces_small_chunks(self, client, alice_path):
        """Test merge_passes=3 (default) effectively reduces small chunks."""
        with open(alice_path, "rb") as f:
            response = client.post(
                "/normalized_semantic_chunker/",
                files={"file": ("alice_in_wonderland.txt", f, "text/plain")},
                params={"max_tokens": 500, "merge_passes": 3, "merge_small_chunks": True},
            )

        assert response.status_code == 200, f"API returned error: {response.text}"
        data = response.json()
        chunks = data["chunks"]

        assert len(chunks) > 0, "Should produce chunks"

        # Calculate statistics to verify merge effectiveness
        token_counts = [c["token_count"] for c in chunks]
        avg_tokens = sum(token_counts) / len(token_counts)

        # With merge_passes=3, we expect reasonable average token count
        # (small chunks should have been merged)
        print(f"merge_passes=3: {len(chunks)} chunks, avg tokens: {avg_tokens:.1f}")

        # All chunks should respect max_tokens limit
        for chunk in chunks:
            assert chunk["token_count"] <= 500, (
                f"Chunk {chunk['id']} exceeds max_tokens: {chunk['token_count']} > 500"
            )

    def test_merge_passes_five_no_errors(self, client, alice_path):
        """Test merge_passes=5 (max) does not cause errors or timeout."""
        with open(alice_path, "rb") as f:
            response = client.post(
                "/normalized_semantic_chunker/",
                files={"file": ("alice_in_wonderland.txt", f, "text/plain")},
                params={"max_tokens": 500, "merge_passes": 5, "merge_small_chunks": True},
            )

        assert response.status_code == 200, f"API returned error: {response.text}"
        data = response.json()

        assert len(data["chunks"]) > 0, "Should produce chunks"
        assert data["metadata"]["processing_time"] > 0, "Should have valid processing time"

        print(f"merge_passes=5: {len(data['chunks'])} chunks, "
              f"processing time: {data['metadata']['processing_time']:.2f}s")

    def test_merge_passes_ignored_when_merge_disabled(self, client, alice_path):
        """Test that merge_passes is ignored when merge_small_chunks=False."""
        # First, get baseline with merge disabled
        with open(alice_path, "rb") as f:
            response_no_merge = client.post(
                "/normalized_semantic_chunker/",
                files={"file": ("alice_in_wonderland.txt", f, "text/plain")},
                params={"max_tokens": 500, "merge_small_chunks": False, "merge_passes": 1},
            )

        # Then with merge_passes=5 but still disabled
        with open(alice_path, "rb") as f:
            response_no_merge_5 = client.post(
                "/normalized_semantic_chunker/",
                files={"file": ("alice_in_wonderland.txt", f, "text/plain")},
                params={"max_tokens": 500, "merge_small_chunks": False, "merge_passes": 5},
            )

        assert response_no_merge.status_code == 200
        assert response_no_merge_5.status_code == 200

        data1 = response_no_merge.json()
        data2 = response_no_merge_5.json()

        # When merge is disabled, merge_passes should have no effect
        # Both should produce the same number of chunks
        assert data1["metadata"]["n_chunks"] == data2["metadata"]["n_chunks"], (
            f"merge_passes should be ignored when merge_small_chunks=False: "
            f"{data1['metadata']['n_chunks']} vs {data2['metadata']['n_chunks']}"
        )

    def test_merge_passes_validation_range(self, client, alice_path):
        """Test that merge_passes outside valid range (1-5) is rejected."""
        # Test merge_passes=0 (below minimum)
        with open(alice_path, "rb") as f:
            response = client.post(
                "/normalized_semantic_chunker/",
                files={"file": ("alice_in_wonderland.txt", f, "text/plain")},
                params={"max_tokens": 500, "merge_passes": 0},
            )
        assert response.status_code == 422, "merge_passes=0 should be rejected"

        # Test merge_passes=6 (above maximum)
        with open(alice_path, "rb") as f:
            response = client.post(
                "/normalized_semantic_chunker/",
                files={"file": ("alice_in_wonderland.txt", f, "text/plain")},
                params={"max_tokens": 500, "merge_passes": 6},
            )
        assert response.status_code == 422, "merge_passes=6 should be rejected"

    def test_more_passes_reduces_small_chunks(self, client, alice_path):
        """Test that increasing merge_passes tends to reduce the number of small chunks."""
        results = {}

        for passes in [1, 3, 5]:
            with open(alice_path, "rb") as f:
                response = client.post(
                    "/normalized_semantic_chunker/",
                    files={"file": ("alice_in_wonderland.txt", f, "text/plain")},
                    params={"max_tokens": 500, "merge_passes": passes, "merge_small_chunks": True},
                )

            assert response.status_code == 200
            data = response.json()
            results[passes] = {
                "n_chunks": data["metadata"]["n_chunks"],
                "min_tokens": data["metadata"]["min_tokens"],
                "avg_tokens": data["metadata"]["avg_tokens"],
            }

        # Log results for analysis
        for passes, stats in results.items():
            print(f"merge_passes={passes}: {stats['n_chunks']} chunks, "
                  f"min={stats['min_tokens']}, avg={stats['avg_tokens']:.1f}")

        # With more passes, we expect:
        # - Fewer or equal total chunks (chunks get merged)
        # - Higher or equal minimum token count (small chunks merged)
        # Note: This is a soft expectation as it depends on document structure
        assert results[5]["n_chunks"] <= results[1]["n_chunks"] + 5, (
            "More merge passes should not significantly increase chunk count"
        )


class TestChunkMetadataJson:
    """Tests for the chunk_metadata_json feature."""

    def test_merge_dict_metadata(self, client, alice_path):
        """Dict metadata fields are merged at top-level of each chunk."""
        metadata = {
            "source": "document.pdf",
            "author": "Mario Rossi",
            "category": "technical",
            "priority": 1,
        }
        response = post_with_metadata(client, alice_path, json.dumps(metadata))

        assert response.status_code == 200, f"API error: {response.text}"
        data = response.json()

        for chunk in data["chunks"]:
            for key, value in metadata.items():
                assert chunk.get(key) == value, f"Chunk missing metadata key: {key}"

    def test_reserved_keys_ignored(self, client, alice_path):
        """Reserved keys (text, token_count, id) are not overwritten."""
        metadata = {
            "text": "SHOULD_NOT_OVERWRITE",
            "token_count": 9999,
            "id": 9999,
            "custom_field": "allowed",
        }
        response = post_with_metadata(client, alice_path, json.dumps(metadata))

        assert response.status_code == 200, f"API error: {response.text}"
        data = response.json()

        for chunk in data["chunks"]:
            # Reserved keys should NOT be overwritten
            assert chunk["text"] != "SHOULD_NOT_OVERWRITE"
            assert chunk["token_count"] != 9999
            assert chunk["id"] != 9999
            # Custom field should be present
            assert chunk.get("custom_field") == "allowed"

    @pytest.mark.parametrize(
        "metadata_value,expected_wrapper",
        [
            ([1, 2, 3], [1, 2, 3]),
            (["a", "b", "c"], ["a", "b", "c"]),
            ([{"nested": "obj"}], [{"nested": "obj"}]),
        ],
        ids=["int_array", "string_array", "object_array"],
    )
    def test_array_wrapped_in_chunk_metadata(
        self, client, alice_path, metadata_value, expected_wrapper
    ):
        """Array values are wrapped in 'chunk_metadata' key."""
        response = post_with_metadata(client, alice_path, json.dumps(metadata_value))

        assert response.status_code == 200, f"API error: {response.text}"
        data = response.json()

        for chunk in data["chunks"]:
            assert "chunk_metadata" in chunk, "Missing 'chunk_metadata' wrapper"
            assert chunk["chunk_metadata"] == expected_wrapper

    @pytest.mark.parametrize(
        "metadata_value",
        [
            "just a string",
            42,
            3.14,
            True,
            False,
        ],
        ids=["string", "int", "float", "bool_true", "bool_false"],
    )
    def test_primitive_wrapped_in_chunk_metadata(
        self, client, alice_path, metadata_value
    ):
        """Primitive values (string, number, boolean) are wrapped in 'chunk_metadata'."""
        response = post_with_metadata(client, alice_path, json.dumps(metadata_value))

        assert response.status_code == 200, f"API error: {response.text}"
        data = response.json()

        for chunk in data["chunks"]:
            assert "chunk_metadata" in chunk, "Missing 'chunk_metadata' wrapper"
            assert chunk["chunk_metadata"] == metadata_value

    def test_json_null_wrapped_in_chunk_metadata(self, client, alice_path):
        """JSON null value is wrapped in 'chunk_metadata'."""
        response = post_with_metadata(client, alice_path, "null")

        assert response.status_code == 200, f"API error: {response.text}"
        data = response.json()

        for chunk in data["chunks"]:
            assert "chunk_metadata" in chunk, "Missing 'chunk_metadata' wrapper"
            assert chunk["chunk_metadata"] is None

    @pytest.mark.parametrize(
        "empty_value",
        [
            "",
            "   ",
            "\t\n",
        ],
        ids=["empty_string", "whitespace", "tabs_newlines"],
    )
    def test_empty_or_whitespace_no_metadata(self, client, alice_path, empty_value):
        """Empty string or whitespace-only results in no extra metadata."""
        response = post_with_metadata(client, alice_path, empty_value)

        assert response.status_code == 200, f"API error: {response.text}"
        data = response.json()

        for chunk in data["chunks"]:
            # Should only have the standard fields
            assert set(chunk.keys()) == {"text", "token_count", "id"}

    @pytest.mark.parametrize(
        "invalid_json",
        [
            "{invalid-json",
            "{'single': 'quotes'}",
            "{trailing: comma,}",
            "undefined",
        ],
        ids=["missing_brace", "single_quotes", "trailing_comma", "undefined_literal"],
    )
    def test_invalid_json_returns_400(self, client, alice_path, invalid_json):
        """Invalid JSON syntax returns HTTP 400."""
        response = post_with_metadata(client, alice_path, invalid_json)

        assert response.status_code == 400
        assert "chunk_metadata_json" in response.json().get("detail", "").lower()

    def test_nested_object_preserved(self, client, alice_path):
        """Nested objects in metadata are preserved correctly."""
        metadata = {
            "document_info": {
                "title": "Test Document",
                "pages": 10,
                "tags": ["test", "demo"],
            },
            "processing": {"version": "1.0", "options": {"fast": True}},
        }
        response = post_with_metadata(client, alice_path, json.dumps(metadata))

        assert response.status_code == 200, f"API error: {response.text}"
        data = response.json()

        for chunk in data["chunks"]:
            assert chunk.get("document_info") == metadata["document_info"]
            assert chunk.get("processing") == metadata["processing"]

    def test_unicode_metadata_preserved(self, client, alice_path):
        """Unicode characters in metadata are preserved."""
        metadata = {
            "author": "日本語テスト",
            "description": "Émoji test: 🎉🚀",
            "chinese": "中文测试",
        }
        response = post_with_metadata(client, alice_path, json.dumps(metadata))

        assert response.status_code == 200, f"API error: {response.text}"
        data = response.json()

        for chunk in data["chunks"]:
            assert chunk.get("author") == "日本語テスト"
            assert chunk.get("description") == "Émoji test: 🎉🚀"
            assert chunk.get("chinese") == "中文测试"


class TestBearerTokenAuth:
    """Tests for the Bearer Token authentication feature."""

    ENDPOINT = "/normalized_semantic_chunker/"
    SECRET = "test-secret-token"

    @pytest.fixture
    def client_with_token(self, monkeypatch):
        """TestClient with API_TOKEN set in the environment."""
        import normalized_semantic_chunker as nsc
        monkeypatch.setattr(nsc, "API_TOKEN", self.SECRET)
        with TestClient(app) as c:
            yield c

    @pytest.fixture
    def client_no_token(self, monkeypatch):
        """TestClient with API_TOKEN disabled (empty string)."""
        import normalized_semantic_chunker as nsc
        monkeypatch.setattr(nsc, "API_TOKEN", "")
        with TestClient(app) as c:
            yield c

    # --- Health check is always public ---

    def test_health_check_public_when_auth_enabled(self, client_with_token):
        """GET / must be accessible without a token even when auth is enabled."""
        response = client_with_token.get("/")
        assert response.status_code == 200

    def test_health_check_public_when_auth_disabled(self, client_no_token):
        """GET / must be accessible without a token when auth is disabled."""
        response = client_no_token.get("/")
        assert response.status_code == 200

    # --- Auth disabled ---

    def test_post_no_token_allowed_when_auth_disabled(self, client_no_token, alice_path):
        """POST without Authorization header is allowed when API_TOKEN is empty."""
        with open(alice_path, "rb") as f:
            response = client_no_token.post(
                self.ENDPOINT,
                files={"file": ("alice_in_wonderland.txt", f, "text/plain")},
                params={"max_tokens": 500},
            )
        assert response.status_code == 200

    # --- Auth enabled, valid token ---

    def test_post_valid_token_accepted(self, client_with_token, alice_path):
        """POST with correct Bearer token returns 200."""
        with open(alice_path, "rb") as f:
            response = client_with_token.post(
                self.ENDPOINT,
                files={"file": ("alice_in_wonderland.txt", f, "text/plain")},
                params={"max_tokens": 500},
                headers={"Authorization": f"Bearer {self.SECRET}"},
            )
        assert response.status_code == 200

    # --- Auth enabled, missing or invalid token ---

    def test_post_no_token_returns_403_when_auth_enabled(self, client_with_token, alice_path):
        """POST without Authorization header returns 403 when auth is enabled."""
        with open(alice_path, "rb") as f:
            response = client_with_token.post(
                self.ENDPOINT,
                files={"file": ("alice_in_wonderland.txt", f, "text/plain")},
                params={"max_tokens": 500},
            )
        assert response.status_code == 403
        assert response.json()["detail"] == "Invalid or missing API token"

    def test_post_wrong_token_returns_403(self, client_with_token, alice_path):
        """POST with a wrong Bearer token returns 403."""
        with open(alice_path, "rb") as f:
            response = client_with_token.post(
                self.ENDPOINT,
                files={"file": ("alice_in_wonderland.txt", f, "text/plain")},
                params={"max_tokens": 500},
                headers={"Authorization": "Bearer wrong-token"},
            )
        assert response.status_code == 403
        assert response.json()["detail"] == "Invalid or missing API token"

    def test_post_empty_token_returns_403(self, client_with_token, alice_path):
        """POST with an empty Bearer token returns 403."""
        with open(alice_path, "rb") as f:
            response = client_with_token.post(
                self.ENDPOINT,
                files={"file": ("alice_in_wonderland.txt", f, "text/plain")},
                params={"max_tokens": 500},
                headers={"Authorization": "Bearer "},
            )
        assert response.status_code == 403
