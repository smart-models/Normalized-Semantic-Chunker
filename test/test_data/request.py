import requests
import json

# Replace with your actual API endpoint if hosted elsewhere
api_url = "http://localhost:8000/normalized_semantic_chunker/"
file_path = "alice_in_wonderland.txt"  # Your input text file
max_tokens_per_chunk = 512
# model_name = "BAAI/bge-m3" # Optional: specify a different model

try:
    with open(file_path, "rb") as f:
        files = {"file": (file_path, f, "text/plain")}
        params = {"max_tokens": max_tokens_per_chunk}
        # if model_name: # Uncomment to specify a model
        #     params['model'] = model_name

        response = requests.post(api_url, files=files, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        result = response.json()

        print(
            f"Successfully chunked document into {result['metadata']['n_chunks']} chunks."
        )
        # Save the response to a file
        output_file = "response.json"
        with open(output_file, "w", encoding="utf-8") as outfile:
            json.dump(result, outfile, indent=4, ensure_ascii=False)
        print(f"Response saved to {output_file}")
        # print("Metadata:", result['metadata'])
        # print("First chunk:", result['chunks'][0])

except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except requests.exceptions.RequestException as e:
    print(f"API Request failed: {e}")
    if e.response is not None:
        print("Error details:", e.response.text)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
