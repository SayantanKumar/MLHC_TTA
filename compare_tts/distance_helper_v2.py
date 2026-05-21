import numpy as np
import pandas as pd
import requests
import argparse
from sklearn.metrics import pairwise_distances

SERVER_PORT = 8000  # Default port (should match run_server.py)

def get_sentence_embedding(sentences: list[str], port: int = SERVER_PORT) -> np.ndarray:
    """
    Get sentence embeddings from the local server.
    """
    try:
        response = requests.post(
            f"http://localhost:{port}/embed",
            json={"sentences": [str(s)[:512] for s in sentences]},
            timeout=60 * 60 * 24 * 7  # 1 week in seconds
        )
        response.raise_for_status()
        return np.array(response.json()["embeddings"])
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error getting embeddings: {e}")

def get_and_write_embeddings(f1: str, f2: str, outfile: str, port: int = SERVER_PORT) -> None:
    """Read two CSV files, compute embeddings, and write distance matrix to output file."""
    try:
        df1 = pd.read_csv(f1, keep_default_na=False)
    except FileNotFoundError:
        print(f"Error: File '{f1}' not found.")
        return

    try:
        df2 = pd.read_csv(f2, keep_default_na=False)
    except FileNotFoundError:
        print(f"Error: File '{f2}' not found.")
        return

    if df2 is not None and df1 is not None:
        embs1 = get_sentence_embedding(df1['event'].tolist(), port)
        embs2 = get_sentence_embedding(df2['event'].tolist(), port)
        
        distance_matrix = pairwise_distances(embs1, embs2, metric='cosine')
        outdf = pd.DataFrame(distance_matrix)
        outdf.to_csv(outfile, index=False)

def main():
    parser = argparse.ArgumentParser(description="Compute embeddings and distance matrix.")
    parser.add_argument("f1", help="The name of the file 1 to read")
    parser.add_argument("f2", help="The name of the file 2 to read")
    parser.add_argument("outfile", help="Where to save the similarity matrix")
    parser.add_argument("--port", type=int, default=SERVER_PORT, help="Port of the embedding server")

    args = parser.parse_args()
    get_and_write_embeddings(args.f1, args.f2, args.outfile, args.port)

# if __name__ == "__main__":
#     main()