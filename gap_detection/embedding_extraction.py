#!/usr/bin/env python3
"""
Embedding extraction utilities for textual-tabular gap detection.

This module provides functions to compute and cache embeddings for:
- Textual mentions from timelines
- Tabular events from RAG results

Uses Qwen3-Embedding-0.6B model with disk caching.
"""

import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# Cache path - prefer existing cache if available
# Try multiple locations for the embedding cache
CACHE_PATHS = [
    Path(__file__).parent / ".embedding_cache.pkl",  # Local cache
    Path("/data/weissjc/claude_sandbox/coannotator_sanitary/.embedding_cache.pkl"),  # Existing cache
]

CACHE_PATH = None
for path in CACHE_PATHS:
    if path.exists():
        CACHE_PATH = path
        break

if CACHE_PATH is None:
    # Use local cache path (will be created)
    CACHE_PATH = CACHE_PATHS[0]

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


def load_cache() -> Dict[str, List[float]]:
    """Load embedding cache from disk."""
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def save_cache(cache: Dict[str, List[float]]) -> None:
    """Save embedding cache to disk."""
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)


def compute_embeddings_batch(texts: List[str]) -> np.ndarray:
    """
    Compute embeddings for a batch of texts using Qwen3-Embedding-0.6B.

    Uses caching to avoid recomputing embeddings for texts seen before.

    Args:
        texts: List of text strings to embed

    Returns:
        numpy array of shape (len(texts), embedding_dim) with normalized embeddings
    """
    cache = load_cache()

    # Find uncached texts
    uncached = [t for t in texts if t not in cache]

    if uncached:
        print(f"  Computing embeddings for {len(uncached)} new texts...")

        # Use embed_helper.py script (runs in lco venv)
        embed_helper_path = Path(__file__).parent.parent.parent.parent.parent / 'claude_sandbox' / 'coannotator_sanitary' / 'embed_helper.py'

        if not embed_helper_path.exists():
            print(f"Warning: embed_helper.py not found at {embed_helper_path}")
            print("Falling back to direct sentence-transformers import...")
            # Fallback: direct import
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(MODEL_NAME)
                embeddings = model.encode(uncached, normalize_embeddings=True)
                for t, emb in zip(uncached, embeddings):
                    cache[t] = emb.tolist()
                save_cache(cache)
            except ImportError:
                raise ImportError(
                    "sentence-transformers not available. "
                    "Install with: pip install sentence-transformers"
                )
        else:
            # Use embed_helper.py via subprocess
            try:
                result = subprocess.run(
                    [sys.executable, str(embed_helper_path)],
                    input=json.dumps(uncached),
                    capture_output=True,
                    text=True,
                    check=True
                )
                embeddings = json.loads(result.stdout)
                for t, emb in zip(uncached, embeddings):
                    cache[t] = emb
                save_cache(cache)
            except subprocess.CalledProcessError as e:
                print(f"Error running embed_helper.py: {e}")
                print(f"stderr: {e.stderr}")
                raise

    # Retrieve all embeddings from cache
    result = [cache[t] for t in texts]
    return np.array(result)


def extract_textual_mentions(timelines: Dict[str, List[Dict]]) -> Tuple[List[str], Dict[str, Dict[str, int]]]:
    """
    Extract all textual mentions from timelines.

    Args:
        timelines: Dictionary mapping case_id to list of events

    Returns:
        Tuple of:
        - List of unique mention texts
        - Dictionary mapping (case_id, uid4) to index in mentions list
    """
    mentions_set = set()
    mention_map = {}  # (case_id, uid4) -> index

    for case_id, events in timelines.items():
        for event in events:
            mention_text = event['mention']
            mentions_set.add(mention_text)
            mention_map[(case_id, event['uid4'])] = len(mentions_set) - 1

    mentions_list = list(mentions_set)
    return mentions_list, mention_map


def compute_mention_embeddings(timelines: Dict[str, List[Dict]]) -> Dict[str, np.ndarray]:
    """
    Compute embeddings for all textual mentions in timelines.

    Args:
        timelines: Dictionary mapping case_id to list of events

    Returns:
        Dictionary mapping (case_id, uid4) to embedding vector
    """
    print("Extracting textual mentions...")
    mentions_list, mention_map = extract_textual_mentions(timelines)
    print(f"  Found {len(mentions_list)} unique mentions")

    print("Computing mention embeddings...")
    mention_embeddings = compute_embeddings_batch(mentions_list)

    # Map back to (case_id, uid4)
    embeddings_dict = {}
    for (case_id, uid4), idx in mention_map.items():
        embeddings_dict[(case_id, uid4)] = mention_embeddings[idx]

    return embeddings_dict


def extract_tabular_events(rag_results: Dict[str, 'pd.DataFrame']) -> Tuple[List[str], Dict[str, Dict[int, int]]]:
    """
    Extract all tabular events from RAG results.

    Args:
        rag_results: Dictionary mapping case_id to DataFrame

    Returns:
        Tuple of:
        - List of unique event descriptions
        - Dictionary mapping (case_id, row_index) to index in events list
    """
    import pandas as pd

    events_set = set()
    event_map = {}  # (case_id, row_index) -> index

    for case_id, df in rag_results.items():
        for idx, row in df.iterrows():
            # Use line_level_event as the event description
            event_text = row['line_level_event']
            events_set.add(event_text)
            event_map[(case_id, idx)] = len(events_set) - 1

    events_list = list(events_set)
    return events_list, event_map


def compute_tabular_embeddings(rag_results: Dict[str, 'pd.DataFrame']) -> Dict[str, np.ndarray]:
    """
    Compute embeddings for all tabular events in RAG results.

    Args:
        rag_results: Dictionary mapping case_id to DataFrame

    Returns:
        Dictionary mapping (case_id, row_index) to embedding vector
    """
    print("Extracting tabular events...")
    events_list, event_map = extract_tabular_events(rag_results)
    print(f"  Found {len(events_list)} unique event types")

    print("Computing tabular event embeddings...")
    event_embeddings = compute_embeddings_batch(events_list)

    # Map back to (case_id, row_index)
    embeddings_dict = {}
    for (case_id, row_idx), idx in event_map.items():
        embeddings_dict[(case_id, row_idx)] = event_embeddings[idx]

    return embeddings_dict


def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Cosine similarity score (float between -1 and 1)
    """
    return float(np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))


def cosine_similarity_matrix(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity between two sets of embeddings.

    Args:
        embeddings1: First set of embeddings (n1 x d)
        embeddings2: Second set of embeddings (n2 x d)

    Returns:
        Similarity matrix (n1 x n2)
    """
    # Normalize embeddings
    embeddings1_norm = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2_norm = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

    # Compute similarity
    return np.dot(embeddings1_norm, embeddings2_norm.T)


if __name__ == '__main__':
    import sys
    from data_loading import load_batch_data

    if len(sys.argv) < 2:
        print("Usage: python embedding_extraction.py <batch_dir>")
        sys.exit(1)

    batch_dir = sys.argv[1]

    # Load batch data
    print(f"Loading batch data from: {batch_dir}")
    data = load_batch_data(batch_dir)

    # Compute embeddings for textual mentions
    print("\n=== Computing Textual Mention Embeddings ===")
    mention_embeddings = compute_mention_embeddings(data['timelines'])
    print(f"Computed embeddings for {len(mention_embeddings)} textual mentions")

    # Compute embeddings for tabular events
    print("\n=== Computing Tabular Event Embeddings ===")
    tabular_embeddings = compute_tabular_embeddings(data['rag_results'])
    print(f"Computed embeddings for {len(tabular_embeddings)} tabular events")

    # Show sample similarities
    print("\n=== Sample Similarities ===")
    sample_case = list(data['timelines'].keys())[0]
    sample_events = data['timelines'][sample_case][:3]

    for event in sample_events:
        uid4 = event['uid4']
        mention = event['mention']

        # Get mention embedding
        mention_emb = mention_embeddings.get((sample_case, uid4))
        if mention_emb is None:
            continue

        print(f"\nMention: {mention}")

        # Find top 5 most similar tabular events for this case
        case_tabular_embs = {
            (cid, idx): emb
            for (cid, idx), emb in tabular_embeddings.items()
            if cid == sample_case
        }

        if case_tabular_embs:
            similarities = []
            for (cid, idx), tab_emb in case_tabular_embs.items():
                sim = cosine_similarity(mention_emb, tab_emb)
                similarities.append((idx, sim))

            similarities.sort(key=lambda x: x[1], reverse=True)

            print("  Top 5 similar tabular events:")
            for idx, sim in similarities[:5]:
                df = data['rag_results'][sample_case]
                if idx < len(df):
                    event_name = df.iloc[idx]['line_level_event']
                    print(f"    {sim:.3f}: {event_name}")