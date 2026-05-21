#!/usr/bin/env python3
"""
Compare GLM-5 timeline extractions vs ground truth timelines for i2m4 dataset.

This script compares events extracted by GLM-5 against ground truth timelines,
using semantic matching with embeddings.

Matching criteria:
- (same text OR semantic similarity > 0.8) AND
- time difference < 50% of smaller time
"""

import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# Add parent directory to path to import embedding_extraction
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Global embedding cache
EMBEDDING_CACHE = None


def load_embedding_cache():
    """Load the global embedding cache."""
    global EMBEDDING_CACHE

    if EMBEDDING_CACHE is not None:
        return EMBEDDING_CACHE

    # Try multiple cache locations
    cache_paths = [
        Path('/data/weissjc/claude_sandbox/coannotator_sanitary/.embedding_cache.pkl'),
        Path(__file__).parent.parent.parent / '.embedding_cache.pkl',
    ]

    for cache_path in cache_paths:
        if cache_path.exists():
            print(f"Loading embedding cache from: {cache_path}")
            with open(cache_path, 'rb') as f:
                EMBEDDING_CACHE = pickle.load(f)
            print(f"  Loaded {len(EMBEDDING_CACHE)} cached embeddings")
            return EMBEDDING_CACHE

    print("Warning: No embedding cache found. Will compute embeddings from scratch.")
    EMBEDDING_CACHE = {}
    return EMBEDDING_CACHE


def parse_bsv_file(bsv_path: Path) -> List[Dict]:
    """
    Parse a BSV file and return list of events.

    Args:
        bsv_path: Path to BSV file

    Returns:
        List of event dictionaries with keys: uid4, mention, time, bounds
    """
    events = []

    with open(bsv_path, 'r') as f:
        lines = f.readlines()

    if len(lines) <= 1:
        return events

    # Skip header
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue

        parts = line.split('|')
        if len(parts) < 5:
            continue

        uid4 = parts[0]
        mention = parts[3]
        time_str = parts[4]

        # Parse time
        try:
            time = float(time_str)
        except (ValueError, TypeError):
            # Skip events with invalid time (like N/A)
            continue

        events.append({
            'uid4': uid4,
            'mention': mention,
            'time': time
        })

    return events


def compute_embeddings_batch_direct(texts: List[str]) -> np.ndarray:
    """
    Compute embeddings for a batch of texts, using cache when available.

    Args:
        texts: List of text strings to embed

    Returns:
        numpy array of shape (len(texts), embedding_dim) with normalized embeddings
    """
    cache = load_embedding_cache()

    # Check which texts are in cache
    uncached = [t for t in texts if t not in cache]

    if uncached:
        print(f"  Warning: {len(uncached)} texts not in cache, computing embeddings...")
        print(f"  Sample uncached texts: {uncached[:3]}")

        # Try to compute embeddings for uncached texts
        try:
            # Find embed_helper.py
            embed_helper_paths = [
                Path('/data/weissjc/claude_sandbox/coannotator_sanitary/embed_helper.py'),
                Path(__file__).parent.parent.parent / 'embed_helper.py',
            ]

            embed_helper_path = None
            for path in embed_helper_paths:
                if path.exists():
                    embed_helper_path = path
                    break

            if embed_helper_path:
                # Use embed_helper.py via subprocess
                result = subprocess.run(
                    [sys.executable, str(embed_helper_path)],
                    input=json.dumps(uncached),
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=60
                )

                embeddings = json.loads(result.stdout)
                for t, emb in zip(uncached, embeddings):
                    cache[t] = emb
            else:
                raise FileNotFoundError("Could not find embed_helper.py")
        except Exception as e:
            print(f"  Error computing embeddings: {e}")
            print(f"  Using random vectors for {len(uncached)} uncached texts")
            # Use random vectors for uncached texts (normalized)
            import random
            for t in uncached:
                # Use a random vector (normalized)
                vec = [random.gauss(0, 1) for _ in range(1024)]  # Qwen3-Embedding-0.6B uses 1024 dims
                # Normalize
                norm = sum(x**2 for x in vec) ** 0.5
                vec = [x / norm for x in vec]
                cache[t] = vec

    # Retrieve all embeddings from cache
    result = [cache[t] for t in texts]
    return np.array(result)


def compute_similarity(text1: str, text2: str, cache: Dict[str, np.ndarray] = None) -> float:
    """
    Compute semantic similarity between two texts using embeddings.

    Args:
        text1: First text
        text2: Second text
        cache: Optional cache dictionary for embeddings

    Returns:
        Cosine similarity score between 0 and 1
    """
    # Check if texts are identical
    if text1 == text2:
        return 1.0

    # Get embeddings
    if cache is not None and text1 in cache and text2 in cache:
        emb1 = cache[text1]
        emb2 = cache[text2]
    else:
        embeddings = compute_embeddings_batch([text1, text2])
        emb1 = embeddings[0]
        emb2 = embeddings[1]

        if cache is not None:
            cache[text1] = emb1
            cache[text2] = emb2

    # Compute cosine similarity
    similarity = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    return similarity


def match_events(gt_events: List[Dict], glm5_events: List[Dict],
                 similarity_threshold: float = 0.8,
                 time_threshold_fraction: float = 0.5) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Match events between ground truth and GLM-5 extractions.

    Matching criteria:
    - (same text OR semantic similarity > threshold) AND
    - time difference < fraction of smaller absolute time

    Args:
        gt_events: List of ground truth events
        glm5_events: List of GLM-5 extracted events
        similarity_threshold: Minimum semantic similarity threshold
        time_threshold_fraction: Maximum time difference as fraction of smaller time

    Returns:
        Tuple of:
        - List of matched pairs (gt_idx, glm5_idx)
        - List of unmatched GT indices
        - List of unmatched GLM-5 indices
    """
    if not gt_events or not glm5_events:
        return [], list(range(len(gt_events))), list(range(len(glm5_events)))

    # Cache for embeddings
    embedding_cache = {}

    # Compute all embeddings at once for efficiency
    all_texts = list(set([e['mention'] for e in gt_events] + [e['mention'] for e in glm5_events]))
    if len(all_texts) > 0:
        print(f"    Computing embeddings for {len(all_texts)} unique texts...")
        embeddings = compute_embeddings_batch_direct(all_texts)
        for text, emb in zip(all_texts, embeddings):
            embedding_cache[text] = emb

    # Track matches
    matched_pairs = []
    matched_gt = set()
    matched_glm5 = set()

    # For each GT event, find best matching GLM-5 event
    for gt_idx, gt_event in enumerate(gt_events):
        best_match_idx = None
        best_similarity = 0.0

        gt_time = gt_event['time']

        for glm5_idx, glm5_event in enumerate(glm5_events):
            if glm5_idx in matched_glm5:
                continue

            glm5_time = glm5_event['time']

            # Check time constraint
            # Time difference must be < 50% of smaller absolute time
            smaller_time = min(abs(gt_time), abs(glm5_time))

            # Handle edge case where time is 0
            if smaller_time == 0:
                time_diff_threshold = 1.0  # Allow 1 hour difference if time is 0
            else:
                time_diff_threshold = time_threshold_fraction * smaller_time

            time_diff = abs(gt_time - glm5_time)

            if time_diff > time_diff_threshold:
                continue

            # Check text similarity
            similarity = compute_similarity(gt_event['mention'], glm5_event['mention'], embedding_cache)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = glm5_idx

        # Accept match if similarity threshold met
        if best_match_idx is not None and best_similarity > similarity_threshold:
            matched_pairs.append((gt_idx, best_match_idx))
            matched_gt.add(gt_idx)
            matched_glm5.add(best_match_idx)

    # Find unmatched events
    unmatched_gt = [i for i in range(len(gt_events)) if i not in matched_gt]
    unmatched_glm5 = [i for i in range(len(glm5_events)) if i not in matched_glm5]

    return matched_pairs, unmatched_gt, unmatched_glm5


def analyze_case(gt_bsv: Path, glm5_bsv: Path) -> Dict:
    """
    Analyze a single case by comparing GT and GLM-5 timelines.

    Args:
        gt_bsv: Path to ground truth BSV file
        glm5_bsv: Path to GLM-5 BSV file

    Returns:
        Dictionary with metrics and details
    """
    # Parse files
    gt_events = parse_bsv_file(gt_bsv)
    glm5_events = parse_bsv_file(glm5_bsv)

    # Match events
    matched_pairs, unmatched_gt, unmatched_glm5 = match_events(gt_events, glm5_events)

    # Compute metrics
    num_gt = len(gt_events)
    num_glm5 = len(glm5_events)
    num_matched = len(matched_pairs)

    # Recall: how many GT events were found
    recall = num_matched / num_gt if num_gt > 0 else 0.0

    # Precision: how many GLM-5 events match GT
    precision = num_matched / num_glm5 if num_glm5 > 0 else 0.0

    # F1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'gt_file': str(gt_bsv),
        'glm5_file': str(glm5_bsv),
        'num_gt_events': num_gt,
        'num_glm5_events': num_glm5,
        'num_matched': num_matched,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'matched_pairs': matched_pairs,
        'unmatched_gt_indices': unmatched_gt,
        'unmatched_glm5_indices': unmatched_glm5
    }


def main():
    """Main function to run extraction comparison."""

    # Paths
    i2m4_dir = Path('/data/weissjc/lns/tta/uidtts/gsandbox/batch_processor/gap_detection/i2m4_analysis')
    gt_dir = i2m4_dir / 'ground_truth' / 'timelines'
    glm5_dir = i2m4_dir / 'glm5_output' / 'timelines'
    output_dir = i2m4_dir / 'comparison' / 'extraction_quality'
    output_file = output_dir / 'extraction_comparison.json'

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all GT files
    gt_files = sorted(gt_dir.glob('*.bsv'))
    print(f"Found {len(gt_files)} ground truth files")

    # Find matching GLM-5 files
    cases = []
    for gt_file in gt_files:
        case_id = gt_file.stem  # filename without extension
        glm5_file = glm5_dir / f"{case_id}_positions_timeline_1.bsv"

        if glm5_file.exists():
            cases.append((case_id, gt_file, glm5_file))
        else:
            print(f"Warning: No GLM-5 output for case {case_id}")

    print(f"Found {len(cases)} matching cases\n")

    # Analyze each case
    results = []
    total_recall = 0.0
    total_precision = 0.0
    total_f1 = 0.0

    for i, (case_id, gt_file, glm5_file) in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] Analyzing case {case_id}...")

        result = analyze_case(gt_file, glm5_file)
        result['case_id'] = case_id
        results.append(result)

        print(f"  GT events: {result['num_gt_events']}")
        print(f"  GLM-5 events: {result['num_glm5_events']}")
        print(f"  Matched: {result['num_matched']}")
        print(f"  Recall: {result['recall']:.3f}")
        print(f"  Precision: {result['precision']:.3f}")
        print(f"  F1: {result['f1']:.3f}\n")

        total_recall += result['recall']
        total_precision += result['precision']
        total_f1 += result['f1']

    # Compute overall metrics
    num_cases = len(cases)
    avg_recall = total_recall / num_cases if num_cases > 0 else 0.0
    avg_precision = total_precision / num_cases if num_cases > 0 else 0.0
    avg_f1 = total_f1 / num_cases if num_cases > 0 else 0.0

    # Save results
    output_data = {
        'summary': {
            'num_cases': num_cases,
            'avg_recall': avg_recall,
            'avg_precision': avg_precision,
            'avg_f1': avg_f1
        },
        'cases': results
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Print summary
    print("\n" + "="*60)
    print("OVERALL RESULTS")
    print("="*60)
    print(f"Number of cases: {num_cases}")
    print(f"Average Recall: {avg_recall:.3f}")
    print(f"Average Precision: {avg_precision:.3f}")
    print(f"Average F1: {avg_f1:.3f}")
    print("="*60)


if __name__ == '__main__':
    main()