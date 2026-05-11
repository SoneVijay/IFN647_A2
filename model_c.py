"""
IFN647 Assignment 2 - Model C
Pseudo-Relevance Feedback (PRF) with BM25 re-ranking.
"""

import os
import math
from collections import defaultdict
from utils import (
    EXTRA_STOPS, DOC_DIR, OUTPUT_DIR,
    load_stop_words, queryParser, build_dataset_index,
    parse_topics, write_ranking, print_top10,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

K1    = 1.2
K2    = 500
B     = 0.75
TOP_K = 5
TOP_M = 10
ALPHA = 1


# =============================================================================
# BM25 helper (same formula as Baseline1, accepts a pre-built query_tf dict)
# =============================================================================

def _bm25(query_tf, collection, inv_index, avdl):
    N = len(collection)
    scores = defaultdict(float)
    for term, qft in query_tf.items():
        if term not in inv_index:
            continue
        nt       = len(inv_index[term])
        idf_part = math.log10(1 + (N - nt + 0.5) / (nt + 0.5))
        qf_part  = ((K2 + 1) * qft) / (K2 + qft)
        for docid, ft in inv_index[term].items():
            dl    = collection[docid].get_doc_size()
            K_val = K1 * ((1 - B) + B * dl / avdl)
            scores[docid] += idf_part * ((K1 + 1) * ft) / (K_val + ft) * qf_part
    for docid in collection:
        if docid not in scores:
            scores[docid] = 0.0
    return dict(scores)


# =============================================================================
# Model C – PRF + BM25 re-ranking
# =============================================================================
# Step 1: Initial BM25 retrieval with the original query.
# Step 2: Take top_k docs as the pseudo-relevant set.
# Step 3: For each term in those docs (excluding original query terms),
#         compute Rocchio-style weight = sum over PRF docs of (freq / dl).
# Step 4: Add top_m expansion terms to the query with weight alpha.
# Step 5: Re-rank the full collection using BM25 with the expanded query.

def model_c_score(query_tf, collection, inv_index, avdl,
                  top_k=TOP_K, top_m=TOP_M, alpha=ALPHA):
    # Step 1
    initial_scores = _bm25(query_tf, collection, inv_index, avdl)
    ranked_initial = sorted(initial_scores.items(), key=lambda x: x[1], reverse=True)

    # Step 2
    prf_ids = [docid for docid, _ in ranked_initial[:top_k]]

    # Step 3
    original_terms = set(query_tf.keys())
    term_weights = {}
    for docid in prf_ids:
        doc = collection[docid]
        dl  = doc.get_doc_size()
        if dl == 0:
            continue
        for term, freq in doc.terms.items():
            if term not in original_terms:
                term_weights[term] = term_weights.get(term, 0) + freq / dl

    # Step 4
    expansion = sorted(term_weights.items(), key=lambda x: x[1], reverse=True)[:top_m]
    expanded_query = dict(query_tf)
    for term, _ in expansion:
        expanded_query[term] = expanded_query.get(term, 0) + alpha

    # Step 5
    return _bm25(expanded_query, collection, inv_index, avdl)


# =============================================================================
# Main
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    stop_words = load_stop_words(os.path.join(BASE_DIR, "common-english-words.txt"))
    stop_words |= set(EXTRA_STOPS)

    topics = parse_topics()
    print(f"Topics: {len(topics)}  |  Output: {OUTPUT_DIR}\n")

    for topic_num in sorted(topics.keys()):
        dataset_path = os.path.join(DOC_DIR, f"Dataset{topic_num[1:]}")
        if not os.path.isdir(dataset_path):
            print(f"[SKIP] {topic_num}: {dataset_path} not found")
            continue

        title    = topics[topic_num]
        query_tf = queryParser(title, stop_words)
        print(f"{topic_num}  \"{title}\"  query={query_tf}")

        collection, inv_index, coll_freq, filt_size, avdl = build_dataset_index(
            dataset_path, stop_words
        )

        ranked = write_ranking(
            os.path.join(OUTPUT_DIR, f"ModelC_{topic_num}_Ranking.dat"),
            model_c_score(query_tf, collection, inv_index, avdl),
            comment=(
                f"Query: {title} | PRF+BM25 | "
                f"top_k={TOP_K} top_m={TOP_M} alpha={ALPHA}"
            ),
        )
        print_top10(topic_num, ranked, "PRF+BM25")

    print(f"\nDone. Rankings saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
