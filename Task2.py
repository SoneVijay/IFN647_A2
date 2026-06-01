"""
IFN647 Assignment 2 - Task 2: Custom Model (Model_C)
KL-Divergence Relevance Model with gap-based automatic PRF set size and RM3.

Pipeline:
  1. BM25 initial ranking + gap-based PRF_K detection in [MIN_K, MAX_K)
  2. JM-smoothed query-likelihood weights for top-K docs
  3. Build relevance model P(w|R) via weighted sum over top-K vocab
  4. RM3 interpolation: rm3 = ALPHA*P(w|R) + (1-ALPHA)*P(w|Q)
  5. Score all docs by log-likelihood under rm3

Gap-based detection:
  gap[i] = score[i] - score[i+1]  for i in [MIN_K, MAX_K)
  K = position of the largest gap (natural boundary in BM25 score curve).

Parameters (selected by GridSearch_Task2.py):
  MIN_K = 5     smallest allowed pseudo-relevant set size
  MAX_K = 20    upper bound on the gap search window
  LAM   = 0.05  Jelinek-Mercer smoothing lambda
  ALPHA = 0.7   RM3 interpolation weight (P(w|R) vs P(w|Q))
"""

import os
import math
from utils import (
    EXTRA_STOPWORDS, DOC_DIR, OUTPUT_DIR,
    load_stopwords, docParser, queryParser, parse_topics,
    avg_len, build_inv_index, collection_term_freq, collection_size, save_ranking,
)
from Task1 import bm_25

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MIN_K = 5
MAX_K = 20
LAM   = 0.05
ALPHA = 0.7


# =============================================================================
# Gap detector
# =============================================================================

def _detect_k(ranked_bm25, bm25_scores, min_k, max_k):
    """Return k = position of the largest BM25 score gap in [min_k, max_k)."""
    n        = len(ranked_bm25)
    limit    = min(max_k, n - 1)
    best_gap = -1.0
    best_k   = min_k
    for i in range(min_k - 1, limit):
        gap = bm25_scores[ranked_bm25[i]] - bm25_scores[ranked_bm25[i + 1]]
        if gap > best_gap:
            best_gap = gap
            best_k   = i + 1
    return best_k


# =============================================================================
# Model C – KL-Divergence Relevance Model with gap-based PRF set size
# =============================================================================

def model_c(query_tf, collection, inv_index, avdl, coll_freq, filt_size, alpha=ALPHA):
    """Score all documents using KL-divergence RM; PRF set size detected from score gaps.

    Args:
        query_tf   : {term: frequency} for the query
        collection : {docid: Doc} for the dataset
        inv_index  : {term: {docid: freq}} inverted index
        avdl       : average document length in the collection
        coll_freq  : {term: total_count} across the collection
        filt_size  : total word count across the collection

    Returns:
        {docid: score} where higher score means closer to the relevance model
    """

    # Step 1: BM25 initial ranking + gap-based PRF_K selection
    bm25_scores = bm_25(query_tf, collection, inv_index, avdl)
    ranked_bm25 = sorted(bm25_scores, key=bm25_scores.get, reverse=True)
    prf_k       = _detect_k(ranked_bm25, bm25_scores, MIN_K, MAX_K)
    top_docs    = ranked_bm25[:prf_k]

    if not top_docs:
        return {docid: 0.0 for docid in collection}

    # Step 2: Document weights via query likelihood (JM smoothed)
    log_w = {}
    for docid in top_docs:
        doc = collection[docid]
        dl  = doc.get_doc_size()
        lw  = 0.0
        for term, qf in query_tf.items():
            tf_d = doc.terms.get(term, 0)
            cf_t = coll_freq.get(term, 0)
            p = ((1 - LAM) * (tf_d / dl if dl > 0 else 0.0)
                 + LAM * (cf_t / filt_size if filt_size > 0 else 0.0))
            lw += qf * math.log(p if p > 0 else 1e-15)
        log_w[docid] = lw

    max_lw  = max(log_w.values())
    weights = {d: math.exp(lw - max_lw) for d, lw in log_w.items()}
    total_w = sum(weights.values())
    if total_w > 0:
        weights = {d: w / total_w for d, w in weights.items()}

    # Step 3: Build relevance model P(w|R)
    vocab = set()
    for docid in top_docs:
        vocab.update(collection[docid].terms.keys())

    rm = {}
    for term in vocab:
        p_w_r = 0.0
        for docid in top_docs:
            doc   = collection[docid]
            dl    = doc.get_doc_size()
            tf_d  = doc.terms.get(term, 0)
            cf_t  = coll_freq.get(term, 0)
            p_w_d = ((1 - LAM) * (tf_d / dl if dl > 0 else 0.0)
                     + LAM * (cf_t / filt_size if filt_size > 0 else 0.0))
            p_w_r += weights[docid] * p_w_d
        if p_w_r > 0:
            rm[term] = p_w_r

    total_rm = sum(rm.values())
    if total_rm > 0:
        rm = {t: v / total_rm for t, v in rm.items()}

    # RM3: interpolate P(w|R) with original query distribution P(w|Q)
    total_qf = sum(query_tf.values())
    p_q      = {t: qf / total_qf for t, qf in query_tf.items()}
    rm3      = {}
    for term in set(rm) | set(p_q):
        rm3[term] = alpha * rm.get(term, 0.0) + (1 - alpha) * p_q.get(term, 0.0)

    # Step 4: Score all documents by log-likelihood under the relevance model
    scores = {}
    for docid, doc in collection.items():
        dl    = doc.get_doc_size()
        score = 0.0
        for term, p_w_r in rm3.items():
            tf_d  = doc.terms.get(term, 0)
            cf_t  = coll_freq.get(term, 0)
            p_w_d = ((1 - LAM) * (tf_d / dl if dl > 0 else 0.0)
                     + LAM * (cf_t / filt_size if filt_size > 0 else 0.0))
            if p_w_d > 0:
                score += p_w_r * math.log(p_w_d)
        scores[docid] = score
    return scores


# =============================================================================
# Main
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    stop_wordList = list(set(
        load_stopwords(os.path.join(BASE_DIR, "common-english-words.txt")) + EXTRA_STOPWORDS
    ))
    topics = parse_topics()

    for topic_id, topic_title in sorted(topics.items()):
        dataset_number = topic_id.replace("R", "")
        dataset_folder = os.path.join(DOC_DIR, f"Dataset{dataset_number}")
        if not os.path.exists(dataset_folder):
            print(f"[SKIP] {dataset_folder}")
            continue

        print(f"\n{'='*60}\nProcessing {topic_id} – \"{topic_title}\"\n{'='*60}")

        coll      = docParser(stop_wordList, dataset_folder)
        inv_index = build_inv_index(coll)
        query_tf  = queryParser(topic_title, stop_wordList)
        cf        = collection_term_freq(coll)
        fs        = collection_size(coll)
        avdl      = avg_len(coll)

        bm25_scores = bm_25(query_tf, coll, inv_index, avdl)
        ranked_bm25 = sorted(bm25_scores, key=bm25_scores.get, reverse=True)
        detected_k  = _detect_k(ranked_bm25, bm25_scores, MIN_K, MAX_K)
        print(f"  Gap-detected prf_k = {detected_k}  (window [{MIN_K}, {MAX_K}))")

        scores  = model_c(query_tf, coll, inv_index, avdl, cf, fs)
        path    = os.path.join(OUTPUT_DIR, f"ModelC_{topic_id}_Ranking.dat")
        ranked  = save_ranking(path, scores, "ModelC_Score", topic_title)

        print(f"\n{topic_id} Model_C Top 10:")
        for doc_id, score in ranked[:10]:
            print(f"  {doc_id}  {score:.6f}")
        print(f"[SAVED] {path}")


if __name__ == "__main__":
    main()
