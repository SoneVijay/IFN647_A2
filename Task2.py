"""
IFN647 Assignment 2 - Task 2: Custom Model (Model_C)
KL-Divergence Relevance Model using pseudo-relevance feedback.
(Section 4 of A2.py)

Model C improves on both baselines by replacing the original query with a
richer representation called a relevance model (RM), estimated from the
documents most likely to be relevant.

Algorithm:
  1. Run BM25 on the original query to get an initial ranking.
  2. Treat the top PRF_K documents as pseudo-relevant (pseudo-relevance
     feedback — we assume BM25's highest-ranked results are probably relevant).
  3. Build a relevance model R over vocabulary terms:
       - Weight each pseudo-relevant document by how well it explains the query.
       - P(w|R) is a weighted mixture of each document's term probabilities.
       - Smoothing: P(x|D) = (1-LAM)*tf(x,D)/dl(D) + LAM*cf(x)/|C|
         where tf is term frequency in D, dl is document length,
         cf is collection term frequency, and |C| is total collection size.
  4. Truncate R to the top TOP_TERMS terms by probability (removes noise from
     low-probability, often generic terms introduced by the smoothing).
  5. Score every document by its log-likelihood under R:
       score(d) = sum over w of  P(w|R) * log P(w|d)
     This is equivalent to ranking by -KL(R || d): documents whose language
     model is closest to the relevance model score highest.

Parameters (selected by unsupervised grid search in GridSearch_Task2.py):
  PRF_K     = 15   top BM25 documents used to build the relevance model
  LAM       = 0.05 Jelinek-Mercer smoothing lambda; low value keeps term
                   probabilities document-specific rather than collection-wide
  TOP_TERMS = 100  number of highest-probability RM terms kept after truncation
"""

import os
import math
from utils import (
    EXTRA_STOPWORDS, DOC_DIR, OUTPUT_DIR,
    load_stopwords, docParser, queryParser, parse_topics,
    avg_len, build_inv_index, collection_term_freq, collection_size, save_ranking,
)
from Task1 import bm_25

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
PRF_K     = 15    # pseudo-relevant set size
LAM       = 0.05  # JM smoothing lambda for P(x|D)
TOP_TERMS = 100   # RM vocabulary size after truncation


# =============================================================================
# Model C – KL-Divergence Relevance Model
# =============================================================================

def model_c(query_tf, collection, inv_index, avdl, coll_freq, filt_size):
    """Score all documents in collection using the KL-divergence relevance model.

    Args:
        query_tf   : {term: frequency} for the query
        collection : {docid: Doc} for the dataset
        inv_index  : {term: {docid: freq}} inverted index
        avdl       : average document length in the collection
        coll_freq  : {term: total_count} across the collection
        filt_size  : total word count across the collection (for smoothing)

    Returns:
        {docid: score} where higher score means more similar to the relevance model
    """

    # ------------------------------------------------------------------
    # Step 1: Initial BM25 retrieval to identify pseudo-relevant documents
    # ------------------------------------------------------------------
    bm25_scores = bm_25(query_tf, collection, inv_index, avdl)
    ranked_bm25 = sorted(bm25_scores, key=bm25_scores.get, reverse=True)
    top_docs    = ranked_bm25[:PRF_K]  # pseudo-relevant set

    if not top_docs:
        return {docid: 0.0 for docid in collection}

    # ------------------------------------------------------------------
    # Step 2: Compute a weight for each pseudo-relevant document
    #
    # Weight reflects how well document D explains the query:
    #   log w(D) = sum_q  tf(q,Q) * log P(q|D)
    #
    # P(q|D) uses JM smoothing so that query terms absent from D still
    # receive a small probability via the collection model (cf/|C|),
    # preventing zero-probability from collapsing the weight to -inf.
    #
    # Subtracting the maximum log-weight before exponentiating keeps
    # all values in a numerically safe range (log-sum-exp trick).
    # ------------------------------------------------------------------
    log_w = {}
    for docid in top_docs:
        doc = collection[docid]
        dl  = doc.get_doc_size()
        lw  = 0.0
        for term, qf in query_tf.items():
            tf_d = doc.terms.get(term, 0)
            cf_t = coll_freq.get(term, 0)
            # JM smoothed probability: mix document model with collection model
            p = ((1 - LAM) * (tf_d / dl if dl > 0 else 0.0)
                 + LAM * (cf_t / filt_size if filt_size > 0 else 0.0))
            # Accumulate log query-likelihood; floor at 1e-15 to avoid log(0)
            lw += qf * math.log(p if p > 0 else 1e-15)
        log_w[docid] = lw

    # Normalise weights to sum to 1 (log-sum-exp for numerical stability)
    max_lw  = max(log_w.values())
    weights = {d: math.exp(lw - max_lw) for d, lw in log_w.items()}
    total_w = sum(weights.values())
    if total_w > 0:
        weights = {d: w / total_w for d, w in weights.items()}

    # ------------------------------------------------------------------
    # Step 3: Build the relevance model P(w|R)
    #
    # For each term w in the vocabulary of the pseudo-relevant documents,
    # P(w|R) is the document-weight-averaged probability of w:
    #   P(w|R) = sum_{D in top-k}  weights[D] * P(w|D)
    #
    # Only terms that actually appear in at least one top-k document are
    # included; terms outside this vocabulary have negligible RM probability
    # when LAM is small (the collection-level contribution from smoothing
    # is uniform and does not help discriminate between topics).
    # ------------------------------------------------------------------
    vocab = set()
    for docid in top_docs:
        vocab.update(collection[docid].terms.keys())

    rm = {}
    for term in vocab:
        p_w_r = 0.0
        for docid in top_docs:
            doc  = collection[docid]
            dl   = doc.get_doc_size()
            tf_d = doc.terms.get(term, 0)
            cf_t = coll_freq.get(term, 0)
            # JM smoothed P(w|D) — same formula as query likelihood above
            p_w_d = ((1 - LAM) * (tf_d / dl if dl > 0 else 0.0)
                     + LAM * (cf_t / filt_size if filt_size > 0 else 0.0))
            p_w_r += weights[docid] * p_w_d
        if p_w_r > 0:
            rm[term] = p_w_r

    # Normalise the relevance model to a proper probability distribution
    total_rm = sum(rm.values())
    if total_rm > 0:
        rm = {t: v / total_rm for t, v in rm.items()}

    # ------------------------------------------------------------------
    # Step 4: Truncate to the top TOP_TERMS terms
    #
    # Low-probability terms in the RM are often generic words that received
    # a non-zero probability only through the smoothing term (LAM * cf/|C|).
    # Keeping only the TOP_TERMS highest-probability terms removes this noise
    # and makes scoring faster. The RM is renormalised after truncation.
    # ------------------------------------------------------------------
    if len(rm) > TOP_TERMS:
        rm = dict(sorted(rm.items(), key=lambda x: x[1], reverse=True)[:TOP_TERMS])
        total_rm = sum(rm.values())
        if total_rm > 0:
            rm = {t: v / total_rm for t, v in rm.items()}

    # ------------------------------------------------------------------
    # Step 5: Score every document by log-likelihood under the relevance model
    #
    # score(d) = sum_w  P(w|R) * log P(w|d)
    #
    # This is the negative KL-divergence -KL(R||d) up to a constant
    # (the entropy of R, which is the same for all documents and does not
    # affect ranking). Documents whose language model P(w|d) is closest
    # to the relevance model P(w|R) receive the highest score.
    #
    # P(w|d) uses the same JM smoothing as above so that terms absent
    # from document d still contribute via the collection model.
    # ------------------------------------------------------------------
    scores = {}
    for docid, doc in collection.items():
        dl    = doc.get_doc_size()
        score = 0.0
        for term, p_w_r in rm.items():
            tf_d  = doc.terms.get(term, 0)
            cf_t  = coll_freq.get(term, 0)
            p_w_d = ((1 - LAM) * (tf_d / dl if dl > 0 else 0.0)
                     + LAM * (cf_t / filt_size if filt_size > 0 else 0.0))
            if p_w_d > 0:
                score += p_w_r * math.log(p_w_d)
        scores[docid] = score
    return scores


# =============================================================================
# Main – run Model_C for all datasets
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
            print(f"[SKIP] Dataset folder not found: {dataset_folder}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {topic_id} – \"{topic_title}\"")
        print(f"{'='*60}")

        coll      = docParser(stop_wordList, dataset_folder)
        inv_index = build_inv_index(coll)
        query_tf  = queryParser(topic_title, stop_wordList)
        cf        = collection_term_freq(coll)
        fs        = collection_size(coll)

        mc_scores = model_c(query_tf, coll, inv_index, avg_len(coll), cf, fs)
        mc_path   = os.path.join(OUTPUT_DIR, f"ModelC_{topic_id}_Ranking.dat")
        mc_ranked = save_ranking(mc_path, mc_scores, "ModelC_Score", topic_title)

        print(f"\n{topic_id} Model_C Top 10 (Doc_ID ModelC_Score):")
        for doc_id, score in mc_ranked[:10]:
            print(f"  {doc_id}  {score:.6f}")
        print(f"\n[SAVED] {mc_path}")


if __name__ == "__main__":
    main()
