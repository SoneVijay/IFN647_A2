"""
IFN647 Assignment 2 - Grid Search for Task 2 Model C Parameters (unsupervised).

Selects PRF_PCT and LAM for the KL-Divergence Relevance Model
WITHOUT accessing relevance judgement files.

Evaluation metric (RM IDF Quality — fully unsupervised):
  For each parameter combination and each topic, build the relevance model R
  and compute its weighted average IDF:

    IDF_quality(R) = sum_w  P(w|R) * log(N / df(w))

  where N = collection size, df(w) = document frequency of term w.

  A higher IDF quality means the RM concentrates probability mass on
  topic-discriminative terms rather than common collection-wide terms.
  This is desirable: query expansion with high-IDF terms produces more
  focused, precise rankings than expansion with common terms.

  Mean IDF quality across all topics is used to rank parameter combinations.

  No relevance judgement files are read.

Grid:
  prf_k [5, 10, 15, 20]
  lam   [0.05, 0.1, 0.3, 0.5]

Total: 4 x 4 = 16 combinations.
All vocabulary terms from the pseudo-relevant documents are always used.
"""

import os
import math
from itertools import product

from utils import (
    EXTRA_STOPWORDS, DOC_DIR,
    load_stopwords, docParser, queryParser, parse_topics,
    avg_len, build_inv_index, collection_term_freq, collection_size,
)
from Task1 import bm_25

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GRID = {
    'prf_k': [5, 10, 15, 20],
    'lam':   [0.05, 0.1, 0.3, 0.5],
}


# =============================================================================
# Relevance model builder
# =============================================================================

def _build_rm(query_tf, ranked_bm25, collection, coll_freq, filt_size,
              prf_k, lam):
    """Build and return the normalised relevance model over the top-k doc vocabulary."""
    top_docs = ranked_bm25[:prf_k]
    if not top_docs:
        return {}

    # Document weights: log P(D|q) proportional to sum_q tf(q,Q)*log P(q|D)
    log_w = {}
    for docid in top_docs:
        doc = collection[docid]
        dl  = doc.get_doc_size()
        lw  = 0.0
        for term, qf in query_tf.items():
            tf_d = doc.terms.get(term, 0)
            cf_t = coll_freq.get(term, 0)
            p = ((1 - lam) * (tf_d / dl if dl > 0 else 0.0)
                 + lam * (cf_t / filt_size if filt_size > 0 else 0.0))
            lw += qf * math.log(p if p > 0 else 1e-15)
        log_w[docid] = lw

    max_lw  = max(log_w.values())
    weights = {d: math.exp(lw - max_lw) for d, lw in log_w.items()}
    total_w = sum(weights.values())
    if total_w > 0:
        weights = {d: w / total_w for d, w in weights.items()}

    # Relevance model over top-k document vocabulary
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
            p_w_d = ((1 - lam) * (tf_d / dl if dl > 0 else 0.0)
                     + lam * (cf_t / filt_size if filt_size > 0 else 0.0))
            p_w_r += weights[docid] * p_w_d
        if p_w_r > 0:
            rm[term] = p_w_r

    total_rm = sum(rm.values())
    if total_rm > 0:
        rm = {t: v / total_rm for t, v in rm.items()}

    return rm


# =============================================================================
# IDF quality metric (unsupervised)
# =============================================================================

def _idf_quality(rm, inv_index, N):
    """Weighted average IDF of the relevance model terms.

    Measures how discriminative the RM's vocabulary is.
    Higher = RM focuses on topic-specific terms (good for retrieval).
    Lower  = RM is dominated by common collection terms (poor expansion).
    """
    score = 0.0
    for term, p_rm in rm.items():
        df_t = len(inv_index.get(term, {}))
        if df_t > 0:
            score += p_rm * math.log(N / df_t)
    return score


# =============================================================================
# Grid search
# =============================================================================

def grid_search(stopword_file=None):
    if stopword_file is None:
        stopword_file = os.path.join(BASE_DIR, "common-english-words.txt")

    stop_wordList = list(set(load_stopwords(stopword_file) + EXTRA_STOPWORDS))
    topics        = parse_topics()

    # ------------------------------------------------------------------
    # Precompute per-topic data (once)
    # ------------------------------------------------------------------
    print("Precomputing BM25 rankings and collection statistics ...")
    precomp = {}

    for topic_id, topic_title in sorted(topics.items()):
        num    = topic_id.replace("R", "")
        folder = os.path.join(DOC_DIR, f"Dataset{num}")
        if not os.path.exists(folder):
            continue

        coll      = docParser(stop_wordList, folder)
        inv_index = build_inv_index(coll)
        query_tf  = queryParser(topic_title, stop_wordList)
        cf        = collection_term_freq(coll)
        fs        = collection_size(coll)
        avdl      = avg_len(coll)
        N         = len(coll)

        bm25        = bm_25(query_tf, coll, inv_index, avdl)
        ranked_bm25 = sorted(bm25, key=bm25.get, reverse=True)

        precomp[topic_id] = {
            'query_tf':    query_tf,
            'collection':  coll,
            'inv_index':   inv_index,
            'coll_freq':   cf,
            'filt_size':   fs,
            'ranked_bm25': ranked_bm25,
            'N':           N,
        }
        print(f"  {topic_id}")

    # ------------------------------------------------------------------
    # Grid search over (prf_k, lam)
    # ------------------------------------------------------------------
    combos = list(product(GRID['prf_k'], GRID['lam']))
    total  = len(combos)
    print(f"\nSearching {total} combinations "
          f"(prf_k x lam = "
          f"{len(GRID['prf_k'])}x{len(GRID['lam'])}) ...\n")

    hdr = f"{'prf_k':>6} {'lam':>5}  {'IDF quality':>11}"
    print(hdr)
    print("-" * len(hdr))

    best_params = None
    best_idf    = -1.0

    for i, (prf_k, lam) in enumerate(combos, 1):
        idf_scores = []
        for pc in precomp.values():
            rm = _build_rm(
                pc['query_tf'], pc['ranked_bm25'], pc['collection'],
                pc['coll_freq'], pc['filt_size'],
                prf_k, lam,
            )
            if rm:
                idf_scores.append(_idf_quality(rm, pc['inv_index'], pc['N']))

        mean_idf = sum(idf_scores) / len(idf_scores) if idf_scores else 0.0
        marker   = "  <-- best" if mean_idf > best_idf else ""

        print(f"{prf_k:>6} {lam:>5.2f}  {mean_idf:>11.4f}  [{i}/{total}]{marker}")

        if mean_idf > best_idf:
            best_idf    = mean_idf
            best_params = dict(prf_k=prf_k, lam=lam, idf_quality=mean_idf)

    # ------------------------------------------------------------------
    # Summary: per-parameter sensitivity tables + best result
    # ------------------------------------------------------------------
    best_lam = best_params['lam']
    best_k   = best_params['prf_k']

    # Lambda sensitivity (hold prf_k=best)
    lam_idf = {}
    print(f"\n--- Lambda sensitivity (prf_k={best_k}) ---")
    for lam in sorted(GRID['lam']):
        idfs = []
        for pc in precomp.values():
            rm = _build_rm(pc['query_tf'], pc['ranked_bm25'], pc['collection'],
                           pc['coll_freq'], pc['filt_size'], best_k, lam)
            if rm:
                idfs.append(_idf_quality(rm, pc['inv_index'], pc['N']))
        v = sum(idfs) / len(idfs) if idfs else 0.0
        lam_idf[lam] = v
        marker = "  <-- best" if lam == best_lam else ""
        print(f"  lam={lam:.2f}  IDF quality={v:.4f}{marker}")

    # PRF_K sensitivity (hold lam=best)
    prf_idf = {}
    print(f"\n--- PRF_K sensitivity (lam={best_lam:.2f}) ---")
    for k in sorted(GRID['prf_k']):
        idfs = []
        for pc in precomp.values():
            rm = _build_rm(pc['query_tf'], pc['ranked_bm25'], pc['collection'],
                           pc['coll_freq'], pc['filt_size'], k, best_lam)
            if rm:
                idfs.append(_idf_quality(rm, pc['inv_index'], pc['N']))
        v = sum(idfs) / len(idfs) if idfs else 0.0
        prf_idf[k] = v
        marker = "  <-- best" if k == best_k else ""
        print(f"  prf_k={k:>2}  IDF quality={v:.4f}{marker}")

    lam_range = max(lam_idf.values()) - min(lam_idf.values())
    prf_range = max(prf_idf.values()) - min(prf_idf.values())

    print("\n" + "=" * 60)
    print("Best parameters found (no relevance judgements used):")
    print(f"  PRF_K = {best_params['prf_k']}")
    print(f"  LAM   = {best_params['lam']}")
    print(f"  IDF quality (mean across topics) = {best_params['idf_quality']:.4f}")
    print()
    print(f"  Lambda effect: IDF quality range = {lam_range:.4f} across tested values")
    print(f"  PRF_K effect:  IDF quality range = {prf_range:.4f} across tested values")
    print("=" * 60)

    return best_params


if __name__ == "__main__":
    grid_search()
