"""
IFN647 Assignment 2 - Grid Search for Task2 Parameters (unsupervised).

Selects MAX_K, LAM, and ALPHA for the gap-based KL model with RM3.
MIN_K is fixed at 5.

Gap detection: largest BM25 score drop within [MIN_K, MAX_K) positions.
RM3: rm3 = alpha * P(w|R) + (1-alpha) * P(w|Q)

Evaluation metric: RM3 IDF Quality (fully unsupervised)
  IDF_quality = sum_w  rm3(w) * log(N / df(w))

Grid:
  max_k [10, 20, 30, 50]
  lam   [0.05, 0.1, 0.3, 0.5]
  alpha [0.1, 0.3, 0.5, 0.7]

Total: 4 x 4 x 4 = 64 combinations.
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
MIN_K    = 5

GRID = {
    'max_k': [10, 20, 30, 50],
    'lam':   [0.05, 0.1, 0.3, 0.5],
    'alpha': [0.1, 0.3, 0.5, 0.7],
}


def _detect_k(ranked_bm25, bm25_scores, min_k, max_k):
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


def _build_rm3(query_tf, ranked_bm25, collection, coll_freq, filt_size, prf_k, lam, alpha):
    top_docs = ranked_bm25[:prf_k]
    if not top_docs:
        return {}

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
            p_w_d = ((1 - lam) * (tf_d / dl if dl > 0 else 0.0)
                     + lam * (cf_t / filt_size if filt_size > 0 else 0.0))
            p_w_r += weights[docid] * p_w_d
        if p_w_r > 0:
            rm[term] = p_w_r

    total_rm = sum(rm.values())
    if total_rm > 0:
        rm = {t: v / total_rm for t, v in rm.items()}

    total_qf = sum(query_tf.values())
    p_q      = {t: qf / total_qf for t, qf in query_tf.items()}
    rm3      = {}
    for term in set(rm) | set(p_q):
        rm3[term] = alpha * rm.get(term, 0.0) + (1 - alpha) * p_q.get(term, 0.0)
    return rm3


def _idf_quality(rm, inv_index, N):
    score = 0.0
    for term, p_rm in rm.items():
        df_t = len(inv_index.get(term, {}))
        if df_t > 0:
            score += p_rm * math.log(N / df_t)
    return score


def grid_search(stopword_file=None):
    if stopword_file is None:
        stopword_file = os.path.join(BASE_DIR, "common-english-words.txt")

    stop_wordList = list(set(load_stopwords(stopword_file) + EXTRA_STOPWORDS))
    topics        = parse_topics()

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

        bm25_scores = bm_25(query_tf, coll, inv_index, avdl)
        ranked_bm25 = sorted(bm25_scores, key=bm25_scores.get, reverse=True)

        precomp[topic_id] = dict(
            query_tf=query_tf, collection=coll, inv_index=inv_index,
            coll_freq=cf, filt_size=fs, ranked_bm25=ranked_bm25,
            bm25_scores=bm25_scores, N=N,
        )
        print(f"  {topic_id}")

    combos = list(product(GRID['max_k'], GRID['lam'], GRID['alpha']))
    total  = len(combos)
    print(f"\nSearching {total} combinations (min_k={MIN_K} fixed) ...\n")

    hdr = f"{'max_k':>6} {'lam':>5} {'alpha':>6}  {'IDF quality':>11}  {'mean k':>6}"
    print(hdr)
    print("-" * len(hdr))

    best_params = None
    best_idf    = -1.0

    for i, (max_k, lam, alpha) in enumerate(combos, 1):
        idf_scores  = []
        detected_ks = []
        for pc in precomp.values():
            k = _detect_k(pc['ranked_bm25'], pc['bm25_scores'], MIN_K, max_k)
            detected_ks.append(k)
            rm3 = _build_rm3(pc['query_tf'], pc['ranked_bm25'], pc['collection'],
                              pc['coll_freq'], pc['filt_size'], k, lam, alpha)
            if rm3:
                idf_scores.append(_idf_quality(rm3, pc['inv_index'], pc['N']))

        mean_idf = sum(idf_scores) / len(idf_scores) if idf_scores else 0.0
        mean_k   = sum(detected_ks) / len(detected_ks) if detected_ks else 0.0
        marker   = "  <-- best" if mean_idf > best_idf else ""
        print(f"{max_k:>6} {lam:>5.2f} {alpha:>6.1f}  {mean_idf:>11.4f}  {mean_k:>6.1f}  [{i}/{total}]{marker}")

        if mean_idf > best_idf:
            best_idf    = mean_idf
            best_params = dict(max_k=max_k, lam=lam, alpha=alpha,
                               idf_quality=mean_idf, mean_k=mean_k)

    best_lam   = best_params['lam']
    best_maxk  = best_params['max_k']
    best_alpha = best_params['alpha']

    lam_idf = {}
    print(f"\n--- Lambda sensitivity (max_k={best_maxk}, alpha={best_alpha:.1f}) ---")
    for lam in sorted(GRID['lam']):
        idfs = []
        for pc in precomp.values():
            k = _detect_k(pc['ranked_bm25'], pc['bm25_scores'], MIN_K, best_maxk)
            rm3 = _build_rm3(pc['query_tf'], pc['ranked_bm25'], pc['collection'],
                              pc['coll_freq'], pc['filt_size'], k, lam, best_alpha)
            if rm3:
                idfs.append(_idf_quality(rm3, pc['inv_index'], pc['N']))
        v = sum(idfs) / len(idfs) if idfs else 0.0
        lam_idf[lam] = v
        print(f"  lam={lam:.2f}  IDF quality={v:.4f}" + ("  <-- best" if lam == best_lam else ""))

    maxk_idf = {}
    print(f"\n--- MAX_K sensitivity (lam={best_lam:.2f}, alpha={best_alpha:.1f}) ---")
    for max_k in sorted(GRID['max_k']):
        idfs = []
        for pc in precomp.values():
            k = _detect_k(pc['ranked_bm25'], pc['bm25_scores'], MIN_K, max_k)
            rm3 = _build_rm3(pc['query_tf'], pc['ranked_bm25'], pc['collection'],
                              pc['coll_freq'], pc['filt_size'], k, best_lam, best_alpha)
            if rm3:
                idfs.append(_idf_quality(rm3, pc['inv_index'], pc['N']))
        v = sum(idfs) / len(idfs) if idfs else 0.0
        maxk_idf[max_k] = v
        print(f"  max_k={max_k:>2}  IDF quality={v:.4f}" + ("  <-- best" if max_k == best_maxk else ""))

    alpha_idf = {}
    print(f"\n--- Alpha sensitivity (max_k={best_maxk}, lam={best_lam:.2f}) ---")
    for alpha in sorted(GRID['alpha']):
        idfs = []
        for pc in precomp.values():
            k = _detect_k(pc['ranked_bm25'], pc['bm25_scores'], MIN_K, best_maxk)
            rm3 = _build_rm3(pc['query_tf'], pc['ranked_bm25'], pc['collection'],
                              pc['coll_freq'], pc['filt_size'], k, best_lam, alpha)
            if rm3:
                idfs.append(_idf_quality(rm3, pc['inv_index'], pc['N']))
        v = sum(idfs) / len(idfs) if idfs else 0.0
        alpha_idf[alpha] = v
        print(f"  alpha={alpha:.1f}  IDF quality={v:.4f}" + ("  <-- best" if alpha == best_alpha else ""))

    lam_range   = max(lam_idf.values()) - min(lam_idf.values())
    maxk_range  = max(maxk_idf.values()) - min(maxk_idf.values())
    alpha_range = max(alpha_idf.values()) - min(alpha_idf.values())

    print("\n" + "=" * 60)
    print("Best parameters (GapK + RM3, no relevance judgements used):")
    print(f"  MIN_K = {MIN_K}  (fixed)")
    print(f"  MAX_K = {best_params['max_k']}")
    print(f"  LAM   = {best_params['lam']}")
    print(f"  ALPHA = {best_params['alpha']}")
    print(f"  IDF quality (mean) = {best_params['idf_quality']:.4f}")
    print(f"  Mean detected k    = {best_params['mean_k']:.1f}")
    print(f"  Lambda effect: range = {lam_range:.4f}")
    print(f"  MAX_K effect:  range = {maxk_range:.4f}")
    print(f"  Alpha effect:  range = {alpha_range:.4f}")
    print("=" * 60)

    return best_params


if __name__ == "__main__":
    grid_search()
