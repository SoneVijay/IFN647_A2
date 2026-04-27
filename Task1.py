"""
IFN647 Assignment 2 - Task 1: Baseline Models
Baseline 1: BM25  (k1=1.2, k2=500, b=0.75)
Baseline 2: Jelinek-Mercer smoothing language model  (lambda=0.3)
(Section 3 of A2.py)
"""

from collections import defaultdict
import os
import math
from utils import (
    EXTRA_STOPWORDS, DOC_DIR, OUTPUT_DIR, TOPICS_FILE,
    load_stopwords, docParser, queryParser, parse_topics,
    df, avg_len, collection_term_freq, collection_size, save_ranking,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# BM25 parameters (Equation 1)
K1 = 1.2
K2 = 500
B  = 0.75

# JM smoothing parameter (Equation 2)
LAMBDA = 0.3

# =============================================================================
# Baseline 1 – BM25
# =============================================================================
# BM25(D,Q) = Σ_{t∈Q} log((N-nt+0.5)/(nt+0.5))
#             * (k1+1)*ft/(K+ft)  *  (k2+1)*qft/(k2+qft)
# K = k1*((1-b) + b*dl/avdl)

def bm_25(query_tf, collection, inv_index, avdl):
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
# Baseline 2 – JM smoothing language model
# =============================================================================
# JM(D,Q) = Σ_{qj∈Q} log((1-λ)*(fqj,D/|D|) + λ*(cqj/|Dataset_i|))

def jm_smoothing(query_tf, collection, coll_freq, filt_size):
    coll_probs = {
        term: coll_freq[term] / filt_size
        for term in query_tf
        if coll_freq.get(term, 0) > 0 and filt_size > 0
    }

    scores = {}
    for docid, doc in collection.items():
        score    = 0.0
        filt_len = doc.get_doc_size()
        for term, coll_p in coll_probs.items():
            fqj_d    = doc.terms.get(term, 0)
            doc_p    = fqj_d / filt_len if filt_len > 0 else 0.0
            smoothed = (1 - LAMBDA) * doc_p + LAMBDA * coll_p
            if smoothed > 0:
                score += math.log10(smoothed)
        scores[docid] = score
    return scores


# =============================================================================
# Main – run Baseline1 and Baseline2 for all datasets
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    stop_wordList = load_stopwords(
        os.path.join(BASE_DIR, "common-english-words.txt")
    )
    stop_wordList = list(set(stop_wordList + EXTRA_STOPWORDS))

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

        coll    = docParser(stop_wordList, dataset_folder)
        query_tf = queryParser(topic_title, stop_wordList)
        inv_index = defaultdict(dict)
        for docid, doc in coll.items():
            for term, freq in doc.terms.items():
                inv_index[term][docid] = freq
        df_dict = df(coll)

        b1_scores = bm_25(query_tf, coll, inv_index, avg_len(coll))
        b1_path   = os.path.join(OUTPUT_DIR, f"Baseline1_{topic_id}_Ranking.dat")
        b1_ranked = save_ranking(b1_path, b1_scores, "BM25_Score", topic_title)
        print(f"\n{topic_id} Baseline1 Top 10 (Doc_ID BM25_Score):")
        for doc_id, score in b1_ranked[:10]:
            print(f"  {doc_id}  {score}")

        b2_scores = jm_smoothing(query_tf, coll, df_dict, sum(len(doc.terms) for doc in coll.values()))
        b2_path   = os.path.join(OUTPUT_DIR, f"Baseline2_{topic_id}_Ranking.dat")
        b2_ranked = save_ranking(b2_path, b2_scores, "JM_Score", topic_title)
        print(f"\n{topic_id} Baseline2 Top 10 (Doc_ID JM_Score):")
        for doc_id, score in b2_ranked[:10]:
            print(f"  {doc_id}  {score}")

        print(f"\n[SAVED] {b1_path}")
        print(f"[SAVED] {b2_path}")


if __name__ == "__main__":
    main()
