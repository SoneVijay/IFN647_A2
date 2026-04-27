"""
IFN647 Assignment 2 - Task 2: Custom Model (Model_C)
Pseudo-Relevance Feedback (PRF) with BM25 re-ranking.
Also includes parameter validation via grid search (Section 8 of A2.py).
(Sections 4 & 8 of A2.py)
"""

import os
import math
from utils import (
    EXTRA_STOPWORDS, DOC_DIR, OUTPUT_DIR, REL_DIR,
    load_stopwords, docParser, queryParser, parse_topics,
    df, avg_len, save_ranking,
)
from Task1 import bm_25
from Task4 import load_relevance_judgements, average_precision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# Model C – PRF + BM25 re-ranking
# =============================================================================
# Step 1: Initial BM25 retrieval with the original query.
# Step 2: Take top_k docs as pseudo-relevant set.
# Step 3: Compute Rocchio-style term weights from pseudo-relevant docs:
#           w(t) = Σ_{D∈PRF} (f_t,D / |D|)  (excludes original query terms)
# Step 4: Add top_m expansion terms to the query with weight alpha.
# Step 5: Re-rank all docs using BM25 with the expanded query.

def model_c(coll, q, df_dict, stop_wordList, top_k=5, top_m=10, alpha=1):
    # Step 1
    initial_scores = bm_25(coll, q, df_dict, stop_wordList)
    ranked_initial = sorted(initial_scores.items(), key=lambda x: x[1], reverse=True)

    # Step 2
    pseudo_relevant_ids = [docid for docid, _ in ranked_initial[:top_k]]

    # Step 3
    original_terms = set(queryParser(q, stop_wordList).keys())
    term_weights   = {}
    for docid in pseudo_relevant_ids:
        doc = coll[docid]
        n   = doc.get_doc_size()
        if n == 0:
            continue
        for term, freq in doc.terms.items():
            if term not in original_terms:
                term_weights[term] = term_weights.get(term, 0) + (freq / n)

    # Step 4
    top_expansion  = sorted(term_weights.items(), key=lambda x: x[1], reverse=True)[:top_m]
    expanded_query = queryParser(q, stop_wordList)
    for term, _ in top_expansion:
        expanded_query[term] = expanded_query.get(term, 0) + alpha

    # Step 5 – re-rank via BM25 with expanded query (inline to accept a dict directly)
    k1, k2, b = 1.2, 500, 0.75
    N    = len(coll)
    avdl = avg_len(coll)
    scores = {}

    for docid, doc in coll.items():
        score = 0.0
        dl    = doc.get_doc_size()
        K     = k1 * ((1 - b) + b * (dl / avdl))

        for term, qf_t in expanded_query.items():
            f_t = doc.terms.get(term, 0)
            if f_t == 0:
                continue
            n_t = df_dict.get(term, 0)
            if n_t == 0:
                continue
            idf       = math.log((N - n_t + 0.5) / (n_t + 0.5))
            tf_weight = ((k1 + 1) * f_t) / (K + f_t)
            qf_weight = ((k2 + 1) * qf_t) / (k2 + qf_t)
            score    += idf * tf_weight * qf_weight

        scores[docid] = score

    return scores


# =============================================================================
# Parameter validation – grid search over top_k and top_m
# =============================================================================

def validate_model_c_parameters(
    stopword_file=None,
    top_k_values=(3, 5, 10),
    top_m_values=(5, 10, 15, 20),
):
    if stopword_file is None:
        stopword_file = os.path.join(BASE_DIR, "common-english-words.txt")

    stop_wordList = load_stopwords(stopword_file)
    stop_wordList = list(set(stop_wordList + EXTRA_STOPWORDS))
    topics        = parse_topics()

    best_map    = -1
    best_params = {}

    print("\nModel_C Parameter Validation (Grid Search)")
    print(f"{'top_k':>8} {'top_m':>8} {'MAP':>10}")
    print("-" * 30)

    for top_k in top_k_values:
        for top_m in top_m_values:
            ap_list = []

            for topic_id, topic_title in sorted(topics.items()):
                dataset_number = topic_id.replace("R", "")
                dataset_folder = os.path.join(DOC_DIR, f"Dataset{dataset_number}")
                judg_file      = os.path.join(REL_DIR, f"Dataset{dataset_number}.txt")

                if not os.path.exists(dataset_folder) or not os.path.exists(judg_file):
                    continue

                coll         = docParser(stop_wordList, dataset_folder)
                df_dict      = df(coll)
                relevant_set = load_relevance_judgements(judg_file)

                mc_scores = model_c(coll, topic_title, df_dict, stop_wordList,
                                    top_k=top_k, top_m=top_m, alpha=1)
                ranked    = sorted(mc_scores.items(), key=lambda x: x[1], reverse=True)
                ap_list.append(average_precision([d for d, _ in ranked], relevant_set))

            map_score = sum(ap_list) / len(ap_list) if ap_list else 0.0
            print(f"{top_k:>8} {top_m:>8} {map_score:>10.4f}")

            if map_score > best_map:
                best_map    = map_score
                best_params = {"top_k": top_k, "top_m": top_m}

    print(f"\nBest Parameters: top_k={best_params['top_k']}, "
          f"top_m={best_params['top_m']} -> MAP={best_map:.4f}")
    return best_params


# =============================================================================
# Main – run Model_C for all datasets
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
        df_dict = df(coll)

        mc_scores = model_c(coll, topic_title, df_dict, stop_wordList,
                            top_k=5, top_m=10, alpha=1)
        mc_path   = os.path.join(OUTPUT_DIR, f"ModelC_{topic_id}_Ranking.dat")
        mc_ranked = save_ranking(mc_path, mc_scores, "ModelC_Score", topic_title)

        print(f"\n{topic_id} Model_C Top 10 (Doc_ID ModelC_Score):")
        for doc_id, score in mc_ranked[:10]:
            print(f"  {doc_id}  {score}")
        print(f"\n[SAVED] {mc_path}")


if __name__ == "__main__":
    main()
    # Uncomment to run parameter grid search:
    validate_model_c_parameters()
