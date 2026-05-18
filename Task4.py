"""
IFN647 Assignment 2 - Task 4: Evaluation
Computes AP, P@10, and DCG@10 for all three models across all topics.
Models: Baseline1 (BM25), Baseline2 (JM), ModelC (KL + gap-detected K).
"""

import os
import math
from utils import OUTPUT_DIR, REL_DIR, parse_topics


# =============================================================================
# Relevance loading and ranking loading
# =============================================================================

def load_relevance_judgements(judg_file):
    """Returns a set of relevant doc_ids (label == 1)."""
    relevant = set()
    with open(judg_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[2] == "1":
                relevant.add(parts[1])
    return relevant


def load_ranking_from_file(ranking_file):
    """Returns a list of doc_ids in ranked order (skips header lines)."""
    ranked_docs = []
    with open(ranking_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].isdigit():
                ranked_docs.append(parts[0])
    return ranked_docs


# =============================================================================
# Evaluation metrics
# =============================================================================

def average_precision(ranked_docs, relevant_set):
    """AP = (1/|R|) * Σ_{k: doc_k relevant} Precision@k"""
    if not relevant_set:
        return 0.0
    hits          = 0
    precision_sum = 0.0
    for k, doc_id in enumerate(ranked_docs, start=1):
        if doc_id in relevant_set:
            hits          += 1
            precision_sum += hits / k
    return precision_sum / len(relevant_set)


def precision_at_10(ranked_docs, relevant_set):
    """P@10 = (# relevant in top 10) / 10"""
    return sum(1 for doc in ranked_docs[:10] if doc in relevant_set) / 10


def dcg_at_10(ranked_docs, relevant_set):
    """DCG10 = rel_1 + Σ_{i=2}^{10} rel_i / log2(i)"""
    dcg = 0.0
    for i, doc_id in enumerate(ranked_docs[:10], start=1):
        rel  = 1 if doc_id in relevant_set else 0
        dcg += rel if i == 1 else rel / math.log2(i)
    return dcg


# =============================================================================
# Evaluate all models
# =============================================================================

def evaluate_all_models(
    output_folder=OUTPUT_DIR,
    relevance_folder=REL_DIR,
):
    topics  = parse_topics()
    models  = ["Baseline1", "Baseline2", "ModelC"]
    results = {model: {} for model in models}

    for topic_id in sorted(topics.keys()):
        dataset_number = topic_id.replace("R", "")
        judg_file = os.path.join(relevance_folder, f"Dataset{dataset_number}.txt")

        if not os.path.exists(judg_file):
            print(f"[SKIP] Judgement file not found: {judg_file}")
            continue

        relevant_set = load_relevance_judgements(judg_file)

        for model in models:
            ranking_file = os.path.join(
                output_folder, f"{model}_{topic_id}_Ranking.dat"
            )
            if not os.path.exists(ranking_file):
                print(f"[SKIP] Ranking file not found: {ranking_file}")
                results[model][topic_id] = {"AP": 0.0, "P@10": 0.0, "DCG10": 0.0}
                continue

            ranked_docs = load_ranking_from_file(ranking_file)
            results[model][topic_id] = {
                "AP":    average_precision(ranked_docs, relevant_set),
                "P@10":  precision_at_10(ranked_docs, relevant_set),
                "DCG10": dcg_at_10(ranked_docs, relevant_set),
            }

    all_topics = sorted(results["Baseline1"].keys())
    n          = len(all_topics)
    W   = 58  # table width for 3 model columns
    HDR = f"{'Topic':<10} {'Baseline1':>12} {'Baseline2':>12} {'Model_C':>12}"

    def _print_table(title, metric):
        print("\n" + "=" * W)
        print(title)
        print(HDR)
        print("-" * W)
        sums = {m: 0.0 for m in models}
        for topic_id in all_topics:
            row = f"{topic_id:<10}"
            for model in models:
                val = results[model].get(topic_id, {}).get(metric, 0.0)
                sums[model] += val
                row += f"{val:>12.3f}"
            print(row)
        print("-" * W)
        label  = "MAP" if metric == "AP" else "Average"
        footer = f"{label:<10}"
        for model in models:
            footer += f"{sums[model]/n if n > 0 else 0:>12.3f}"
        print(footer)

    _print_table("Table 1. Performance of 3 models on Average Precision (AP)", "AP")
    _print_table("Table 2. Performance of 3 models on Precision@10",           "P@10")
    _print_table("Table 3. Performance of 3 models on DCG@10",                 "DCG10")

    return results


if __name__ == "__main__":
    evaluate_all_models()
