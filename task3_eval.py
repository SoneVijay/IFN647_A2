"""
Shared summary writer for Task3 runs.
Writes method description + Task4 evaluation tables to Task3Test/.
"""

import os
from utils import BASE_DIR, REL_DIR, parse_topics
from Task4 import load_relevance_judgements, average_precision, precision_at_10, dcg_at_10

TASK3_TEST_DIR = os.path.join(BASE_DIR, "Task3Test")


def _load_all_relevance():
    topics = parse_topics()
    rel = {}
    for topic_id in topics:
        num  = topic_id.replace("R", "")
        path = os.path.join(REL_DIR, f"Dataset{num}.txt")
        if os.path.exists(path):
            rel[topic_id] = load_relevance_judgements(path)
    return rel


def save_task3_summary(method_name, description, b1_all, b2_all, mc_all):
    """
    Write Task3Test/<method_name>_summary.txt with method description + Task4 tables.

    b1_all, b2_all, mc_all: {topic_id: {doc_id: score}}
    """
    os.makedirs(TASK3_TEST_DIR, exist_ok=True)
    out_path = os.path.join(TASK3_TEST_DIR, f"{method_name}_summary.txt")

    relevant_sets = _load_all_relevance()
    all_topics    = sorted(set(b1_all) & set(b2_all) & set(mc_all) & set(relevant_sets))
    n             = len(all_topics)

    def _eval(scores_all):
        ap, p10, dcg = {}, {}, {}
        for tid in all_topics:
            rel    = relevant_sets.get(tid, set())
            ranked = [d for d, _ in sorted(scores_all[tid].items(),
                                           key=lambda x: x[1], reverse=True)]
            ap[tid]  = average_precision(ranked, rel)
            p10[tid] = precision_at_10(ranked, rel)
            dcg[tid] = dcg_at_10(ranked, rel)
        return ap, p10, dcg

    b1_ap, b1_p10, b1_dcg = _eval(b1_all)
    b2_ap, b2_p10, b2_dcg = _eval(b2_all)
    mc_ap, mc_p10, mc_dcg = _eval(mc_all)

    W = 65

    def table(title, b1v, b2v, mcv, row_label, avg_label):
        rows = ["", "=" * W, title,
                f"{'Topic':<10} {'Baseline1':>12} {'Baseline2':>12} {'Model_C':>12}",
                "-" * W]
        for tid in all_topics:
            rows.append(f"{tid:<10} {b1v[tid]:>12.3f} {b2v[tid]:>12.3f} {mcv[tid]:>12.3f}")
        rows.append("-" * W)
        avg_b1 = sum(b1v.values()) / n if n else 0
        avg_b2 = sum(b2v.values()) / n if n else 0
        avg_mc = sum(mcv.values()) / n if n else 0
        rows.append(f"{avg_label:<10} {avg_b1:>12.3f} {avg_b2:>12.3f} {avg_mc:>12.3f}")
        return rows

    lines = [f"Method: {method_name}", "=" * W, description.strip()]
    lines += table("Table 1. Performance of 3 models on Average Precision (AP)",
                   b1_ap, b2_ap, mc_ap, "AP", "MAP")
    lines += table("Table 2. Performance of 3 models on Precision@10",
                   b1_p10, b2_p10, mc_p10, "P@10", "Average")
    lines += table("Table 3. Performance of 3 models on DCG@10",
                   b1_dcg, b2_dcg, mc_dcg, "DCG@10", "Average")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[Task3Test] Summary saved: {out_path}")
    return out_path
