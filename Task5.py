"""
IFN647 Assignment 2 - Task 5: Statistical Significance Testing
Paired two-tailed t-test comparing Model_C against each baseline.
"""

import scipy.stats as stats
from Task4 import evaluate_all_models


def significance_test(results, measure="AP"):
    """
    Paired t-test: Model_C vs Baseline1 and Model_C vs Baseline2.
    H0: no significant difference.  Threshold: p < 0.05.
    """
    all_topics = sorted(results["Baseline1"].keys())

    def get_scores(model):
        return [results[model].get(t, {}).get(measure, 0.0) for t in all_topics]

    mc_scores = get_scores("ModelC")
    b1_scores = get_scores("Baseline1")
    b2_scores = get_scores("Baseline2")

    t1, p1 = stats.ttest_rel(mc_scores, b1_scores)
    t2, p2 = stats.ttest_rel(mc_scores, b2_scores)

    print(f"\n{'='*60}")
    print(f"Task 5 - Significance Test (Paired t-test) | Measure: {measure}")
    print(f"{'='*60}")
    print(f"  Model_C vs Baseline1: t = {t1:.4f},  p = {p1:.4f}",
          "--> SIGNIFICANT" if p1 < 0.05 else "--> not significant")
    print(f"  Model_C vs Baseline2: t = {t2:.4f},  p = {p2:.4f}",
          "--> SIGNIFICANT" if p2 < 0.05 else "--> not significant")
    print(f"\n  Significance threshold: p < 0.05")
    print(f"  Model_C   mean {measure}: {sum(mc_scores)/len(mc_scores):.3f}")
    print(f"  Baseline1 mean {measure}: {sum(b1_scores)/len(b1_scores):.3f}")
    print(f"  Baseline2 mean {measure}: {sum(b2_scores)/len(b2_scores):.3f}")


if __name__ == "__main__":
    results = evaluate_all_models()
    for measure in ["AP", "P@10", "DCG10"]:
        significance_test(results, measure=measure)
