"""
IFN647 Assignment 2 - Task 5: Statistical Significance Testing
One-tailed paired t-test comparing Model_C against each baseline.
H1: Model_C > baseline.  Threshold: p < 0.05.
"""

import scipy.stats as stats
from Task4 import evaluate_all_models


def significance_test(results, measure="AP"):
    all_topics = sorted(results["Baseline1"].keys())

    def get_scores(model):
        return [results[model].get(t, {}).get(measure, 0.0) for t in all_topics]

    b1_scores = get_scores("Baseline1")
    b2_scores = get_scores("Baseline2")
    mc_scores = get_scores("ModelC")

    def _row(label, s_a, s_b):
        t, p = stats.ttest_rel(s_a, s_b, alternative="greater")
        sig  = "--> SIGNIFICANT" if p < 0.05 else "--> not significant"
        print(f"  {label}: t = {t:+.4f},  p = {p:.4f}  {sig}")

    print(f"\n{'='*65}")
    print(f"Task 5 - Significance Test (one-tailed paired t-test) | {measure}")
    print(f"{'='*65}")

    _row("Model_C vs Baseline1", mc_scores, b1_scores)
    _row("Model_C vs Baseline2", mc_scores, b2_scores)

    print(f"\n  Significance threshold: p < 0.05 (one-tailed, H1: Model_C > baseline)")
    print(f"  Baseline1 mean {measure}: {sum(b1_scores)/len(b1_scores):.3f}")
    print(f"  Baseline2 mean {measure}: {sum(b2_scores)/len(b2_scores):.3f}")
    print(f"  Model_C   mean {measure}: {sum(mc_scores)/len(mc_scores):.3f}")


if __name__ == "__main__":
    results = evaluate_all_models()
    for measure in ["AP", "P@10", "DCG10"]:
        significance_test(results, measure=measure)
