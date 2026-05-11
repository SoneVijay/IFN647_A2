"""
IFN647 Assignment 2 - Task 3: Main Runner
Runs Baseline1 (BM25), Baseline2 (JM), and Model_C (PRF+BM25) for all datasets.
"""

import os
from utils import (
    EXTRA_STOPWORDS, DOC_DIR, OUTPUT_DIR,
    load_stopwords, docParser, parse_topics,
    df, save_ranking,
)
from Task1 import bm_25, jm_smoothing
from Task2 import model_c

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_all_models(stopword_file=None):
    if stopword_file is None:
        stopword_file = os.path.join(BASE_DIR, "common-english-words.txt")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    stop_wordList = load_stopwords(stopword_file)
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

        b1_scores = bm_25(coll, topic_title, df_dict, stop_wordList)
        b1_path   = os.path.join(OUTPUT_DIR, f"Baseline1_{topic_id}_Ranking.dat")
        b1_ranked = save_ranking(b1_path, b1_scores, "BM25_Score", topic_title)
        print(f"\n{topic_id} Baseline1 Top 10 (Doc_ID BM25_Score):")
        for doc_id, score in b1_ranked[:10]:
            print(f"  {doc_id}  {score}")

        b2_scores = jm_smoothing(coll, topic_title, stop_wordList, lambda_=0.3)
        b2_path   = os.path.join(OUTPUT_DIR, f"Baseline2_{topic_id}_Ranking.dat")
        b2_ranked = save_ranking(b2_path, b2_scores, "JM_Score", topic_title)
        print(f"\n{topic_id} Baseline2 Top 10 (Doc_ID JM_Score):")
        for doc_id, score in b2_ranked[:10]:
            print(f"  {doc_id}  {score}")

        mc_scores = model_c(coll, topic_title, df_dict, stop_wordList,
                            top_k=5, top_m=10, alpha=1)
        mc_path   = os.path.join(OUTPUT_DIR, f"ModelC_{topic_id}_Ranking.dat")
        mc_ranked = save_ranking(mc_path, mc_scores, "ModelC_Score", topic_title)
        print(f"\n{topic_id} Model_C Top 10 (Doc_ID ModelC_Score):")
        for doc_id, score in mc_ranked[:10]:
            print(f"  {doc_id}  {score}")

        print(f"\n[SAVED] {b1_path}")
        print(f"[SAVED] {b2_path}")
        print(f"[SAVED] {mc_path}")


if __name__ == "__main__":
    run_all_models()
