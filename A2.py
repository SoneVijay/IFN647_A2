# ============================================================
# IFN647 – Assignment 2 | Complete Implementation
# ============================================================

import os
import math
import string
import numpy as np
from stemming.porter2 import stem

# ============================================================
# SECTION 1: Document & Query Parsing
# ============================================================

class Doc:
    """Bag-of-Words document representation."""

    def __init__(self, docID):
        self.docID = docID        # Document identifier (itemid from XML)
        self.terms = {}           # {term: frequency}
        self.doc_size = 0         # Total number of terms after preprocessing

    def add_term(self, term):
        self.doc_size += 1
        self.terms[term] = self.terms.get(term, 0) + 1

    def get_docID(self):
        return self.docID

    def get_termList(self):
        return sorted(self.terms.keys())

    def get_doc_size(self):
        return self.doc_size

    def set_doc_size(self, size):
        self.doc_size = size


def load_stopwords(filepath):
    """Load comma-separated stopwords from file."""
    with open(filepath, 'r') as f:
        return f.read().split(",")


# Additional stopwords found by inspecting high-frequency non-informative terms
EXTRA_STOPWORDS = [
    "one", "over", "day", "five", "month", "up", "between", "three",
    "ec", "gm", "cm", "fm", "sc", "b", "bb", "db", "e", "s", "v",
    "x", "tf", "c", "w", "ag", "gmc", "lx"
]


def parse_doc(file, stop_wordList):
    """
    Parse a single XML news document.
    Extracts text only from <text>...</text> blocks.
    Applies: lowercasing, punctuation/digit removal, stopword removal, Porter2 stemming.
    """
    in_text = False
    doc = None

    for line in file:
        line = line.strip()

        if line.startswith("<newsitem "):
            for part in line.split():
                if part.startswith("itemid="):
                    docid = part.split("=")[1].strip('"')
                    doc = Doc(docid)

        if "<text>" in line:
            in_text = True
            line = line.split("<text>", 1)[1]

        if in_text and doc:
            if "</text>" in line:
                line = line.split("</text>", 1)[0]
                in_text = False

            line = line.replace("<p>", "").replace("</p>", "")
            line = line.translate(
                str.maketrans('', '', string.punctuation + string.digits)
            )

            for word in line.lower().split():
                if word not in stop_wordList:
                    doc.add_term(stem(word))

    return doc


def docParser(stop_wordList, folder):
    """Parse all XML documents in a given dataset folder."""
    collection = {}
    for filename in os.listdir(folder):
        if filename.endswith(".xml"):
            filepath = os.path.join(folder, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                doc = parse_doc(f, stop_wordList)
                if doc:
                    collection[doc.docID] = doc
    return collection


def queryParser(query, stop_wordList):
    """
    Parse and preprocess a query string.
    Returns {term: frequency} after stopword removal and stemming.
    """
    query_terms = {}
    query = query.replace("-", " ")
    query = query.translate(
        str.maketrans('', '', string.punctuation + string.digits)
    )
    for term in query.lower().split():
        if term not in stop_wordList:
            stemmed = stem(term)
            query_terms[stemmed] = query_terms.get(stemmed, 0) + 1
    return query_terms


def parse_topics(topic_file):
    """Parse Topics.txt and return {topic_id: title} dictionary."""
    topics = {}
    with open(topic_file, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    for block in content.split("<Topic>"):
        if "<num>" not in block:
            continue
        num_raw = block.split("<num>", 1)[1]
        for end_tag in ["</num>", "<title>", "<desc>", "<narr>"]:
            if end_tag in num_raw:
                num_raw = num_raw.split(end_tag, 1)[0]
                break
        num_part = num_raw.strip()

        title_raw = block.split("<title>", 1)[1] if "<title>" in block else ""
        for end_tag in ["</title>", "<desc>", "<narr>", "</Topic>"]:
            if end_tag in title_raw:
                title_raw = title_raw.split(end_tag, 1)[0]
                break

        topic_id = num_part[num_part.find("R"):].strip() if "R" in num_part else num_part
        topics[topic_id] = title_raw.strip()

    return topics


# ============================================================
# SECTION 2: Helper Functions
# ============================================================

def df(coll):
    """
    Compute document frequency for each term across a collection.
    Returns {term: document_frequency}.
    """
    df_dict = {}
    for doc in coll.values():
        for term in set(doc.terms.keys()):
            df_dict[term] = df_dict.get(term, 0) + 1
    return df_dict


def avg_len(coll):
    """Compute the average document length (in terms) across the collection."""
    return sum(doc.get_doc_size() for doc in coll.values()) / len(coll)


def collection_term_freq(coll):
    """
    Compute total frequency of each term across the entire collection.
    Returns {term: total_count}. Used by JM and Model_C.
    """
    ctf = {}
    for doc in coll.values():
        for term, freq in doc.terms.items():
            ctf[term] = ctf.get(term, 0) + freq
    return ctf


def collection_size(coll):
    """Return the total number of word occurrences in the collection (Dataset)."""
    return sum(doc.get_doc_size() for doc in coll.values())


def save_ranking(output_path, scores, score_name, query_title):
    """
    Save ranked document scores to a .dat file.
    Returns a sorted list of (doc_id, score) tuples.
    """
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Query is the topic title = \"{query_title}\"\n")
        f.write(f"Doc_ID {score_name}\n")
        for doc_id, score in ranked:
            f.write(f"{doc_id} {score}\n")
    return ranked


# ============================================================
# SECTION 3: TASK 1 – Baseline Models
# ============================================================

# ----------------------------------------------------------
# Baseline 1: BM25-Based IR Model
#
# Algorithm (for all datasets):
#   For each dataset i (corresponding to topic i):
#     Parse all documents in Dataset_i → collection C_i
#     Compute:
#       N    = total number of documents in C_i
#       avdl = average document length in C_i
#       df   = {term: n_t} where n_t = no. of docs containing term t
#     Parse topic title as query Q_i → {q_j: qf_t} where qf_t = query term frequency
#     For each document D in C_i:
#       score = 0
#       For each query term q_j in Q_i:
#         f_t  = frequency of q_j in D
#         n_t  = no. of docs in C_i containing q_j  (document frequency)
#         qf_t = frequency of q_j in query Q_i
#         K    = k1 * ((1-b) + b * dl / avdl)   where dl = |D|
#         IDF  = log((N - n_t + 0.5) / (n_t + 0.5))
#         TF   = ((k1 + 1) * f_t) / (K + f_t)
#         QF   = ((k2 + 1) * qf_t) / (k2 + qf_t)
#         score += IDF * TF * QF
#     Rank documents in C_i by BM25 score (descending)
#     Save ranked list as Baseline1_R<i>_Ranking.dat
#
# Parameters:
#   N    = number of documents in Dataset_i
#   n_t  = document frequency of term t (number of docs containing t)
#   f_t  = term frequency of t in document D
#   qf_t = frequency of query term t in the query Q_i
# ----------------------------------------------------------

def bm_25(coll, q, df_dict, stop_wordList):
    """
    Compute BM25 scores for all documents in a collection given query q.

    BM25(D, Q) = Σ_{t∈Q} log((N - n_t + 0.5)/(n_t + 0.5))
                          * ((k1+1)*f_t) / (K + f_t)
                          * ((k2+1)*qf_t) / (k2 + qf_t)

    Parameters:
        coll        : {docID: Doc} – the document collection (Dataset_i)
        q           : str – the topic title used as query
        df_dict     : {term: n_t} – document frequency dictionary
        stop_wordList: list of stopwords
    Returns:
        {docID: bm25_score}
    """
    k1, k2, b = 1.2, 500, 0.75
    N = len(coll)
    avdl = avg_len(coll)
    query_terms = queryParser(q, stop_wordList)   # {term: qf_t}
    scores = {}

    for docid, doc in coll.items():
        score = 0.0
        dl = doc.get_doc_size()
        K = k1 * ((1 - b) + b * (dl / avdl))

        for term, qf_t in query_terms.items():
            f_t = doc.terms.get(term, 0)
            if f_t == 0:
                continue
            n_t = df_dict.get(term, 0)
            if n_t == 0:
                continue

            idf = math.log((N - n_t + 0.5) / (n_t + 0.5))  # natural log, as per spec
            tf_weight = ((k1 + 1) * f_t) / (K + f_t)
            qf_weight = ((k2 + 1) * qf_t) / (k2 + qf_t)    # FIX: was missing in original

            score += idf * tf_weight * qf_weight

        scores[docid] = score

    return scores


# ----------------------------------------------------------
# Baseline 2: Jelinek–Mercer Smoothing Language Model
#
# Algorithm (for all datasets):
#   For each dataset i (corresponding to topic i):
#     Parse all documents in Dataset_i → collection C_i
#     Compute:
#       |Dataset_i|  = total number of word occurrences in C_i
#       ctf          = {term: c_qj} where c_qj = total occurrences of term in C_i
#     Parse topic title as query Q_i → terms {q_j}
#     For each document D in C_i:
#       score = 0
#       For each query term q_j in Q_i:
#         f_qj,D = frequency of q_j in D
#         |D|    = total word occurrences in D
#         c_qj   = total occurrences of q_j in C_i
#         score += log((1-λ) * (f_qj,D / |D|) + λ * (c_qj / |Dataset_i|))
#     Rank documents in C_i by JM score (descending)
#     Save ranked list as Baseline2_R<i>_Ranking.dat
#
# Parameters:
#   n   = |D|, the total number of word occurrences in document D
#   c_qj = total occurrences of query term q_j across Dataset_i (collection frequency)
# ----------------------------------------------------------

def jm_smoothing(coll, q, stop_wordList, lambda_=0.3):
    """
    Compute JM-smoothed language model scores for all documents.

    JM(D, Q) = Σ_{q_j∈Q} log((1-λ) * (f_qj,D / |D|) + λ * (c_qj / |Dataset_i|))

    Parameters:
        coll         : {docID: Doc} – the document collection
        q            : str – topic title as query
        stop_wordList: list of stopwords
        lambda_      : smoothing parameter (default 0.3)
    Returns:
        {docID: jm_score}
    """
    query_terms = queryParser(q, stop_wordList)
    ctf = collection_term_freq(coll)        # {term: c_qj}
    total_coll_size = collection_size(coll) # |Dataset_i|
    scores = {}

    for docid, doc in coll.items():
        score = 0.0
        n = doc.get_doc_size()  # |D|

        for term in query_terms:
            f_qj_D = doc.terms.get(term, 0)    # frequency in document
            c_qj = ctf.get(term, 0)            # collection frequency

            # Smoothed probability
            p_smooth = (1 - lambda_) * (f_qj_D / n if n > 0 else 0) \
                       + lambda_ * (c_qj / total_coll_size if total_coll_size > 0 else 0)

            if p_smooth > 0:
                score += math.log(p_smooth)
            # If p_smooth == 0 (term absent from both doc and collection), skip term

        scores[docid] = score

    return scores


# ============================================================
# SECTION 4: TASK 2 – Custom Model (Model_C)
# Pseudo-Relevance Feedback (PRF) with BM25 Re-ranking
# ============================================================
#
# Design:
#   Model_C extends BM25 (Baseline1) by applying Pseudo-Relevance Feedback (PRF).
#   The intuition is that the top-k documents from an initial retrieval are likely
#   relevant, so we extract their most discriminative terms and expand the query.
#
# Algorithm (for all datasets):
#   For each dataset i (corresponding to topic i):
#     Step 1: Initial retrieval
#       Run BM25 on C_i with original query Q_i (same as Baseline1)
#     Step 2: Pseudo-relevant set
#       Select top-k documents from Step 1 as pseudo-relevant set PRF_k
#     Step 3: Term extraction
#       For each term in PRF_k, compute a term weight using Rocchio's formula:
#         w(t) = Σ_{D ∈ PRF_k} (f_t,D / |D|)  (normalised term frequency sum)
#       Select top-m terms by weight (excluding original query terms)
#     Step 4: Query expansion
#       Add top-m expansion terms to the original query with weight α
#       {expanded_query} = original_query ∪ {top-m terms with freq = α}
#     Step 5: Re-rank
#       Re-run BM25 on C_i with expanded_query → final ranking
#   Save ranked list as ModelC_R<i>_Ranking.dat
#
# Differences from Baseline1 and Baseline2:
#   • Baseline1 uses only the raw topic title; Model_C expands it with PRF terms.
#   • Baseline2 models document relevance probabilistically via language modelling;
#     Model_C remains a BM25-based term-weighting model but uses an enriched query.
#   • Model_C is generic: it can be applied to any topic with no domain-specific tuning.
#
# Parameters:
#   top_k : number of top documents used as pseudo-relevant set (default 5)
#   top_m : number of expansion terms added to query (default 10)
#   alpha : weight given to expansion terms in the expanded query (default 1)

def model_c(coll, q, df_dict, stop_wordList, top_k=5, top_m=10, alpha=1):
    """
    Pseudo-Relevance Feedback model built on BM25.

    Step 1: Initial BM25 retrieval.
    Step 2: Take top_k docs as pseudo-relevant.
    Step 3: Extract top_m discriminative expansion terms from pseudo-relevant docs.
    Step 4: Expand query with expansion terms (weighted by alpha).
    Step 5: Re-rank all documents using BM25 with the expanded query.

    Parameters:
        coll         : {docID: Doc}
        q            : str – original topic title
        df_dict      : {term: doc_frequency}
        stop_wordList: list of stopwords
        top_k        : number of pseudo-relevant documents to use
        top_m        : number of expansion terms to add
        alpha        : weight for expansion terms in expanded query
    Returns:
        {docID: final_bm25_score}
    """
    # Step 1: Initial retrieval with BM25
    initial_scores = bm_25(coll, q, df_dict, stop_wordList)
    ranked_initial = sorted(initial_scores.items(), key=lambda x: x[1], reverse=True)

    # Step 2: Pseudo-relevant set – top_k documents
    pseudo_relevant_ids = [docid for docid, _ in ranked_initial[:top_k]]

    # Step 3: Compute term weights from pseudo-relevant documents (Rocchio-style)
    original_terms = set(queryParser(q, stop_wordList).keys())
    term_weights = {}

    for docid in pseudo_relevant_ids:
        doc = coll[docid]
        n = doc.get_doc_size()
        if n == 0:
            continue
        for term, freq in doc.terms.items():
            if term not in original_terms:  # Only add new terms
                term_weights[term] = term_weights.get(term, 0) + (freq / n)

    # Step 4: Select top_m expansion terms and build expanded query
    top_expansion_terms = sorted(term_weights.items(), key=lambda x: x[1], reverse=True)[:top_m]

    expanded_query = queryParser(q, stop_wordList)  # Start with original query terms
    for term, _ in top_expansion_terms:
        expanded_query[term] = expanded_query.get(term, 0) + alpha

    # Step 5: Re-rank using BM25 with expanded query
    # We pass the expanded query directly to a modified BM25 that accepts a dict
    k1, k2, b = 1.2, 500, 0.75
    N = len(coll)
    avdl = avg_len(coll)
    scores = {}

    for docid, doc in coll.items():
        score = 0.0
        dl = doc.get_doc_size()
        K = k1 * ((1 - b) + b * (dl / avdl))

        for term, qf_t in expanded_query.items():
            f_t = doc.terms.get(term, 0)
            if f_t == 0:
                continue
            n_t = df_dict.get(term, 0)
            if n_t == 0:
                continue

            idf = math.log((N - n_t + 0.5) / (n_t + 0.5))
            tf_weight = ((k1 + 1) * f_t) / (K + f_t)
            qf_weight = ((k2 + 1) * qf_t) / (k2 + qf_t)

            score += idf * tf_weight * qf_weight

        scores[docid] = score

    return scores


# ============================================================
# SECTION 5: TASK 3 – Main Runner (All Models, All Datasets)
# ============================================================

def run_all_models(
    topics_file="Topics.txt",
    doc_collection_folder="Doc_Collection",
    output_folder="ModelOutputs",
    stopword_file="common-english-words.txt"
):
    """
    Run Baseline1 (BM25), Baseline2 (JM), and Model_C (PRF+BM25) over all datasets.
    Saves .dat files and prints top-10 results for each topic/model.
    """
    os.makedirs(output_folder, exist_ok=True)

    stop_wordList = load_stopwords(stopword_file)
    stop_wordList = list(set(stop_wordList + EXTRA_STOPWORDS))

    topics = parse_topics(topics_file)

    for topic_id, topic_title in sorted(topics.items()):
        dataset_number = topic_id.replace("R", "")
        dataset_folder = os.path.join(doc_collection_folder, f"Dataset{dataset_number}")

        if not os.path.exists(dataset_folder):
            print(f"[SKIP] Dataset folder not found: {dataset_folder}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {topic_id} – \"{topic_title}\"")
        print(f"{'='*60}")

        # Parse documents for this dataset
        coll = docParser(stop_wordList, dataset_folder)
        df_dict = df(coll)

        # --- BASELINE 1: BM25 ---
        b1_scores = bm_25(coll, topic_title, df_dict, stop_wordList)
        b1_path = os.path.join(output_folder, f"Baseline1_{topic_id}_Ranking.dat")
        b1_ranked = save_ranking(b1_path, b1_scores, "BM25_Score", topic_title)

        print(f"\n{topic_id} Baseline1 Top 10 (Doc_ID BM25_Score):")
        for doc_id, score in b1_ranked[:10]:
            print(f"  {doc_id}  {score}")

        # --- BASELINE 2: JM Smoothing ---
        b2_scores = jm_smoothing(coll, topic_title, stop_wordList, lambda_=0.3)
        b2_path = os.path.join(output_folder, f"Baseline2_{topic_id}_Ranking.dat")
        b2_ranked = save_ranking(b2_path, b2_scores, "JM_Score", topic_title)

        print(f"\n{topic_id} Baseline2 Top 10 (Doc_ID JM_Score):")
        for doc_id, score in b2_ranked[:10]:
            print(f"  {doc_id}  {score}")

        # --- MODEL C: PRF + BM25 ---
        mc_scores = model_c(coll, topic_title, df_dict, stop_wordList,
                            top_k=5, top_m=10, alpha=1)
        mc_path = os.path.join(output_folder, f"ModelC_{topic_id}_Ranking.dat")
        mc_ranked = save_ranking(mc_path, mc_scores, "ModelC_Score", topic_title)

        print(f"\n{topic_id} Model_C Top 10 (Doc_ID ModelC_Score):")
        for doc_id, score in mc_ranked[:10]:
            print(f"  {doc_id}  {score}")

        print(f"\n[SAVED] {b1_path}")
        print(f"[SAVED] {b2_path}")
        print(f"[SAVED] {mc_path}")


# ============================================================
# SECTION 6: TASK 4 – Evaluation Functions
# ============================================================

def load_relevance_judgements(judg_file):
    """
    Load relevance judgements from a file.
    Format per line: <topic_id> <doc_id> <relevance_label>
    Returns a set of relevant doc_ids: {doc_id, ...}
    """
    relevant = set()
    with open(judg_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[2] == "1":
                relevant.add(parts[1])
    return relevant


def load_ranking_from_file(ranking_file):
    """
    Load a ranked list from a .dat file.
    Returns a list of doc_ids in ranked order.
    """
    ranked_docs = []
    with open(ranking_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            # Skip header lines (non-numeric first column)
            if len(parts) >= 2 and parts[0].isdigit():
                ranked_docs.append(parts[0])
    return ranked_docs


def average_precision(ranked_docs, relevant_set):
    """
    Compute Average Precision (AP) for a single ranked list.

    AP = (1 / |R|) * Σ_{k: doc_k relevant} Precision@k

    Parameters:
        ranked_docs  : list of doc_ids in ranked order
        relevant_set : set of relevant doc_ids
    Returns:
        float AP value
    """
    if not relevant_set:
        return 0.0
    num_relevant = len(relevant_set)
    hits = 0
    precision_sum = 0.0
    for k, doc_id in enumerate(ranked_docs, start=1):
        if doc_id in relevant_set:
            hits += 1
            precision_sum += hits / k
    return precision_sum / num_relevant


def precision_at_10(ranked_docs, relevant_set):
    """
    Compute Precision at rank 10 (P@10).

    P@10 = (# relevant docs in top 10) / 10

    Parameters:
        ranked_docs  : list of doc_ids in ranked order
        relevant_set : set of relevant doc_ids
    Returns:
        float P@10 value
    """
    top10 = ranked_docs[:10]
    hits = sum(1 for doc in top10 if doc in relevant_set)
    return hits / 10


def dcg_at_10(ranked_docs, relevant_set):
    """
    Compute Discounted Cumulative Gain at rank 10 (DCG10).

    DCG_10 = rel_1 + Σ_{i=2}^{10} rel_i / log2(i)

    where rel_i = 1 if document at position i is relevant, else 0.

    Parameters:
        ranked_docs  : list of doc_ids in ranked order
        relevant_set : set of relevant doc_ids
    Returns:
        float DCG10 value
    """
    top10 = ranked_docs[:10]
    dcg = 0.0
    for i, doc_id in enumerate(top10, start=1):
        rel = 1 if doc_id in relevant_set else 0
        if i == 1:
            dcg += rel
        else:
            dcg += rel / math.log2(i)
    return dcg


def evaluate_all_models(
    topics_file="Topics.txt",
    output_folder="ModelOutputs",
    relevance_folder="Relevant_Judgements"
):
    """
    Evaluate Baseline1, Baseline2, and Model_C across all topics using:
      - Average Precision (AP) and MAP
      - Precision@10 and its average
      - DCG@10 and its average

    Prints summary tables for each measure.
    Returns a dict of results: {model: {topic: {measure: value}}}
    """
    topics = parse_topics(topics_file)
    models = ["Baseline1", "Baseline2", "ModelC"]
    results = {model: {} for model in models}

    for topic_id in sorted(topics.keys()):
        dataset_number = topic_id.replace("R", "")
        judg_file = os.path.join(relevance_folder, f"Dataset{dataset_number}.txt")

        if not os.path.exists(judg_file):
            print(f"[SKIP] Judgement file not found: {judg_file}")
            continue

        relevant_set = load_relevance_judgements(judg_file)

        for model in models:
            prefix = "ModelC" if model == "ModelC" else model
            ranking_file = os.path.join(output_folder, f"{prefix}_{topic_id}_Ranking.dat")

            if not os.path.exists(ranking_file):
                print(f"[SKIP] Ranking file not found: {ranking_file}")
                results[model][topic_id] = {"AP": 0.0, "P@10": 0.0, "DCG10": 0.0}
                continue

            ranked_docs = load_ranking_from_file(ranking_file)

            ap   = average_precision(ranked_docs, relevant_set)
            p10  = precision_at_10(ranked_docs, relevant_set)
            dcg  = dcg_at_10(ranked_docs, relevant_set)

            results[model][topic_id] = {"AP": ap, "P@10": p10, "DCG10": dcg}

    # --- Print Table 1: Average Precision ---
    print("\n" + "="*65)
    print("Table 1. Performance of 3 models on Average Precision (AP)")
    print(f"{'Topic':<10} {'Baseline1':>12} {'Baseline2':>12} {'Model_C':>12}")
    print("-"*65)
    ap_sums = {m: 0.0 for m in models}
    all_topics = sorted(results["Baseline1"].keys())
    for topic_id in all_topics:
        row = f"{topic_id:<10}"
        for model in models:
            val = results[model].get(topic_id, {}).get("AP", 0.0)
            ap_sums[model] += val
            row += f"{val:>12.3f}"
        print(row)
    n = len(all_topics)
    print("-"*65)
    map_row = f"{'MAP':<10}"
    for model in models:
        map_row += f"{ap_sums[model]/n if n>0 else 0:>12.3f}"
    print(map_row)

    # --- Print Table 2: Precision@10 ---
    print("\n" + "="*65)
    print("Table 2. Performance of 3 models on Precision@10")
    print(f"{'Topic':<10} {'Baseline1':>12} {'Baseline2':>12} {'Model_C':>12}")
    print("-"*65)
    p10_sums = {m: 0.0 for m in models}
    for topic_id in all_topics:
        row = f"{topic_id:<10}"
        for model in models:
            val = results[model].get(topic_id, {}).get("P@10", 0.0)
            p10_sums[model] += val
            row += f"{val:>12.3f}"
        print(row)
    print("-"*65)
    avg_row = f"{'Average':<10}"
    for model in models:
        avg_row += f"{p10_sums[model]/n if n>0 else 0:>12.3f}"
    print(avg_row)

    # --- Print Table 3: DCG@10 ---
    print("\n" + "="*65)
    print("Table 3. Performance of 3 models on DCG@10")
    print(f"{'Topic':<10} {'Baseline1':>12} {'Baseline2':>12} {'Model_C':>12}")
    print("-"*65)
    dcg_sums = {m: 0.0 for m in models}
    for topic_id in all_topics:
        row = f"{topic_id:<10}"
        for model in models:
            val = results[model].get(topic_id, {}).get("DCG10", 0.0)
            dcg_sums[model] += val
            row += f"{val:>12.3f}"
        print(row)
    print("-"*65)
    avg_row = f"{'Average':<10}"
    for model in models:
        avg_row += f"{dcg_sums[model]/n if n>0 else 0:>12.3f}"
    print(avg_row)

    return results


# ============================================================
# SECTION 7: TASK 5 – Statistical Significance Testing (t-test)
# ============================================================

def significance_test(results, measure="AP"):
    """
    Conduct paired two-tailed t-tests comparing Model_C vs Baseline1 and Baseline2.

    H0: There is no significant difference in performance between Model_C and Baseline.
    H1: Model_C significantly differs from the Baseline.

    Uses scipy.stats.ttest_rel (paired t-test).

    Parameters:
        results : dict returned by evaluate_all_models()
        measure : one of "AP", "P@10", "DCG10"
    """
    all_topics = sorted(results["Baseline1"].keys())

    def get_scores(model):
        return [results[model].get(t, {}).get(measure, 0.0) for t in all_topics]

    mc_scores  = get_scores("ModelC")
    b1_scores  = get_scores("Baseline1")
    b2_scores  = get_scores("Baseline2")

    t1, p1 = stats.ttest_rel(mc_scores, b1_scores)
    t2, p2 = stats.ttest_rel(mc_scores, b2_scores)

    print(f"\n{'='*60}")
    print(f"Task 5 – Significance Test (Paired t-test) | Measure: {measure}")
    print(f"{'='*60}")
    print(f"  Model_C vs Baseline1: t = {t1:.4f},  p = {p1:.4f}",
          "--> SIGNIFICANT" if p1 < 0.05 else "--> not significant")
    print(f"  Model_C vs Baseline2: t = {t2:.4f},  p = {p2:.4f}",
          "--> SIGNIFICANT" if p2 < 0.05 else "--> not significant")
    print(f"\n  Significance threshold: p < 0.05")
    print(f"  Model_C mean {measure}: {sum(mc_scores)/len(mc_scores):.3f}")
    print(f"  Baseline1 mean {measure}: {sum(b1_scores)/len(b1_scores):.3f}")
    print(f"  Baseline2 mean {measure}: {sum(b2_scores)/len(b2_scores):.3f}")


# ============================================================
# SECTION 8: TASK 2 – Model_C Parameter Validation (Grid Search)
# ============================================================

def validate_model_c_parameters(
    topics_file="Topics.txt",
    doc_collection_folder="Doc_Collection",
    relevance_folder="Relevant_Judgements",
    stopword_file="common-english-words.txt",
    output_folder="ModelOutputs"
):
    """
    Grid search over Model_C hyperparameters (top_k, top_m) using MAP as criterion.
    Evaluates each configuration over all datasets and reports the best settings.
    """
    stop_wordList = load_stopwords(stopword_file)
    stop_wordList = list(set(stop_wordList + EXTRA_STOPWORDS))
    topics = parse_topics(topics_file)

    top_k_values = [3, 5, 10]
    top_m_values = [5, 10, 15, 20]

    best_map = -1
    best_params = {}
    results_grid = {}

    print("\nModel_C Parameter Validation (Grid Search)")
    print(f"{'top_k':>8} {'top_m':>8} {'MAP':>10}")
    print("-"*30)

    for top_k in top_k_values:
        for top_m in top_m_values:
            ap_list = []

            for topic_id, topic_title in sorted(topics.items()):
                dataset_number = topic_id.replace("R", "")
                dataset_folder = os.path.join(doc_collection_folder, f"Dataset{dataset_number}")
                judg_file = os.path.join(relevance_folder, f"Dataset{dataset_number}.txt")

                if not os.path.exists(dataset_folder) or not os.path.exists(judg_file):
                    continue

                coll = docParser(stop_wordList, dataset_folder)
                df_dict = df(coll)
                relevant_set = load_relevance_judgements(judg_file)

                mc_scores = model_c(coll, topic_title, df_dict, stop_wordList,
                                    top_k=top_k, top_m=top_m, alpha=1)
                ranked = sorted(mc_scores.items(), key=lambda x: x[1], reverse=True)
                ranked_docs = [doc_id for doc_id, _ in ranked]

                ap = average_precision(ranked_docs, relevant_set)
                ap_list.append(ap)

            map_score = sum(ap_list) / len(ap_list) if ap_list else 0.0
            results_grid[(top_k, top_m)] = map_score
            print(f"{top_k:>8} {top_m:>8} {map_score:>10.4f}")

            if map_score > best_map:
                best_map = map_score
                best_params = {"top_k": top_k, "top_m": top_m}

    print(f"\nBest Parameters: top_k={best_params['top_k']}, top_m={best_params['top_m']} → MAP={best_map:.4f}")
    return best_params


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    os.chdir("/Users/bmscanoza/Desktop/IFN647_A2")

    # Task 3: Run all models on all datasets and save .dat files
    run_all_models()

    # Task 4: Evaluate all models using AP, P@10, DCG@10
    results = evaluate_all_models()

    # Task 5: Statistical significance testing across all three measures
    for measure in ["AP", "P@10", "DCG10"]:
        significance_test(results, measure=measure)

    # Optional: Run parameter grid search for Model_C validation (Task 4b)
    # Uncomment the line below if you want to run validation (takes longer)
    # validate_model_c_parameters()
