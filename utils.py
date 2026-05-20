"""
IFN647 Assignment 2 - Shared Utilities
Document/query parsing, collection helpers, and output functions.
(Sections 1 & 2 of A2.py)
"""

import os
import re
import string
from collections import defaultdict
from stemming.porter2 import stem

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
TOPICS_FILE = os.path.join(BASE_DIR, "Topics.txt")
DOC_DIR     = os.path.join(BASE_DIR, "Doc_Collection")
OUTPUT_DIR  = os.path.join(BASE_DIR, "ModelOutputs")
REL_DIR     = os.path.join(BASE_DIR, "Relevant_Judgements")

EXTRA_STOPWORDS = [
    's', 'gm', 'co', 'sc', 'v', 'quot', 'amp', 'lt', 'gt', 'apos',
    'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 't', 'u', 'w', 'x', 'y', 'z',
    'up', 'down', 'left', 'right', 'be', 'been', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'shall', 'should',
    'hadn', 'hasn', 'haven', 'doesn', 'don', 'didn', 'won', 'wouldn',
    'shan', 'shouldn', 'can', 'could', 'might', 'must', 'need', 'ought',
    'uk', 'us', 'eu',
    'one', 'over', 'day', 'five', 'month', 'between', 'three',
    'ec', 'cm', 'fm', 'bb', 'db', 'tf', 'ag', 'gmc', 'lx',
]


# =============================================================================
# Section 1 – Document & Query Parsing
# =============================================================================

class Doc:
    """Bag-of-Words document representation."""

    def __init__(self, docID):
        self.docID    = docID
        self.terms    = {}
        self.doc_size = 0

    def add_term(self, term):
        self.doc_size += 1
        self.terms[term] = self.terms.get(term, 0) + 1

    def get_docID(self):    return self.docID
    def get_termList(self): return sorted(self.terms.keys())
    def get_doc_size(self): return self.doc_size
    def set_doc_size(self, size): self.doc_size = size


def load_stopwords(filepath):
    """Load comma-separated stop words from file."""
    with open(filepath, 'r') as f:
        return f.read().split(",")


def parse_doc(file, stop_wordList):
    """Parse a single XML news document; extracts text from <text>…</text>."""
    in_text = False
    doc     = None

    for line in file:
        line = line.strip()

        if line.startswith("<newsitem "):
            for part in line.split():
                if part.startswith("itemid="):
                    docid = part.split("=")[1].strip('"')
                    doc   = Doc(docid)

        if "<text>" in line:
            in_text = True
            line    = line.split("<text>", 1)[1]

        if in_text and doc:
            if "</text>" in line:
                line    = line.split("</text>", 1)[0]
                in_text = False

            line = line.replace("<p>", "").replace("</p>", "")
            line = line.translate(
                str.maketrans('', '', string.punctuation + string.digits)
            )
            for word in line.lower().split():
                if word not in stop_wordList:
                    doc.add_term(stem(word))

    return doc


def load_stop_words(file_path):
    """Reads comma-separated stop-word file and returns a set."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return set(f.read().split(','))


def extract_docID(raw):
    """Extracts itemid attribute from <newsitem> tag."""
    match = re.search(r'<newsitem\b[^>]*\bitemid=["\']([^"\']+)["\']', raw, re.IGNORECASE)
    return match.group(1) if match else None


def extract_text(raw):
    """Extracts plain text from <text>…</text>, strips tags and HTML entities."""
    match = re.search(r'<text\b[^>]*>(.*?)</text>', raw, re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    block = match.group(1)
    block = re.sub(r'<[^>]+>', ' ', block)
    for entity, char in [('&quot;', '"'), ('&amp;', '&'), ('&lt;', '<'),
                          ('&gt;', '>'), ('&apos;', "'")]:
        block = block.replace(entity, char)
    return block.strip()

def preprocess_text(text, stop_words):
    """Tokenize → drop digit-containing tokens → remove stops → stem.
    Returns (raw_words, stemmed_terms); raw_words length drives doc_size.
    """
    words = re.findall(r'\w+', text.lower())
    words = [w for w in words if not any(c.isdigit() for c in w)]
    terms = [stem(w) for w in words if w not in stop_words]
    return words, terms


def docParser(stop_words, folder):
    """Builds {docID: Doc} from all XML files in folder.
    Uses iso-8859-1 encoding to match the Reuters XML corpus.
    """
    collection = {}
    for filename in os.listdir(folder):
        if not filename.endswith('.xml'):
            continue
        with open(os.path.join(folder, filename), 'r',
                  encoding='iso-8859-1', errors='replace') as f:
            raw = f.read()
        docID = extract_docID(raw)
        if docID is None:
            continue
        raw_words, terms = preprocess_text(extract_text(raw), stop_words)
        doc = Doc(docID)
        for term in terms:
            doc.add_term(term)
        doc.set_doc_size(len(raw_words))
        collection[docID] = doc
    return collection


def queryParser(query, stop_words):
    """Applies the same pipeline as docParser to a query string.
    Returns {term: frequency}.
    """
    _, terms = preprocess_text(query, stop_words)
    freq = {}
    for term in terms:
        freq[term] = freq.get(term, 0) + 1
    return freq


def parse_topics(topic_file=TOPICS_FILE):
    """Parse Topics.txt using regex; returns {topic_id: title}."""
    with open(topic_file, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    topics = {}
    for block in re.finditer(r"<Topic>(.*?)</Topic>", content, re.DOTALL | re.IGNORECASE):
        text = block.group(1)
        num_m   = re.search(r"<num>\s*(.*?)\s*(?:</num>|<title>|<desc>|<narr>)", text, re.DOTALL | re.IGNORECASE)
        title_m = re.search(r"<title>\s*(.*?)\s*(?:</title>|<desc>|<narr>|$)", text, re.DOTALL | re.IGNORECASE)
        if not num_m or not title_m:
            continue
        num_str  = num_m.group(1).strip()
        topic_id = num_str[num_str.find("R"):].strip() if "R" in num_str else num_str
        topics[topic_id] = title_m.group(1).strip()

    return topics


# =============================================================================
# Section 2 – Helper Functions
# =============================================================================

def df(coll):
    """Returns {term: document_frequency}."""
    df_dict = {}
    for doc in coll.values():
        for term in set(doc.terms.keys()):
            df_dict[term] = df_dict.get(term, 0) + 1
    return df_dict


def avg_len(coll):
    """Mean document length (in terms) across the collection."""
    return sum(doc.get_doc_size() for doc in coll.values()) / len(coll)


def collection_term_freq(coll):
    """Returns {term: total_count_across_collection}."""
    ctf = {}
    for doc in coll.values():
        for term, freq in doc.terms.items():
            ctf[term] = ctf.get(term, 0) + freq
    return ctf


def collection_size(coll):
    """Total word occurrences across the collection."""
    return sum(doc.get_doc_size() for doc in coll.values())


def build_inv_index(coll):
    """Returns {term: {docid: freq}} inverted index for a collection."""
    inv_index = defaultdict(dict)
    for docid, doc in coll.items():
        for term, freq in doc.terms.items():
            inv_index[term][docid] = freq
    return dict(inv_index)


def save_ranking(output_path, scores, score_name, query_title):
    """Sort by score descending and write to a .dat file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Query is the topic title = \"{query_title}\"\n")
        f.write(f"Doc_ID {score_name}\n")
        for doc_id, score in ranked:
            f.write(f"{doc_id} {score}\n")
    return ranked
