# BONN MICHAEL S. CANOZA, Student Number: N12349208
# IFN647-Machine Learning for Natural Language Processing
# Assignment 2

# Import relevant Libraries
import os
import string
import math
from stemming.porter2 import stem

# ==================================================================================================================================================================================

# Question 1- Parsing of Documents and Queries

# ==================================================================================================================================================================================

# Task 1.1 - Doc class using Bag-of-Words
# Step 1) Document Reading and Processing

class Doc:  # Doc class using Bag-of-Words
    def __init__(self, docID):  # a constructor created to run automatically when a new Doc objecty is created
        self.docID = docID  # document identifier attribute from itemid. It stores the document ID
        self.terms = {}  # an empty dictionary storing term-frequency pairs of the form (string term: int frequency
        self.doc_size = 0  # an attribute representing the document length/total number of terms in document after preprocessing

    def add_term(self, term):  # add_term in the Doc class to add a new term or increment its frequency if the term already exists
        """Add a term or increase its frequency if it already exists."""
        self.doc_size += 1  # each time a term is added, the document length increases by 1
        if term in self.terms:  # to check if the term already exists in the dictionary
            self.terms[term] += 1  # if the term already exists, increase its count.
        else:
            self.terms[
                term] = 1  # if the word appears for the first time, it is added to the dictionary with frequency 1

    def get_docID(self):
        return self.docID  # additional method to return the document identifier (docID)

    def get_termList(self):
        return sorted(
            self.terms.keys())  # additional method to return a sorted list of all terms occuring in the document, the sorted() arranges the term alphabetically

    def get_doc_size(self):  # returns the total number of terms in the document
        return self.doc_size

    def set_doc_size(self, size):
        self.doc_size = size

# Loading stopwords
def load_stopwords(filepath):  # defines a function name load_stopwords, the parameter filepath is the location of the stop-word file
    with open(filepath, 'r') as file:  # encoding="utf-8" ensures the file is read correctly even if its contains special characters
        stop_wordList = file.read().split(",") # ensures all words are separated consistently
    return stop_wordList


# Parsing a single document
def parse_doc(file, stop_wordList):
    in_text = False
    doc = None

    for line in file:
        line = line.strip()

        if line.startswith("<newsitem "):
            for part in line.split():
                if part.startswith("itemid="):
                    docid = part.split("=")[1].split("\"")[1]
                    doc = Doc(docid)

        if "<text>" in line:
            in_text = True
            line = line.split("<text>", 1)[1]

# process only content inside <text>...</text>
# it was found out during the checking of top 30 most frequent items that "copyright", "metadata", "newsitem", "editdetail" are not article text and XML tags must be excluded
# This is to normalize and reduce dimensionality of document vocabulary.
        if in_text and doc:
            if "</text>" in line:
                line = line.split("</text>", 1)[0]
                in_text = False

            line = line.replace("<p>", "").replace("</p>", "")
            line = line.translate(str.maketrans('', '', string.punctuation + string.digits)) # remove punctuation and digits
            

# DEFINITIONS:
# Word: A raw token extracted from the document text after splitting by whitespace.
#       Words may include stopwords and unstemmed forms, but should consist of alphabetic characters only (punctuation and digits are excluded).

# Term: A processed word that has pass through all preprocessing steps:
#      (1) tokenized from the <text> content,
#      (2) lowercase, stripped of punctuation and digits,
#      (3) not present in the stop-word list and
#      (4) reduced to its root form (using Porter2 stemming algorithm)

            words = line.lower().split() #tokenize into words
            for word in words:
                if word not in stop_wordList:
                    stemmed = stem(word)
                    doc.add_term(stemmed)
    return doc

# Parsing all documents in folder

def docParser(stop_wordList, folder):
    collection = {}
    for filename in os.listdir(folder):
        if filename.endswith(".xml"):
            with open(os.path.join(folder, filename), 'r', encoding='utf-8') as f:
                doc = parse_doc(f, stop_wordList)
                if doc:
                    collection[doc.docID] = doc
    return collection

# Task 1.2 - Parse query

def queryParser(query, stop_wordList):  # this defines the function named queryParser, where the input query is assumed to be a simple sentence
    query_terms = {}  # this dictionary will store the terms and their frequencies in the query.
    query = query.replace("-", " ")
    query = query.translate(str.maketrans('', '', string.punctuation + string.digits))
    terms = query.lower().split()

    for term in terms:
        if term not in stop_wordList:
            stemmed = stem(term)
            query_terms[stemmed] = query_terms.get(stemmed, 0) + 1
    return query_terms

# Task 1.3 - Main function
# The additional stop words selected after examining high-frequency terms in the parsed collection.These words appeared very often across many documents
# but contribute little to distinguishing topics. Removing them reduces noise and improves retrieval effectiveness. In addition, I examined the entire parsed collection
# and found that there were some terms or letters with no meaning at all

extra_stopwords = ["one", "over", "day", "five", "month", "up", "between", "three", "ec", "gm", "cm", "fm", "sc", "b", "bb", "db", "e", "s", "v", "x", "tf", "c", "w", "ag", "gmc", "lx"]



# Task 2.1: Document Frequency: Defining function df(coll) and computing document frequency in a given doc collection.
# The function should return a dictionary of the form {term: df}

def df(coll): # from week 4
    df_dict = {} # creates an empty dictionary to store document frequency values
    for doc in coll.values(): # loops for each document
        unique_terms = set(doc.terms.keys()) # get the distinct terms in the documents
        for term in unique_terms: # loops each distinct term in the document
            df_dict[term] = df_dict.get(term, 0) + 1 # adds 1 to that term's DF count
    return df_dict # the function should return a dictionary of the form {term: df}


# ==================================================================================================================================================================================

# Question 3- BM25-Based IR Model

# ==================================================================================================================================================================================

# Task 3.1: define function avg_len to calculate and return average doc length
def avg_len(coll):
    totalDocLength = 0
    for doc in coll.values():
        totalDocLength += doc.get_doc_size()
    return totalDocLength / len(coll)

# Task 3.2: Define bm_25 function and calculating document score
# The extra parameter stop_wordList was added to bm_25() so the function could call queryParser(q, stop_wordList) and apply the same preprocessing steps used for documents, including stop-word
# removal and stemming.This helps ensure consistency between document and query representations, as required by the assignment, while leaving the BM25 scoring formula unchanged.
def bm_25(coll, q, df, stop_wordList): 
    scores = {}
    N = len(coll)
    avg_doc_len = avg_len(coll)
    query_terms = queryParser(q, stop_wordList)
    k1 = 1.2
    k2 = 500
    b = 0.75
    
    for docid, doc in coll.items():
        score = 0.0
        doc_len = doc.get_doc_size()
        
        for term in query_terms:
            tf = doc.terms.get(term, 0)
            if tf == 0:
                continue
            df_t = df.get(term, 0)
            if df_t == 0:
                continue
            
            idf = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1)   # The assignment requires to use the base of log is e
            denominator1 = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
            score += idf * ((tf * (k1 + 1)) /denominator1)
        scores[docid] = score
    return scores


def parse_topics(topic_file):
    topics = {}

    with open(topic_file, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    blocks = content.split("<Topic>")

    for block in blocks:
        if "<num>" not in block or "<title>" not in block:
            continue

        num_part = block.split("<num>", 1)[1].split("</num>", 1)[0].strip()
        title_part = block.split("<title>", 1)[1].split("</title>", 1)[0].strip()

        # Make sure topic ID becomes R101, R102, etc.
        if "R" in num_part:
            topic_id = num_part[num_part.find("R"):].strip()
        else:
            topic_id = num_part.strip()

        topics[topic_id] = title_part

    return topics


def save_ranking(output_path, scores, score_name):
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    with open(output_path, "w", encoding="utf-8") as outfile:
        outfile.write(f"Doc_ID {score_name}\n")
        for doc_id, score in ranked:
            outfile.write(f"{doc_id} {score}\n")

    return ranked


def main_assignment2():
    TOPICS_FILE = "Topics.txt"
    DOC_COLLECTION_FOLDER = "Doc_Collection"
    OUTPUT_FOLDER = "ModelOutputs"
    STOPWORD_FILE = "common-english-words.txt"

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    stop_wordList = load_stopwords(STOPWORD_FILE)
    stop_wordList = list(set(stop_wordList + extra_stopwords))

    topics = parse_topics(TOPICS_FILE)

    for topic_id, topic_title in topics.items():
        dataset_number = topic_id.replace("R", "")
        dataset_folder = os.path.join(DOC_COLLECTION_FOLDER, f"Dataset{dataset_number}")

        if not os.path.exists(dataset_folder):
            print("Dataset folder not found:", dataset_folder)
            continue

        print("Processing:", topic_id, "-", topic_title)

        coll = docParser(stop_wordList, dataset_folder)
        df_dict = df(coll)

        bm25_scores = bm_25(coll, topic_title, df_dict, stop_wordList)

        output_path = os.path.join(
            OUTPUT_FOLDER,
            f"Baseline1_{topic_id}_Ranking.dat"
        )

        ranked = save_ranking(output_path, bm25_scores, "BM25_Score")

        print(f"{topic_id} Baseline1 Top 10:")
        for doc_id, score in ranked[:10]:
            print(doc_id, score)

        print("Saved to:", output_path)
        print("-" * 50)


if __name__ == "__main__":
    main_assignment2()
