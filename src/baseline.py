import math
import os
import pickle
import re
import string
from collections import Counter
from statistics import mean

from bs4 import BeautifulSoup

from src import util

data_root = "../data/"
ans_patterns = data_root + "patterns.txt"
test_questions = data_root + "test_questions.txt"
trec_corpus_xml = data_root + "trec_documents.xml"

processed_root = data_root + "processed/"
os.makedirs(processed_root, exist_ok=True)

processed_tfids = processed_root + "tfids.pkl"
processed_tfidf_repr = processed_root + "tfrepr.pkl"

question_extraction_pattern = "Number: (\d+) *\n\n\<desc\> Description\:\n(\w+.*)\n\n\<\/top>"


# create representation of all docs in terms of their term freqs
def compute_tfidf_doc_repr(corpus, term_idfs, save=False):
    try:
        print("Loading from saved pickle")
        corpus_tfidf_repr = pickle.load(open(processed_tfidf_repr, "rb"))

        return corpus_tfidf_repr

    except Exception as e:

        corpus_tfidf_repr = []

        for doc in corpus:

            tf_repr = Counter(doc['tokens'])
            doc_max = max(tf_repr.values())

            for k, v in tf_repr.items():
                # normalize by max freq
                tf_repr[k] = tf_repr[k] / doc_max
                # weight tf by idf
                tf_repr[k] = tf_repr[k] * term_idfs[k]

            corpus_tfidf_repr.append(tf_repr)

        if save:
            print("saving tfid repr, existing data will be overwritten")
            pickle.dump(
                corpus_tfidf_repr,
                open(processed_tfidf_repr, "wb")
            )

        return corpus_tfidf_repr


# returns the idf weighted representation for a query, given tf based repr as input
def get_tfidfs_repr(query, term_idfs):
    tokens = util.preprocess(query)
    tf_idf_repr = Counter(tokens)
    q_max = max(tf_idf_repr.values())

    for k, val in tf_idf_repr.items():
        # normalize by max freq
        tf_idf_repr[k] = tf_idf_repr[k] / q_max
        # weight tf by idf
        try:
            tf_idf_repr[k] = tf_idf_repr[k] * term_idfs[k]
        except KeyError as ke:
            # we might not have IDF score for some question terms.
            # so we just use TF value, this is same as setting IDF = 1
            pass

    return tf_idf_repr


def cosine_sim(q, d):
    # only terms common b/w q and d affect the dot product
    # all other entries are either zero in query or in doc    
    common_terms = set(q.keys()).intersection(set(d.keys()))

    dot_prod = 0

    for ct in common_terms:
        dot_prod += q[ct] * d[ct]

    mag_q = sum([v ** 2 for v in q.values()])
    mag_d = sum([v ** 2 for v in d.values()])

    denom = math.sqrt(mag_q) * math.sqrt(mag_d)

    score = dot_prod / denom

    return score


def compute_term_idfs(corpus, save=False):
    try:
        print("Loading from saved pickle")
        term_doc_freq = pickle.load(open(processed_tfids, "rb"))
        return term_doc_freq

    except Exception as e:

        term_doc_freq = {}
        N = len(corpus)

        # first we get the document freq of a term 
        # i.e. how many docs contain that term
        # this is upper bounded by num of docs, of course
        for doc in corpus:

            # we are interested in just occurrence, and not actual freqs
            # that's why we convert the doc to set of non-repeating terms
            terms = set(doc['tokens'])

            for term in terms:

                if term in term_doc_freq.keys():
                    term_doc_freq[term] += 1
                else:
                    term_doc_freq[term] = 1

        # now that we have term's df, we inverse it and apply log normalization
        for t in term_doc_freq.keys():
            term_doc_freq[t] = math.log(N / term_doc_freq[t])

        if save:
            print("saving tfids, existing data will be overwritten")
            pickle.dump(
                term_doc_freq,
                open(processed_tfids, "wb")
            )

        return term_doc_freq


# TODO: Rename - there are not relevant docs, just retrieved docs
def get_relevant_docs(query, tfidf_reprs, term_idfs, corpus, how_many=1000):
    assert how_many < len(tfidf_reprs)

    q = get_tfidfs_repr(query, term_idfs)

    doc_scores = [cosine_sim(q, x) for x in tfidf_reprs]

    # Return only top 1000 results by default
    sorted_docs = sorted(zip(corpus, doc_scores), key=lambda x: x[1], reverse=True)[:how_many]

    # Remove docs with a score of 0
    sorted_docs = list(filter(lambda x: x[1] > 0, sorted_docs))

    docs, scores = zip(*sorted_docs)
    return docs, scores


if __name__ == "__main__":

    corpus = util.get_corpus(save=True)
    term_idfs = compute_term_idfs(corpus, save=True)
    tfidf_reprs = compute_tfidf_doc_repr(corpus, term_idfs, save=True)

    test_qs = util.get_test_questions(save=True)

    for r in range(10, 60, 10):
        precision_values = []
        for question in test_qs.values():
            docs, scores = get_relevant_docs(question['raw_question'], tfidf_reprs,
                                             term_idfs, corpus)

            ans_pattern = "|".join(question['ans_patterns'])

            precision = util.precision_at_r(docs, ans_pattern, r)
            # print(f'precision({r}): {precision} for question: {question["raw_question"]}')
            precision_values.append(precision)

        print(f'Precision at r={r}: {mean(precision_values)}')
