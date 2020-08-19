import math
import os
import pickle
import re
import string
from collections import Counter

from bs4 import BeautifulSoup

data_root = "../data/"
ans_patterns = data_root + "patterns.txt"
test_questions = data_root + "test_questions.txt"
trec_corpus_xml = data_root + "trec_documents.xml"

processed_root = data_root + "processed/"
os.makedirs(processed_root, exist_ok=True)

processed_corpus = processed_root + "corpus.pkl"
processed_text_qs = processed_root + "test_qs.pkl"
processed_tfids = processed_root + "tfids.pkl"
processed_tfidf_repr = processed_root + "tfrepr.pkl"

question_extraction_pattern = "Number: (\d+) *\n\n\<desc\> Description\:\n(\w+.*)\n\n\<\/top>"


def get_test_questions(test_questions, ans_patterns, save=False):
    try:
        print("Loading from saved pickle")
        test_qs = pickle.load(open(processed_text_qs, "rb"))
        return test_qs

    except Exception as e:

        # get question id and text
        qs = {}
        questions_doc = open(test_questions).read()
        question_extraction_pattern = "^\<num\> Number: (\d+) *\n\n\<desc\> Description\:\n(\w+.*)\n\n\<\/top>$"
        result = re.findall(question_extraction_pattern, questions_doc, re.MULTILINE)

        for q in result:
            processed_q = q[1].lower()
            processed_q = processed_q.translate(str.maketrans('', '', string.punctuation))

            qs[int(q[0])] = {'raw_question': q[1], 'question': processed_q, 'ans_patterns': []}

        # get associated answer patterns
        ans_doc = open(ans_patterns).readlines()

        for ap in ans_doc:
            # print(ap)
            ap = ap.split(" ")
            id, pattern = ap[0], " ".join(ap[1:]).strip()
            qs[int(id)]['ans_patterns'].append(pattern)

        if save:
            print("saving processed questions, existing data will be overwritten")
            pickle.dump(
                qs,
                open(processed_text_qs, "wb")
            )

        return qs


def process_trec_xml(trec_corpus_xml, save=False):
    try:
        print("Loading from saved pickle")
        corpus = pickle.load(open(processed_corpus, "rb"))
        return corpus

    except Exception as e:

        print("Data doesn't exit or other error", e)
        print("Processing from scratch")

        corpus = {
            # doc_id -> doc_text
        }

        with open(trec_corpus_xml, 'r') as dh:

            soup = BeautifulSoup(dh, 'html.parser')
            article_texts = soup.find_all('doc')
            print("Found %d articles..." % len(article_texts))

            for a in article_texts:
                # for now we don't separate byline / headline etc
                doc_id = a.docno.get_text().lower().strip()
                text = a.get_text().lower()
                # remove common punct
                text = text.translate(str.maketrans('', '', string.punctuation))
                text = text.replace('\n', '')
                text = text.replace('\r', '')

                corpus[doc_id] = text

        if save:
            print("saving processed corpus, existing data will be overwritten")
            pickle.dump(
                corpus,
                open(processed_corpus, "wb")
            )

        return corpus


# create representation of all docs in terms of their term freqs
def compute_tfidf_doc_repr(corpus, term_idfs, save=False):
    try:
        print("Loading from saved pickle")
        corpus_tfidf_repr = pickle.load(open(processed_tfidf_repr, "rb"))

        return corpus_tfidf_repr

    except Exception as e:

        corpus_tfidf_repr = {}

        for doc_id in corpus:

            tf_repr = Counter(corpus[doc_id].split(" "))
            doc_max = max(tf_repr.values())

            for k, v in tf_repr.items():
                # normalize by max freq
                tf_repr[k] = tf_repr[k] / doc_max
                # weight tf by idf
                tf_repr[k] = tf_repr[k] * term_idfs[k]

            corpus_tfidf_repr[doc_id] = tf_repr

        if save:
            print("saving tfid repr, existing data will be overwritten")
            pickle.dump(
                corpus_tfidf_repr,
                open(processed_tfidf_repr, "wb")
            )

        return corpus_tfidf_repr


# returns the idf weighted representation, given tf based repr as input
def get_tfidfs_repr(v, term_idfs):
    v = Counter(v.split(" "))
    q_max = max(v.values())

    for k, val in v.items():
        # normalize by max freq
        v[k] = v[k] / q_max
        # weight tf by idf
        try:
            v[k] = v[k] * term_idfs[k]
        except KeyError as ke:
            # we might not have IDF score for some question terms.
            # so we just use TF value, this is same as setting IDF = 1
            pass

    return v


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
        N = len(corpus.keys())

        # first we get the document freq of a term 
        # i.e. how many docs contain that term
        # this is upper bounded by num of docs, of course
        for doc in corpus:

            # we are interested in just occurrence, and not actual freqs
            # that's why we convert the doc to set of non-repeating terms
            terms = set(corpus[doc].split(" "))

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


def get_relevant_docs(q, tfidf_reprs, term_idfs, how_many=1):
    assert how_many < len(tfidf_reprs)

    doc_scores = {
        # doc id -> doc score
    }

    q = get_tfidfs_repr(q['question'], term_idfs)

    for d in tfidf_reprs:
        doc_scores[d] = cosine_sim(q, tfidf_reprs[d])

    sorted_scores = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:how_many]

    # unpack the dict into separate lists
    doc_ids, scores = zip(*sorted_scores)

    return doc_ids, scores


def precision_at_r(returned_docs, q, corpus):
    R = len(returned_docs)
    relevant_count = 0

    # print(R, q)
    # check if doc is relevant wrt any of the answer patterns
    for d in returned_docs:
        rel = []
        for ap in q['ans_patterns']:
            rel.append(bool(re.search(ap.strip(), corpus[d], flags=re.IGNORECASE)))

        relevant_count += int(any(rel))

    # print(relevant_count)
    return relevant_count / R

# a utility function to find out how many docs are relevant 
# to a given query. Can be considered a histogram.
# we expect to see a highly skewed histogram, with 1-5 docs relevant per query.    
def relevant_per_query(corpus, queries):
    #print(queries)
    # init rel counts with zero per query
    from collections import defaultdict
    rel_counts = [0 for i in range(len(queries))]
    rel_anspat = defaultdict(int)
    # check if doc is relevant wrt any of the answer patterns
    for doc in corpus:
        for q in queries:
            #print(q)
            for ap in queries[q]['ans_patterns']:
                rel_anspat[ap.strip()] += bool(re.search(ap.strip(), corpus[doc], flags=re.IGNORECASE))
                rel_counts[q-1] += bool(re.search(ap.strip(), corpus[doc], flags=re.IGNORECASE))

    return rel_counts, rel_anspat
     


if __name__ == "__main__":

    corpus = process_trec_xml(trec_corpus_xml, save=True)
    term_idfs = compute_term_idfs(corpus, save=True)
    tfidf_reprs = compute_tfidf_doc_repr(corpus, term_idfs, save=True)

    test_qs = get_test_questions(test_questions, ans_patterns, save=True)

    # ps = []
    # for q in test_qs:
    #     rel_docs, scores = get_relevant_docs(test_qs[q], tfidf_reprs, term_idfs, how_many=50)

    #     ps.append(
    #         precision_at_r(rel_docs, test_qs[q], corpus)
    #     )

    print( 
        relevant_per_query(corpus, test_qs)
    )
    #print(sum(ps) / 50)
