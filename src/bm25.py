import re
from statistics import mean

from rank_bm25 import BM25Okapi
from rank_bm25 import BM25L
from rank_bm25 import BM25Plus

from src import baseline, util
from src.util import get_stemmed_corpus, preprocess
from gensim.summarization.bm25 import BM25


def calculate_bm25(corpus, query):
    bm25 = BM25Okapi([x['tokens'] for x in corpus])
    tokenized_query = preprocess(query)
    doc_scores = bm25.get_scores(tokenized_query)
    for i in range(len(corpus)):
        corpus[i]['score'] = doc_scores[i]
    ranked_docs = sorted(corpus, key=lambda x: x['score'], reverse=True)
    return ranked_docs


def cal_bm25_gensim(corpus, query):
    bm25 = BM25([x['tokens'] for x in corpus])
    tokenized_query = preprocess(query)
    doc_scores = bm25.get_scores_bow(tokenized_query)
    sorted_scores = sorted(doc_scores, key=lambda x: x[1], reverse=True)
    ranked_docs = [corpus[index] for index, score in sorted_scores]
    return ranked_docs


if __name__ == '__main__':
    corpus = get_stemmed_corpus(save=True)
    test_qs = util.get_test_questions(save=True)

    term_idfs = baseline.compute_term_idfs(corpus, save=True)
    tfidf_reprs = baseline.compute_tfidf_doc_repr(corpus, term_idfs, save=True)

    r_values = []
    for q in test_qs:
        # docs, scores = baseline.search_docs(test_qs[q], corpus, 1000)
        docs, scores = baseline.get_relevant_docs(test_qs[q], tfidf_reprs, term_idfs, how_many=1000)

        ans_pattern = "|".join(test_qs[q]['ans_patterns'])
        ranked_docs = calculate_bm25(corpus, test_qs[q]['raw_question'])
        ranked_docs1 = cal_bm25_gensim(corpus, test_qs[q]['raw_question'])

        relevant_docs = 0
        # Consider the best 50 documents
        top_ranked_docs = ranked_docs1[:50]
        for doc in top_ranked_docs:
            if bool(re.search(ans_pattern, doc['text'], flags=re.IGNORECASE)):
                relevant_docs = relevant_docs + 1

        r_value = relevant_docs / len(top_ranked_docs)
        # 0.11144324324324324
        r_values.append(r_value)

    print(mean(r_values))
