from statistics import mean

from gensim.summarization.bm25 import BM25
from rank_bm25 import BM25L
from rank_bm25 import BM25Okapi
from rank_bm25 import BM25Plus

from source_code import baseline, util
from source_code.util import get_corpus, preprocess


def calculate_bm25(corpus, query, variant='Okapi'):
    # Function for calculating BM25 scores for a prepocessed corpus.
    tokens = [x['tokens'] for x in corpus]
    if variant == 'Okapi':
        bm25 = BM25Okapi(tokens)
    elif variant == 'BM25L':
        bm25 = BM25L(tokens)
    elif variant == 'BM25Plus':
        bm25 = BM25Plus(tokens)
    tokenized_query = preprocess(query)
    doc_scores = bm25.get_scores(tokenized_query)

    ranked_docs = sorted(zip(corpus, doc_scores), key=lambda x: x[1], reverse=True)

    docs, scores = zip(*ranked_docs)
    return docs, scores


def calculate_bm25_sentences(corpus, query, variant='Okapi'):
    # Function for calculating BM25 scores for a corpus of sentences.
    # Only differece to `calculate_bm25` is the additional `prepocessing`
    tokens = [preprocess(x) for x in corpus]
    if variant == 'Okapi':
        bm25 = BM25Okapi(tokens)
    elif variant == 'BM25L':
        bm25 = BM25L(tokens)
    elif variant == 'BM25Plus':
        bm25 = BM25Plus(tokens)
    tokenized_query = preprocess(query)
    doc_scores = bm25.get_scores(tokenized_query)

    ranked_docs = sorted(zip(corpus, doc_scores), key=lambda x: x[1], reverse=True)

    docs, scores = zip(*ranked_docs)
    return docs, scores


def calculate_bm25_gensim(corpus, query):
    bm25 = BM25([x['tokens'] for x in corpus])
    tokenized_query = preprocess(query)
    doc_scores = bm25.get_scores_bow(tokenized_query)
    sorted_scores = sorted(doc_scores, key=lambda x: x[1], reverse=True)
    ranked_docs = [corpus[index] for index, score in sorted_scores]
    return ranked_docs


if __name__ == '__main__':
    corpus = get_corpus(save=True)
    test_qs = util.get_test_questions(save=True)

    term_idfs = baseline.compute_term_idfs(corpus, save=True)
    tfidf_reprs = baseline.compute_tfidf_doc_repr(corpus, term_idfs, save=True)

    # Benchmark different BM25 variiants
    for variant in ['Okapi', 'BM25L', 'BM25Plus']:
        # Precision for different r values
        for r in range(10, 60, 10):

            precision_values = []
            for question in test_qs.values():
                docs, _ = baseline.search_docs(question['raw_question'], tfidf_reprs,
                                                    term_idfs, corpus, 1000)

                ans_pattern = "|".join(question['ans_patterns'])

                ranked_docs, _ = calculate_bm25(corpus, question['raw_question'], variant)
                # ranked_docs = calculate_bm25_gensim(docs, question['raw_question'])

                precision = util.precision_at_r(ranked_docs, ans_pattern, r)
                # print(f'precision({r}): {precision} for question: {question["raw_question"]}')
                precision_values.append(precision)
            print(f'Precision at r={r} using variant {variant}: {mean(precision_values)}')
