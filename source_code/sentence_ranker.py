import pprint
import re
from statistics import mean
import matplotlib.pyplot as plt

import numpy as np
from nltk import sent_tokenize

from source_code import baseline, util
from source_code.bm25 import calculate_bm25, calculate_bm25_sentences
from source_code.util import get_corpus

if __name__ == '__main__':
    corpus = get_corpus(save=True)
    test_qs = util.get_test_questions(save=True)

    term_idfs = baseline.compute_term_idfs(corpus, save=True)
    tfidf_reprs = baseline.compute_tfidf_doc_repr(corpus, term_idfs, save=True)

    reciprocal_ranks = []
    ranks = []
    question_answer = {}
    unanswered = []

    for question in test_qs.values():
        docs, _ = baseline.search_docs(question['raw_question'], tfidf_reprs,
                                            term_idfs, corpus, 1000)

        ranked_docs, _ = calculate_bm25(corpus, question['raw_question'])

        # 1. First, get sentences from top 50 documents
        sentences = []
        for doc in ranked_docs[:50]:
            sent_tokens = sent_tokenize(doc['text'])
            for token in sent_tokens:
                token = token.replace('\r', '').replace('\n', '')
                sentences.append(token)

        # 2. Use sentences as documents to rank them using BM25
        ranked_sent, _ = calculate_bm25_sentences(sentences, question['raw_question'])

        # 3. Find first relevant sentence
        ans_pattern = "|".join(question['ans_patterns'])
        is_answered = False
        for i, sent in enumerate(ranked_sent):
            if bool(re.search(ans_pattern, sent)):
                # rank is the first relevant sentence
                reciprocal_ranks.append(1 / (i + 1))
                ranks.append(i + 1)
                is_answered = True
                question_answer[question['raw_question']] = sent
                break
        if not is_answered:
            unanswered.append(question['raw_question'])

    print(f'Mean reciprocal rank = {sum(reciprocal_ranks)/len(test_qs)}')
    print(f'Percentile ranks: 50p:{np.percentile(ranks, 50)}, '
          f'75p:{np.percentile(ranks, 75)}, 90p:{np.percentile(ranks, 90)}')
    pprint.pprint(f'Unanswered questions: {unanswered}')
    pprint.pprint(f'Answered questions/answers: {question_answer}')

    # plt.hist(ranks, bins=100)
    # plt.ylabel('freq')
    # plt.xlabel('reciprocal ranks')
    # plt.title("distribution of rank of relevant answer")
    # plt.savefig('rank.png')
