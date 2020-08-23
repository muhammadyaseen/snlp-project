import pprint
import re
from collections import Counter
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np

from src import baseline, util
from src.util import test_questions, ans_patterns, get_test_questions


def count_answer_per_question(corpus, queries):
    query_answers = {}
    for q in queries:
        c = 0
        ans_pattern = "|".join(queries[q]['ans_patterns'])
        for doc in corpus:
            if re.search(ans_pattern, doc):
                c = c + 1
        query_answers[str(q) + '. ' + queries[q]['raw_question']] = c
    pprint.pprint(Counter(query_answers).most_common(20))
    return query_answers


def count_answer_per_pattern(corpus, queries):
    pattern_answers = {}
    for q in queries.values():
        for p in q['ans_patterns']:
            c = 0
            for doc in corpus:
                if re.search(p, doc['text']):
                    c = c + 1
        pattern_answers[str(p)] = c
    pprint.pprint(Counter(pattern_answers).most_common(10))
    return pattern_answers


if __name__ == '__main__':
    test_qs = get_test_questions()
    corpus = util.get_corpus(baseline.trec_corpus_xml)
    # relevant_per_query(corpus, test_qs)
    c = count_answer_per_question([x['text'] for x in corpus], test_qs)
    # c = count_answer_per_pattern(corpus, test_qs)
    x = [v for k, v in c.items()]

    print("50th percentile of arr : ",
          np.percentile(x, 50))
    print("25th percentile of arr : ",
          np.percentile(x, 90))
    print("75th percentile of arr : ",
          np.percentile(x, 75))

    plt.hist(x, bins=100)
    plt.ylabel('freq')
    plt.xlabel('no of relevant docs')
    plt.title("distribution of relevant docs for queries")
    plt.savefig('query-docs.png')
