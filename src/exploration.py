
import pprint
import re
from collections import Counter
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np

from src import baseline, util
from src.util import test_questions, ans_patterns, get_test_questions


def relevant_per_query(corpus, queries):
    # init rel counts with zero per query
    from collections import defaultdict

    rel_counts = [0 for i in range(len(queries))]
    rel_anspat = defaultdict(int)
    # check if doc is relevant wrt any of the answer patterns
    for doc in corpus:
        for q in queries:
            # print(q)
            for ap in queries[q]['ans_patterns']:
                rel_anspat[ap.strip()] += bool(re.search(ap.strip(), corpus[doc], flags=re.IGNORECASE))
                rel_counts[q - 1] += bool(re.search(ap.strip(), corpus[doc], flags=re.IGNORECASE))

    return rel_counts, rel_anspat

def count_answer_per_question(corpus, queries):
    query_answers = {}
    for q in queries:
        c = 0
        ans_pattern = "|".join(test_qs[q]['ans_patterns'])
        for doc in corpus:
            if re.search(ans_pattern, doc):
                c = c + 1
        query_answers[str(q)+'. '+queries[q]['raw_question']] = c
    pprint.pprint(Counter(query_answers).most_common(20))
    return query_answers


test_qs = get_test_questions()
corpus = util.get_stemmed_corpus(baseline.trec_corpus_xml)
# relevant_per_query(corpus, test_qs)
c = count_answer_per_question([x['text'] for x in corpus] , test_qs)
x = [v for k,v in c.items()]

print("50th percentile of arr : ",
       np.percentile(x, 50))
print("25th percentile of arr : ",
       np.percentile(x, 90))
print("75th percentile of arr : ",
       np.percentile(x, 75))

plt.hist(x, bins=100)  # `density=False` would make counts
plt.ylabel('freq')
plt.xlabel('no of relevant docs')
plt.title("distribution of relevant docs for queries")
plt.savefig('query-docs.png')
