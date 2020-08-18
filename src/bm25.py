from rank_bm25 import BM25Okapi

from src.util import process_trec_xml, preprocess
from gensim.summarization.bm25 import BM25


def calculate_bm25(corpus, query):
    bm25 = BM25Okapi(corpus)
    tokenized_query = preprocess(query)
    doc_scores = bm25.get_scores(tokenized_query)
    return doc_scores

def cal_bm25_gensim(corpus, query):
    bm25 = BM25(corpus)




if __name__ == '__main__':
    corpus = process_trec_xml(save=True)
    query = "enamored of the Disney Channel movie Back to Hannibal. At first I was a bit apprehensive at the " \
            "idea of seeing Tom Sawyer,  Huckleberry Finn and Mark Twain's oth"
    print(calculate_bm25(corpus, query))

