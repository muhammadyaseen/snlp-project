import concurrent
import os
import pickle
import re
import string

import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
# TODO: Uncomment
# nltk.download('punkt')
# nltk.download('stopwords')
from rank_bm25 import BM25Okapi

data_root = "../data/"
ans_patterns = data_root + "patterns.txt"
test_questions = data_root + "test_questions.txt"
trec_corpus_xml = data_root + "trec_documents.xml"

processed_root = data_root + "processed/"
os.makedirs(processed_root, exist_ok=True)

processed_corpus = processed_root + "stemmed_corpus.pkl"
processed_text_qs = processed_root + "test_qs.pkl"
# processed_tfids = processed_root + "tfids.pkl"
# processed_tfidf_repr = processed_root + "tfrepr.pkl"

stemmer = SnowballStemmer("english")
stopwordslist = stopwords.words()


def get_test_questions(save=False):
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


def clean(text):
    text = text.replace("'s","")
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


def tokenize(text):
    text = clean(text)
    tokens = nltk.word_tokenize(text)
    words = list(filter(lambda token: token not in string.punctuation, tokens))

    return words


def preprocess(text):
    tokens = tokenize(text.lower())

    tokens_without_sw = [word for word in tokens if word not in stopwordslist]

    stemmed = [stemmer.stem(word) for word in tokens_without_sw]

    return stemmed


def get_stemmed_corpus(trec_corpus_xml=trec_corpus_xml, save=False):
    try:
        print("Loading from saved pickle")
        corpus = pickle.load(open(processed_corpus, "rb"))
        return corpus

    except Exception as e:

        print("Data doesn't exit or other error", e)
        print("Processing from scratch")

    with open(trec_corpus_xml, 'r') as dh:
        # TODO: Change to html parser for portability
        # soup = BeautifulSoup(dh, 'html.parser')
        soup = BeautifulSoup(dh, 'lxml')
        article_texts = soup.find_all('doc')
        print("Found %d articles..." % len(article_texts))

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            results = list(executor.map(process_doc, article_texts))
            corpus = list(filter(None.__ne__, results))

    if save:
        print("saving processed corpus, existing data will be overwritten")
        pickle.dump(
            corpus,
            open(processed_corpus, "wb")
        )

    return corpus


def process_doc(a):
    try:
        doc_id = a.docno.get_text(strip=True).lower()
        print(f'Processing doc_id: {doc_id}')
        if a.headline:
            headline = a.headline.get_text(strip=True)
        else:
            headline = ''
        text = a.find('text').get_text(strip=True)
        doc = {'headline': headline, 'text': text, 'tokens': preprocess(text)}
        return doc

    except Exception as e:
        print(f'Error processing {doc_id}:  {e}')
        return None


def precision_at_r(docs, ans_pattern, r=50):
    relevant_docs = 0
    for doc in docs[:r]:
        if bool(re.search(ans_pattern, doc['text'], flags=re.IGNORECASE)):
            relevant_docs = relevant_docs + 1

    r_value = relevant_docs / len(docs)
    return r_value


if __name__ == '__main__':
    corpus = get_stemmed_corpus(trec_corpus_xml, save=True)
