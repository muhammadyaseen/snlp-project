import re
import pickle
import math
from bs4 import BeautifulSoup

data_root = "../data/"
ans_patterns = data_root + "patterns.txt"
test_questions = data_root + "test_questions.txt"
trec_corpus_xml = data_root + "trec_documents.xml"

processed_corpus = data_root + "processed/corpus.pkl"
processed_text_qs = data_root + "processed/test_qs.pkl"

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
            qs[int(q[0])] = {'question': q[1], 'ans_patterns': []}

        # get associated answer patterns
        ans_doc = open(ans_patterns).readlines()

        for ap in ans_doc:
            #print(ap)
            ap = ap.split(" ")
            id, pattern = ap[0], " ".join(ap[1:])
            qs[int(id)]['ans_patterns'].append(pattern) 

        if save:
            print("saving processed questions, existing data will be overwritten")
            pickle.dump(
                qs, 
                open(processed_text_qs, "wb" )
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

        with open(trec_corpus_xml,'r') as dh:

            soup = BeautifulSoup(dh, 'html.parser')
            article_texts = soup.find_all('doc')
            print("Found %d articles..." % len(article_texts))
            
            for a in article_texts:
                # for now we don't separate byline / headline etc
                corpus[a.docno.get_text().lower()] = a.get_text().lower()
        
        if save:
            print("saving processed corpus, existing data will be overwritten")
            pickle.dump(
                corpus, 
                open(processed_corpus, "wb" )
            )

        return corpus


def compute_term_freqs():
    
    # create representation of all docs in terms of their term freqs

    pass

def compute_term_idfs(corpus, save=False):
    
    term_doc_freq = {}
    N = len(corpus.keys())

    # first we get the document freq of a term 
    # i.e. how many docs contain that term
    # this is upper bounded by num of docs, of course
    for doc in corpus.values():

        # we are interested in just occurence, and not actual freqs
        # that's why we convert the doc to set of non-repeating terms
        terms = set(doc.split(" "))
        
        for term in terms:
            
            if term in term_doc_freq.keys():
                term_doc_freq[term] += 1
            else:
                term_doc_freq[term] = 1

    # now that we have term's df, we inverse it and apply log normalization
    for t in term_doc_freq.keys():
        term_doc_freq[t] = math.log(N/term_doc_freq[t])

    return term_doc_freq


def precision_at_r(r):
    pass

if __name__ == "__main__":
    
    parsed = get_test_questions(test_questions, ans_patterns, save=True)

    #for p in parsed.values(): print(p)

    c = process_trec_xml(trec_corpus_xml, save=True)

    #print(c['ft911-5'])

    tdfs = compute_term_idfs(c)

    print(len(tdfs))
    print(tdfs['the'])
    print(tdfs['of'])
    print(tdfs['is'])
    print(tdfs['pakistan'])
