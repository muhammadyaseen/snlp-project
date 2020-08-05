import re

data_root = "../data/"
ans_patterns = data_root + "patterns.txt"
test_questions = data_root + "test_questions.txt"
trec_corpus_xml = data_root + "trec_documents.xml"

question_extraction_pattern = "Number: (\d+) *\n\n\<desc\> Description\:\n(\w+.*)\n\n\<\/top>"

def get_test_questions(test_questions, ans_patterns):
    
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

    return qs

def process_trec_xml(trec_corpus_xml):
    pass




if __name__ == "__main__":
    
    parsed = get_test_questions(test_questions, ans_patterns)

    for p in parsed.values(): print(p)

    print("Oh boi! I'm up!")
