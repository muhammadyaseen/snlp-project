import re

data_root = "../data/"
ans_patterns = data_root + "patterns.txt"
test_questions = data_root + "test_questions.txt"
trec_corpus_xml = data_root + "trec_documents.xml"

question_extraction_pattern = "Number: (\d+) *\n\n\<desc\> Description\:\n(\w+.*)\n\n\<\/top>"

def get_test_questions(test_questions):
    
    questions_doc = open(test_questions).read()

    question_extraction_pattern = "^\<num\> Number: (\d+) *\n\n\<desc\> Description\:\n(\w+.*)\n\n\<\/top>$"
    result = re.findall(question_extraction_pattern, questions_doc, re.MULTILINE)

    for r in result:
        print (r[0], " : " + r[1])

def get_answer_patterns(ans_patterns):
    pass

def process_trec_xml(trec_corpus_xml):
    pass




if __name__ == "__main__":
    
    get_test_questions(test_questions)

    print("Oh boi! I'm up!")
