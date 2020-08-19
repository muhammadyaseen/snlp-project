"""
The purpose of this processing script is to convert the SQuAD dataset 
in a similar format to WikiQA dataset so that we can use it with the 
built-in data loader and use it in pairwise ranking setting.
"""
import json

SQUAD_ROOT = "/home/yaseen/Course Work/snlp-project/data/"
SQUAD_DEV = SQUAD_ROOT + "dev-v1.1.json"
SQUAD_TRAIN = SQUAD_ROOT + "train-v1.1.json"

SQUAD_DEV_P = SQUAD_ROOT + "dev-v1.1-processed.json"
SQUAD_DEV_P = SQUAD_ROOT + "train-v1.1-processed.json"


def load_and_process(path=SQUAD_DEV):
    
    data_json = json.load(open(path,'r'))
    # list of pages, where each element has 'title' and 'paragraphs' keys.
    # title is title of Wiki page and, paragraphs are paras from that page 
    # Each title/pages contains list of paragraph objects which consist of 
    # 'context' and 'question-answers' i.e. 'qas'
    pages = data_json['data']

    for wiki_page in pages:

        pg_title = wiki_page['title']
        pg_paras = wiki_page['paragraphs'] # list object - with context and qas keys

        for para in pg_paras:

            print("\n\n")
            print(para['context'])
            print("\n\n")

            for qa in para['qas']:
            
                q = qa['question']
                print("\nQuestion: ", q)

                ans_texts = set([ans_text['text'] for ans_text in qa['answers']])
                print(ans_texts)

            break
        
        break
           
if __name__ == "__main__":

    load_and_process(SQUAD_DEV)
    print("done")
