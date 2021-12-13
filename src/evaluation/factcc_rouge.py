import pandas
import os
import sys
from rouge_metric import PyRouge
from datasets import load_dataset
import json
import pickle
import random

global hypotheses
global references

hypotheses = []
references = []

def read_text_file(file_path):
    text_file = ""
    with open(file_path, 'rb') as f:
        text_file += str(f.read())
    
    return text_file

def get_docs(data):
    documents = {}
    for d in data:
        documents[d['id']] = [d['document'], d['summary']]
    
    return documents
        

def eval_summaries(documents, file, corrupt):
    random.seed(2021)
    xsum_train = load_dataset("xsum", split="train")
    print(xsum_train.column_names)
    pickle_off = open(file,"rb")
    docs = pickle.load(pickle_off)
    ids = list(docs.keys())
    ids = random.sample(ids, 100)
    print(ids)
    file_to_write = ""
    
    counter = 0
    for id in ids:
        row_dict = dict()
        print(str(id) + ": " + docs[id][0])
        print(documents[id][1])
        row_dict['text'] = documents[id][0]
        if corrupt:
            #row_dict['claim'] = docs[id]
            row_dict['claim'] = docs[id][0]
        else:
            row_dict['claim'] = documents[id][1]
        hypotheses.append(docs[id][1])
        references.append([documents[id][1]])
        row_dict['label'] = 'CORRECT'
        file_to_write += json.dumps(row_dict) + "\n"
    
    return file_to_write
    

    

if __name__ == '__main__':
    xsum_test = load_dataset("xsum", split="test")
    test_documents = get_docs(xsum_test)
    test_ids = xsum_test['id']
    random.seed(2021)
    
    
    # for writing to factcc
    file_to_write = eval_summaries(test_documents, "test_smart_entity.pkl", True)
    file_to_write += eval_summaries(test_documents, "test_predicate_entity.pkl", True)
    file_to_write += eval_summaries(test_documents, "test_s_o_entity.pkl", True)
    
    '''
    f = open('./factcc-data/data-dev.jsonl', "w")
    f.write(str(file_to_write))
    f.close()
    '''
    
    
    
    print("Rouge")
    # Evaluate document-wise ROUGE scores
    rouge = PyRouge(rouge_n=(1, 2, 4), rouge_l=True, skip_gap=4)
    scores = rouge.evaluate(hypotheses, references)

    print(scores)
    
    


