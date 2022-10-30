from pymilvus import connections, Collection
import argparse
import time
import os
from tqdm import tqdm
import pandas as pd
from subprocess import Popen, PIPE
from utils import *
import numpy as np

def save_result(id, rank, path_result):
    df = pd.DataFrame.from_records(rank, columns=['document_id', 'score'])
    df.reset_index(inplace=True)
    df = df.rename(columns = {'index':'rank'})
    df['query_id'] = id + 1
    df['Q0'] = 'Q0'
    df['STANDARD'] = 'STANDARD'
    df['score'] = df['score'].round(3)

    # Sort columns
    new_cols = ["query_id","Q0","document_id","rank","score","STANDARD"]
    df=df[new_cols]
    df=df.reindex(columns=new_cols)

    df.to_csv(path_result, mode='a', index=False, header=False, sep="\t")

def getScore(id, results_h, results_c, table_size):

    result_c = list(filter(lambda d: d.id==id, results_c))[:table_size]
    result_c =np.array([s.distance for s in result_c])
    score_c = np.sum(result_c)/table_size

    score_h = list(filter(lambda d: d.id==id, results_h))
    if len(score_h)>0:
        score_h = score_h[0].distance
    else:
        score_h = 0

    score = score_c*0.5 + score_h*0.5

    return score


def search(embs, collection_h, collection_c):

    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    ids_list = []

    # Header search
    results_h = collection_h.search(
        data=[embs['header']], 
        anns_field="data", 
        param=search_params, 
        limit=10, 
        expr=None,
        round_decimal=3,
        consistency_level="Strong"
    )

    ids_list+=results_h[0].ids

    # Content search
    results_c = []
    for col in embs['columns']:
       
        results = collection_c.search(
            data=[embs['columns'][col]], 
            anns_field="data", 
            param=search_params, 
            limit=10, 
            expr=None,
            round_decimal = 3,
            output_fields = ["data_desc"],
            consistency_level = "Strong"
        )
        results_c+=results[0]
        ids_list+=results[0].ids

 
    ids_list = list(set(ids_list)) # List with candidate tables id

    # Ranking tablas
    ranking = dict()
    table_size = len(embs['columns']) # Columns number
    for id in ids_list:
        ranking[id]= getScore(id, results_h[0], results_c, table_size)

    # Ordenar ranking
    ranking_sort = sorted(ranking.items(), key=lambda x: x[1], reverse=True)    

    return ranking_sort

def create_embeddings(table):

    embeddings = dict()

    # header embeddings
    text =" ".join(map(str,table.columns.values)).lower()
    embeddings['header'] =  getEmbeddings(text)

    # column embeddings
    embeddings['columns'] = dict()

    for column in table.columns.values:
        text = " ".join(map(str,table[column].values)).lower()
        # The maximum token length admitted by SentenceBERT is 256
        max_sequence_length = 256
        # Larger texts are cut into 256 token length pieces and their vectors are averaged
        # Take into account that a cell could have more than one token
        list_tokens = text.split()
        if len(list_tokens) > max_sequence_length:
            list_vectors = []
            for i in range(0, column.size, max_sequence_length):
                list_vectors.append(getEmbeddings(' '.join(list_tokens[i:i+max_sequence_length])))
                embeddings['columns'][column]= np.mean(list_vectors, axis=0).tolist()
        else:
            embeddings['columns'][column]=getEmbeddings(text)
    
    return embeddings


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser(description='Search in wikitables')
    parser.add_argument('-i', '--input', default='experiments/data/benchmarks/table/queries.txt', help='Name of the input folder storing CSV tables')
    parser.add_argument('-d', '--data', default='experiments/data/wikitables_clean', help='Data directory')
    parser.add_argument('-m', '--model', default='stb', choices=['w2v', 'ftw', 'fts', 'stb', 'rbt','brt'],
                        help='Model to use: "w2v" (Word2Vec), "ftw" (fastText word),"fts" (fastText subword), "stb" (Sentence-BERT), "rbt" (ROBERTA) or "brt" (BERT)')
    parser.add_argument('-p', '--percent', default='90', help='Content percentage index')
    parser.add_argument('-r', '--result', default='./search_result/results.csv', help='Name of the output folder that stores the search results')
    
    args = parser.parse_args()

    # remove old result files
    os.remove(args.result)

    # start microservice embeddings
    #Popen(['env/bin/python','services/embeddings/main.py', '-m', args.model], stdin=PIPE, stdout=PIPE)
    #time.sleep(10)

    # connect to milvus server
    connections.connect(
        alias="default", 
        host='localhost', 
        port='19530'
    )

    # load the collections

    # Headers collection
    collection_h = Collection(args.model+"_headers")
    #collection_h.load()

    # Content collection
    collection_c = Collection(args.model+"_content_"+args.percent)
    #collection_c.load()

    # Read input file
    queries = pd.read_csv(args.input, sep="\t", header=None)
    files = queries[1].values
    # Read table

    for i, path in enumerate(tqdm(files)):
        #load table
        file = open(os.path.join(args.data, path)+'.csv')

        table = pd.read_csv(file)

        # create embeddings
        embs = create_embeddings(table)

        # search
        rank = search(embs, collection_h, collection_c)

        # save result
        save_result(i, rank, args.result)

    end_time = time.time()
    print('Search time: ' + str(round(end_time - start_time, 2)) + ' seconds')


if __name__ == "__main__":
    main()
