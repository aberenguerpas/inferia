import argparse
import time
import os
from tqdm import tqdm
import pandas as pd
from utils import *
import numpy as np
import faiss

def save_result(id, rank, path_result):
    df = pd.DataFrame.from_records(rank, columns=['document_id', 'score'])
    df.reset_index(inplace=True)
    df = df.rename(columns = {'index':'rank'})
    df['query_id'] = id
    df['Q0'] = 'Q0'
    df['STANDARD'] = 'STANDARD'
    df['score'] = df['score'].round(3)

    # Sort columns
    new_cols = ["query_id","Q0","document_id","rank","score","STANDARD"]
    df=df[new_cols]
    df=df.reindex(columns=new_cols)

    df.to_csv(path_result, mode='a', index=False, header=False, sep="\t")

def getScore(id, results_h, results_c, table_size):
    
    result_c = list(filter(lambda d: d[0]==id, results_c))[:table_size]
        
    result_c = np.array([s[1] for s in result_c])

    score_c = np.sum(result_c)/table_size
  
    score_h = list(filter(lambda d: d[0]==id, results_h))


    if len(score_h)>0:
        score_h = score_h[0][1]
    else:
        score_h = 0

    score = score_c*0.5 + score_h*0.5

    return score


def search(embs, index_h, index_c, inverted, file):

    k = 2000 # n results

    ids_list = []

    # Header search
    h_emb = embs['header']

    while len(h_emb) == 1:
        h_emb = h_emb[0]
    
    h_emb = np.array([h_emb]).astype(np.float32)

    faiss.normalize_L2(h_emb)

    distances_h, indices_h = index_h.search(h_emb, 200)

    results_h = [(inverted[r], distances_h[0][i]) for i, r in enumerate(indices_h[0])]
   
    ids_list += [k for k,_ in results_h]

    # Content search
    results_c = []
    for col in embs['columns']:
        c_emb = embs['columns'][col]
        while len(c_emb) == 1:
            c_emb = c_emb[0]

        c_emb = np.array([c_emb]).astype(np.float32)
        faiss.normalize_L2(c_emb)

        distances_c, indices_c = index_c.search(c_emb, k)
        results_c+=[(inverted[r], distances_c[0][i]) for i, r in enumerate(indices_c[0])]
        ids_list+=[k for k,v in results_c]

    ids_list = list(set(ids_list)) # List with candidate tables id


    # Ranking tablas
    ranking = dict()
    table_size = len(embs['columns']) # Columns number

    for id in ids_list:
        ranking[id]= getScore(id, results_h, results_c, table_size)
        
    # Ordenar ranking
    ranking_sort = sorted(ranking.items(), key=lambda x: x[1], reverse=True)
   
    ranking_sort = list(filter(lambda d: d[1]>0.75,  ranking_sort))
    ranking_sort = list(filter(lambda d: d[0] != file, ranking_sort)) # Se quita la propia query-table

    return ranking_sort[:10]

def create_embeddings(table):

    embeddings = dict()

    # header embeddings
    headers = table.columns.values
    headers = filter(lambda col: 'Unnamed' not in col, headers) # Skip unnamed column
    headers_text = ' '.join(map(str, headers))
    
    embeddings['header'] =  getEmbeddings(headers_text)
    # column embeddings
    embeddings['columns'] = dict()

    for column in table.columns.values:

        try:
            # Lowercase the text because the sentence model is uncased
            text = ' '.join(table[column].to_list()).lower()
        except TypeError:
            text = ' '.join(table[column].apply(str).to_list()).lower()

        # The maximum token length admitted by SentenceBERT is 256
        max_sequence_length = 256
        # Larger texts are cut into 256 token length pieces and their vectors are averaged
        # Take into account that a cell could have more than one token
        
        if len(text.split()) > max_sequence_length:
            list_tokens = text.split()
            list_vectors = []
            for i in range(0, len(list_tokens), max_sequence_length):
                list_vectors.append(getEmbeddings(' '.join(list_tokens[i:i+max_sequence_length])))
            embeddings['columns'][column] = np.mean(list_vectors, axis=0).tolist()
        else:
            embeddings['columns'][column] = getEmbeddings(text)
    
    return embeddings


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser(description='Search in wikitables')
    parser.add_argument('-i', '--input', default='experiments/data/benchmarks/table/queries.txt', help='Name of the input folder storing CSV tables')
    parser.add_argument('-d', '--data', default='experiments/data/wikitables_clean', help='Data directory')
    parser.add_argument('-n', '--indexDir', default='services/indexation/indexData', help='Inv')
    parser.add_argument('-m', '--model', default='brt', choices=['stb', 'apn','brt'],
                        help='"stb" (Sentence-BERT), "apn" (Allmpnet) or "brt" (BERT)')
    parser.add_argument('-p', '--percent', default='100', help='Content percentage index')
    parser.add_argument('-r', '--result', default='search_result/results', help='Name of the output folder that stores the search results')
    
    args = parser.parse_args()

    # remove old result files
    try:
        if os.path.exists(args.result+'_'+args.model+'.csv'):
            os.remove(args.result+'_'+args.model+'.csv')
        
        dir = "./"+"/".join(args.result.split("/")[:-1])

        if not os.path.exists(dir):
            os.makedirs(dir)

    except Exception as e:
        print(e)

    # Headers collection
    index_headers = loadIndex(os.path.join(args.indexDir, args.model+'_headers.faiss'))

    # Content collection
    index_content = loadIndex(os.path.join(args.indexDir, args.model+'_content.faiss'))

    # Read input file
    queries = pd.read_csv(args.input, sep="\t", header=None)
    files = queries[1].values

    # Read inversed Index
    inverted = loadInversedIndex(os.path.join(args.indexDir, args.model+'_invertedIndex'))

    # Read table
    for path in tqdm(files):
        #load table
        file = open(os.path.join(args.data, path)+'.csv')
        id = queries[queries[1]==path].iloc[:,0].values[0]
        table = pd.read_csv(file)

        # create embeddings
        embs = create_embeddings(table)

        # search
        rank = search(embs, index_headers, index_content, inverted, path)

        # save result
        save_result(id, rank, args.result+'_'+args.model+'.csv')
    
    print('Search time: ' + str(round(time.time() - start_time, 2)) + ' seconds')


if __name__ == "__main__":
    main()
