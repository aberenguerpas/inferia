from milv import milv
import os
import json
from utils import *
import numpy as np
from tqdm import tqdm
import warnings 

def main():

    # Config
    model_name = "Roberta_base"
    np.set_printoptions(suppress=True)
    warnings.filterwarnings("ignore")

    # CONSTANTS
    DATA_PATH = './experiments/data/wikitables'
    # Prepare index server
    milvus = milv('localhost')

    milvus.createCollection(model_name+"_headers")
    milvus.createCollection(model_name+"_content")
    milvus.loadCollection()

    # Index data

    for path in os.listdir(DATA_PATH):
        file = open(os.path.join(DATA_PATH, path))
        tables = json.load(file)
        for key, table in tqdm(tables.items()):
            try:
                key = getIDFromWiki(key)
                table = getTableFromWiki(table)
                embedding = getEmbeddings(table)['table']
                head_emb = [[key],['headers'], [normalize(np.array(embedding["headers_embedding"]))]]
                milvus.insertData(head_emb, model_name+"_headers")
                for col in embedding['content_embedding']:
                    col_emb = [[key], ['Col'+ col], [normalize(np.array(embedding["content_embedding"][col]))]]
                    milvus.insertData(col_emb, model_name+"_content")

            except Exception as e:
                print(e)
        break

    milvus.buildIndex(model_name+"_headers")
    milvus.buildIndex(model_name+"_content")

    milvus.closeConnection()


if __name__ == "__main__":
    main()
