import pandas as pd
import requests
import numpy as np

def getTableFromWiki(tab):

    df = pd.DataFrame(columns=tab['title'])
    
    for row in tab['data']:
        df.loc[len(df.index)] = row

    return df.to_dict(orient='list')


def getEmbeddings(table):

    response = requests.post('http://localhost:5000/getEmbeddings', json = {'data':table})

    if response.status_code == 200:
        return response.json()['emb']
    elif response.status_code == 404:
        print('Error getting embedding', response.status_cod )

def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0: 
            return v
        return v / norm