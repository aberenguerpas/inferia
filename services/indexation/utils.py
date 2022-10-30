import pandas as pd
import requests
import numpy as np

def getIDFromWiki(tabName):

    tabName = tabName.split('-')
    tabName = int(tabName[1] + tabName[2])

    return tabName

def getTableFromWiki(tab):

    df = pd.DataFrame(columns=tab['title'])
    
    for row in tab['data']:
        df.loc[len(df.index)] = row

    return df.to_dict(orient='list')


def getEmbeddings(table):

    response = requests.post('http://localhost:5000/getEmbeddings', json = table)

    if response.status_code == 200:
        return response.json()
    elif response.status_code == 404:
        print('Error getting embedding', response.status_cod )

def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0: 
            return v
        return v / norm