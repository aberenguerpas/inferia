import requests
import faiss
import pickle

def getEmbeddings(data):
    
    response = requests.post('http://localhost:5000/getEmbeddings', json = {"data":data})

    if response.status_code == 200:
        return response.json()['emb']
    elif response.status_code == 404:
        print('Error getting embedding', response.status_cod )

def loadIndex(filename):
    f = open(filename, 'rb')

    reader = faiss.PyCallbackIOReader(f.read, 1234)
    reader = faiss.BufferedIOReader(reader, 1234)

    index = faiss.read_index(reader)

    return index

def loadInversedIndex(path):
    with open(path+'.pickle', 'rb') as handle:
        b = pickle.load(handle)

    return b