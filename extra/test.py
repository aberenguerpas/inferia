import requests
import numpy as np

def getEmbeddings(data):
    
    response = requests.post('http://localhost:5000/getEmbeddings', json = {"data":data})

    if response.status_code == 200:
        return response.json()['emb']
    elif response.status_code == 404:
        print('Error getting embedding', response.status_cod )


t1 = """Quis commodo in Lorem et ipsum minim. Consectetur sunt esse do est exercitation. Elit dolor ipsum aliquip aliqua nisi elit est minim esse. Quis commodo in Lorem et ipsum minim. Consectetur sunt esse do est exercitation. Elit dolor ipsum aliquip aliqua nisi elit est minim esse. Quis commodo in Lorem et ipsum minim. Consectetur sunt esse do est exercitation. Elit dolor ipsum aliquip aliqua nisi elit est minim esse. Quis commodo in Lorem et ipsum minim. Consectetur sunt esse do est exercitation. Elit dolor ipsum aliquip aliqua nisi elit est minim esse. Quis commodo in Lorem et ipsum minim. Consectetur sunt esse do est exercitation. Elit dolor ipsum aliquip aliqua nisi elit est minim esse.
    sdadasd
    Quis commodo in Lorem et ipsum minim. Consectetur est exercitation. Elit dolor ipsum aliquip aliqua nisi elit est minim esse. Quis commodo in Lorem et ipsum minim. Consectetur sunt esse do est exercitation. Elit dolor ipsum aliquip aliqua nisi elit est minim esse. Quis commodo in Lorem et ipsum minim. Consectetur sunt esse do est exercitation. Elit dolor ipsum aliquip aliqua nisi elit est minim esse. Quis commodo in Lorem et ipsum minim. Consectetur sunt esse do est exercitation. Elit dolor ipsum aliquip aliqua nisi elit est minim esse. Quis commodo in Lorem et ipsum minim. Consectetur sunt esse do est exercitation. Elit dolor ipsum aliquip aliqua nisi elit est minim esse.


    """
emb = getEmbeddings(['cosita linda mi amor', 'we'])

print(emb)
#print(emb[1])
#print(emb[2])

