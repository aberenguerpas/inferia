import requests

def getEmbeddings(data):
    
    response = requests.post('http://localhost:5000/getEmbeddings', json = {"data":data})

    if response.status_code == 200:
        return response.json()['response']
    elif response.status_code == 404:
        print('Error getting embedding', response.status_cod )