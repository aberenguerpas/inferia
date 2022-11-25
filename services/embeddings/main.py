import argparse
import uvicorn
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from SentenceBert import SentenceBert
from Bert import Bert
from Allmpnet import Allmpnet
from FastText import FastText

app = FastAPI()

def checkGPU():
    if torch.cuda.is_available():
        print("Cuda GPU available")
        print(torch.cuda.device_count(), "Devices")
        print("Current device:")
        print(torch.cuda.get_device_name(0))

    elif torch.backends.mps.is_available():
        print("Mac M1 acceleration available!")


@app.post("/getEmbeddings")
async def getEmbeddings(request: Request):
    response = await request.json()
    data = model.getEmbedding(response['data'])
    data = [l.tolist() for l in data]
    return JSONResponse(content={'emb':data})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Embeddings microservice')
    parser.add_argument('-m', '--model', default='stb', choices=['stb','brt','apn','fst'],
                        help='Model to use: "stb" (Sentence-Bert, by default), "brt" (bert-base-uncased),'
                             ' "apn" (Allmpnet) "fst" (fastText)', )
    args = parser.parse_args()

    checkGPU()

    if args.model == 'stb':
        model = SentenceBert()
    elif args.model == 'brt':
         model = Bert()
    elif args.model == 'apn':
        model = Allmpnet()
    elif args.model == 'fst':
        model = FastText()
    else:
        model = SentenceBert()

    uvicorn.run(app, host="0.0.0.0", port=5000)
