import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from model import Model
from utils import *
import numpy as np
app = FastAPI()


@app.post("/compare")
async def compare(request: Request):
    response = await request.json()

    table1 = dictToDataframe(response['table1'])
    table2 = dictToDataframe(response['table2'])

    similarity = model.compareTables(table1, table2).tolist()

    return JSONResponse(content={"similarity":round(similarity, 2)})


@app.post("/getEmbeddings")
async def getEmbeddings(request: Request):
    response = await request.json()
    table = dictToDataframe(response)

    if split_table:
        headers = list(table.columns.values)
        headers = " ".join(headers)
        headers_emb = model.getEmbedding(headers)

        content_emb = dict()
        for col in table.columns.values:
            col_content = table[col].tolist() 
            col_content = " ".join(col_content)

            aux_avg = np.empty((0, 768))
            for i in range(0,len(col_content),512):
                aux_avg = np.append(aux_avg, [model.getEmbedding(col_content[i:i+512])], axis=0)

            content_emb[col] = np.mean(aux_avg, axis=0).tolist()


    res = {
        "table":{
            "headers_embedding": headers_emb,
            "content_embedding": content_emb
        }
    }

    return JSONResponse(content=res)


if __name__ == "__main__":
    model = Model("roberta-base")

    split_table = True # El modelo procesa la tabla entera o va por partes?

    uvicorn.run(app, debug=True, host="0.0.0.0", port=5000)
