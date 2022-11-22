from sentence_transformers import SentenceTransformer

class Allmpnet:

    def __init__(self):
        self.model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v3_mpnet-base')
        self.tokenizer = None
        self.dimensions = 768

    def getEmbedding(self, data):
        return self.model.encode(data)