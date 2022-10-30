from sentence_transformers import SentenceTransformer

class SentenceBert:

    def __init__(self):
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.tokenizer = None

    def getEmbedding(self, data):
        return self.model.encode(data)
