from transformers import BertTokenizer, BertModel
import torch

class Bert:

    def __init__(self):
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dimensions = 768

    def getEmbedding(self, data):
        tab = self.tokenizer(
                data,
                padding=True,
                return_tensors="pt"
        ).to(self.device)

        self.model = self.model.to(self.device)
        emb = self.model(**tab).last_hidden_state[0][0]
        return emb