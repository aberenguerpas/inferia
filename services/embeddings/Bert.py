from transformers import BertTokenizer, BertModel
import torch

class Bert:

    def __init__(self):
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def getEmbedding(self, data):
        tab = self.tokenizer(
                data,
                padding=True,
                truncation=True,
                return_tensors="pt"
        ).to(self.device)

        self.model = self.model.to(self.device)
        emb = self.model(**tab).last_hidden_state[0][0]
        return emb