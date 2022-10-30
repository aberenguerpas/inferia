from transformers import RobertaTokenizer, RobertaModel
import torch

class Roberta:

    def __init__(self):
        self.model = RobertaModel.from_pretrained("roberta-base")
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def getEmbedding(self, data):
        tab = self.tokenizer(
                data,
                truncation=True,
                return_tensors="pt"
        ).to(self.device)

        self.model = self.model.to(self.device)
        emb = self.model(**tab).last_hidden_state[0][0]
        return emb