from transformers import RobertaTokenizerFast, AutoModel
import torch.nn as nn
import torch

class Model:
    def __init__(self, name):
        print(torch.backends.mps.is_available())
        print(torch.backends.mps.is_built())
        print("Loading model...", end="")

        self.device = torch.device("cpu") #torch.device("mps" if torch.backends.mps.is_available() else "cpu")


        self.name = name

        self.tokenizer = RobertaTokenizerFast.from_pretrained(name)

        self.model = AutoModel.from_pretrained(name)

        print('Ok')

    def compareTables(self, t1, t2):
        try:
            # Tokenize tables
            table1 = self.tokenizer(
                        text=t1,
                        return_tensors="pt"
            ).to(self.device)

            table2 = self.tokenizer(
                        text=t2,
                        return_tensors="pt"
            ).to(self.device)

            self.model = self.model.to(self.device)

            # Obtain embeddings
            if 'tapex' in self.name:
                emb_t1 = self.model(**table1, output_hidden_states=True).encoder_last_hidden_state[0][0]
                emb_t2 = self.model(**table2, output_hidden_states=True).encoder_last_hidden_state[0][0]
            else:
                emb_t1 = self.model(**table1).last_hidden_state[0][0]
                emb_t2 = self.model(**table2).last_hidden_state[0][0]

            # Define similarity method
            cos = nn.CosineSimilarity(dim=0, eps=1e-6)

            return cos(emb_t1, emb_t2)
            
        except Exception as e:
            print('Error comparing tables')
            print(e)
            return False

    def getEmbedding(self, text):
        tab = self.tokenizer(
                text,
                return_tensors="pt"
        ).to(self.device)

        self.model = self.model.to(self.device)
        emb = self.model(**tab).last_hidden_state[0][0]

        return emb.tolist()
