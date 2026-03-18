import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel


class BERTEmbedder(torch.nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model_bert = BertModel.from_pretrained("bert-base-uncased")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_bert = self.model_bert.to(self.device)
        self.model_bert.eval()

        for p in self.model_bert.parameters():
            p.requires_grad = False

    def forward(self, input_text):
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model_bert(**inputs)

        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]

        mask = attention_mask.unsqueeze(-1).float()
        sentence_embeddings = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        embeddings = F.normalize(sentence_embeddings, p=2, dim=-1)
        return embeddings
