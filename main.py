from models import BERTEmbedder, OpenClipVitEmbedder
import requests
from PIL import Image
from io import BytesIO
import torch

bert_embedder = BERTEmbedder()
clip_embedder = OpenClipVitEmbedder()

response = requests.get(
    "https://raw.githubusercontent.com/mikolalysenko/lena/refs/heads/master/lena.png"
)
cimage = Image.open(BytesIO(response.content)).convert("RGB")

image_tensor = clip_embedder.preprocess(cimage).unsqueeze(0)

with torch.no_grad():
    text_embedding = bert_embedder("Hello gng")
    image_embedding = clip_embedder(image_tensor)

print(text_embedding.shape)   # (1, 768)
print(image_embedding.shape)  # (1, 512)
