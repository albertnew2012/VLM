import torch
import clip
from PIL import Image

model, preprocess = clip.load("ViT-B/32")

device = "cpu" if not torch.cuda.is_available() else "cuda"

img = preprocess(Image.open("VLM/dog.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a photo of a cat", "a photo of a dog"]).to(device)

with torch.no_grad():
    img_feat = model.encode_image(img)
    txt_feat = model.encode_text(text)
    logits = img_feat @ txt_feat.T

print(logits)
