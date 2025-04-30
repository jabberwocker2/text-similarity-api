from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertModel
import numpy as np


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def prepare_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
    return cls_embedding

def calculate_cosine_similarity(embedding1, embedding2):
    embedding1 = embedding1.numpy()
    embedding2 = embedding2.numpy()
    cos_sim = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return cos_sim

app = FastAPI()

class TextPair(BaseModel):
    text1: str
    text2: str

@app.post("/")
async def get_similarity(data: TextPair):
    embedding1 = prepare_embedding(data.text1)
    embedding2 = prepare_embedding(data.text2)
    similarity_score = calculate_cosine_similarity(embedding1, embedding2)
    return {"similarity score": round(float(similarity_score), 4)}

@app.get("/")
def read_root():
    return {"message": "API is live. Use POST with 'text1' and 'text2'."}
