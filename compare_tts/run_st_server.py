from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import os 

os.environ["TOKENIZERS_PARALLELISM"] = "false"
app = FastAPI()

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('pritamdeka/S-PubMedBert-MS-MARCO', device_map='auto')
model = AutoModel.from_pretrained('pritamdeka/S-PubMedBert-MS-MARCO', device_map='auto')
# tokenizer = AutoTokenizer.from_pretrained('MohammadKhodadad/MedTE-cl15-step-8000', device_map='auto')
# model = AutoModel.from_pretrained('MohammadKhodadad/MedTE-cl15-step-8000', device_map='auto')

# Pooling function
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Request schema
class EmbeddingRequest(BaseModel):
    sentences: list[str]

@app.post("/embed")
def embed(req: EmbeddingRequest):
    
    #encoded_input = tokenizer(req.sentences, padding=True, truncation=True, return_tensors='pt', max_length=512).to('cuda')
    
    encoded_input = tokenizer(req.sentences, padding=True, truncation=True, return_tensors='pt', max_length=512)
    # Use CPU or MPS (Apple Silicon)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    encoded_input = encoded_input.to(device)

    model.to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return {"embeddings": sentence_embeddings.tolist()}
