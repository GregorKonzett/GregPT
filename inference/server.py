import torch
from fastapi import FastAPI
from pydantic import BaseModel

from model.GptModel import GptModel, get_device
from tokenizer.TikTokenTokenizer import TikTokenTokenizer
from weights.GCPStorageWeightLoader import GCPStorageWeightLoader


class InferenceResponse(BaseModel):
    response: str

class InferenceRequest(BaseModel):
    prompt: str

app = FastAPI()

tokenizer = TikTokenTokenizer()
weight_loader = GCPStorageWeightLoader()
gpt = GptModel(vocab_size=tokenizer.get_vocab_size(), tokenizer=tokenizer)
gpt = gpt.to(get_device())
weight_loader.load_checkpoint(gpt, False)

@app.get("/")
def get_inference(req: InferenceRequest) -> InferenceResponse:
    encoded_query = torch.tensor([tokenizer.encode(req.prompt)], dtype=torch.long, device=get_device())
    out = gpt.generate(encoded_query)
    out = tokenizer.decode(out[0].cpu().tolist()).strip()
    response = InferenceResponse(response=out)

    return response
