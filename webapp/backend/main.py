from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and tokenizer (adjust path as needed)
MODEL_PATH = "../../models/gpt2-finetuned-europepmc"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
model.eval()

def generate_response(prompt: str, max_length: int = 128) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

class InferenceRequest(BaseModel):
    prompt: str
    max_length: int = 128

class InferenceResponse(BaseModel):
    result: str

@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    result = generate_response(request.prompt, request.max_length)
    return {"result": result}

@app.get("/")
async def root():
    return {"message": "Model inference API is running."}
