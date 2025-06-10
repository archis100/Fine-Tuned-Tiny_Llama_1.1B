from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch

app = FastAPI()

# Load base model + LoRA 
base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # hugging face model id
adapter_path = "./Tiny-Llama-1.1B-chat-finetune"     # path to the adapter folder

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
base_model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto", torch_dtype=torch.float16)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, adapter_path, device_map="auto", offload_folder="/app/offload")

# Create generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

class PromptInput(BaseModel):
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.7

@app.post("/infer")
async def infer(input_data: PromptInput):
    try:
        result = generator(
            input_data.prompt,
            max_new_tokens=input_data.max_new_tokens,
            temperature=input_data.temperature,
            return_full_text=False
        )
        return {
            "response": result[0]["generated_text"],
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
