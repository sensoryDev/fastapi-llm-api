from fastapi import FastAPI, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

MAX_INPUT_LENGTH = 512

@app.post("/generate/")
async def generate_text(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=MAX_INPUT_LENGTH)
    try:
        outputs = model.generate(inputs["input_ids"], max_length=MAX_INPUT_LENGTH + 50, do_sample=True, temperature=0.7)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": result}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=True)
