from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# יצירת אפליקציה של FastAPI
app = FastAPI()

# שם המודל
model_name = "distilgpt2"

# טעינת הטוקנייזר והמודל
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# בקשה שמכילה פרומפט
class PromptRequest(BaseModel):
    prompt: str

# נקודת קצה שמחזירה טקסט מהמודל
@app.post("/chat")
def chat(req: PromptRequest):
    try:
        inputs = tokenizer(req.prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")
