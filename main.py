from fastapi import FastAPI, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn

# יצירת אפליקציה של FastAPI
app = FastAPI()

# טוען את המודל והטוקניזר רק פעם אחת
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# הגדרת פורט מקסימלי לפרומפטים
MAX_INPUT_LENGTH = 512  # מספר הטוקנים המקסימלי

@app.post("/generate/")
async def generate_text(prompt: str):
    # טוקניזציה של הפרומפט
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=MAX_INPUT_LENGTH)

    # ביצוע הפעלת המודל
    try:
        outputs = model.generate(inputs["input_ids"], max_length=MAX_INPUT_LENGTH + 50, do_sample=True, temperature=0.7)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")

    # המרת הפלט חזרה לטקסט
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {"generated_text": result}

# האזנה על פורט 10000 (או כל פורט אחר ש-Render מציינת)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)  # כאן הגדרנו את הפורט להיות 10000
