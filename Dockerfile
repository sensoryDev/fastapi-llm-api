# שימוש בתמונה של Python
FROM python:3.9-slim

# הגדרת ספריית עבודה
WORKDIR /app

# התקנת התלויות
COPY requirements.txt .
RUN pip install -r requirements.txt

# חשיפת הפורט ש-Render דורש
EXPOSE 10000

# הגדרת הפקודה להריץ את האפליקציה
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
