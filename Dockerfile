# Dockerfile
FROM python:3.10-slim

# התקנת תלות למודלים ו־uvicorn
RUN apt-get update && apt-get install -y gcc

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# הפורט ש-Azure מאזינה עליו
ENV PORT=8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
