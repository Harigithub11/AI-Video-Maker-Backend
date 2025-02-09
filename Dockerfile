 
FROM python:3.9

# Set the working directory
WORKDIR /app

# Install system dependencies for Coqui TTS
RUN apt-get update && apt-get install -y \
    espeak \
    libespeak1 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy all files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
