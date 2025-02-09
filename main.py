import os
import aiohttp
import asyncio
import logging
import hashlib
from collections import Counter
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from docx import Document
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from TTS.api import TTS
import nltk
import gdown
import uvicorn

# Download necessary NLTK data
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Directories (Using Railway Volume for persistence)
BASE_DIR = "/data"
MODEL_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_DIR = os.path.join(BASE_DIR, "data")
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
VIDEO_DIR = os.path.join(BASE_DIR, "videos")
CACHE_DIR = os.path.join(BASE_DIR, "cache")

for directory in [MODEL_DIR, UPLOAD_DIR, AUDIO_DIR, VIDEO_DIR, CACHE_DIR]:
    os.makedirs(directory, exist_ok=True)

# Google Drive Model File IDs
MODEL_FILES = {
    "tacotron2-DDC.pth": "17XArguEHT4BD84VQt_S-W5KdE_CyzIiP",  # Changed to hyphen
    "vits_model.pth": "14EzYU1_scEItpxO4_IAk6g1_IMSO7t6-",
    "hifigan_v2.pth": "1R_FCiBo_E1N1xvrqf15wyBcWGFOtiRou",
}

models_checked = False

def download_models():
    global models_checked
    if models_checked:
        return

    for filename, file_id in MODEL_FILES.items():
        model_path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(model_path):
            logging.info(f"✅ {filename} already exists, skipping download.")
            continue
        logging.info(f"⬇️ Downloading {filename} from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
        logging.info(f"✅ {filename} downloaded successfully!")
    
    models_checked = True

# Ensure models exist before running TTS
download_models()

# Initialize TTS with explicit model path
tts_model_path = os.path.join(MODEL_DIR, "tacotron2-DDC.pth")
coqui_tts = TTS(model_path=tts_model_path, progress_bar=False)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rest of the code remains unchanged...
# [Keep all other functions and routes the same as original]

@app.get("/")
def root():
    return {"message": "🚀 API is running successfully!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))