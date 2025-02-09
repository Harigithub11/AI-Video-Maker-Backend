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

# API Keys
PIXABAY_API_KEY = "48738698-6ae327a6f8a04d813fa6c6101"
FREESOUND_API_KEY = "YG213ZwV2kJwHtcXAfs3Ik6s0lTJzjwI45WnFA23"

# Google Drive Model File IDs
MODEL_FILES = {
    "tacotron2_DDC.pth": "17XArguEHT4BD84VQt_S-W5KdE_CyzIiP",
    "vits_model.pth": "14EzYU1_scEItpxO4_IAk6g1_IMSO7t6-",
    "hifigan_v2.pth": "1R_FCiBo_E1N1xvrqf15wyBcWGFOtiRou",
}

# Global variable to track if models have been checked
models_checked = False

# Download models only if they don't exist
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
coqui_tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)

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

# Utility function to generate cache paths
def get_cache_path(key, directory):
    return os.path.join(directory, hashlib.md5(key.encode()).hexdigest())

# Download and cache files
async def cached_download(url, directory):
    cache_path = get_cache_path(url, directory)
    if os.path.exists(cache_path):
        logging.info(f"✅ Using cached file: {cache_path}")
        return cache_path
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                content = await response.read()
                with open(cache_path, "wb") as f:
                    f.write(content)
                return cache_path
            logging.error(f"❌ Failed to download {url} - Status {response.status}")
    return None

# Extract text from files
def extract_text(file_path):
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        return "\n".join(para.text for para in doc.paragraphs if para.text.strip())
    return "No readable text found."

# Extract keywords
def extract_keywords(text, num_keywords=3):
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    return [word for word, _ in Counter(filtered_words).most_common(num_keywords)]

# Fetch video from Pixabay
async def fetch_video(keyword):
    url = f"https://pixabay.com/api/videos/?key={PIXABAY_API_KEY}&q={keyword}&per_page=3"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                videos = data.get("hits", [])
                return videos[0]["videos"]["medium"]["url"] if videos else None
    return None

# Fetch background audio from Freesound
async def fetch_background_audio(keyword):
    url = f"https://freesound.org/apiv2/search/text/?query={keyword}&token={FREESOUND_API_KEY}&fields=previews"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                sounds = data.get("results", [])
                return sounds[0]["previews"]["preview-hq-mp3"] if sounds else None
    return None

# Convert text to speech with caching
def text_to_speech(text, filename="output.wav"):
    cache_path = get_cache_path(text, AUDIO_DIR)
    if os.path.exists(cache_path):
        return cache_path
    coqui_tts.tts_to_file(text=text, file_path=cache_path)
    return cache_path

@app.post("/generate_video/")
async def generate_video(text: str = Form(None), file: UploadFile = File(None)):
    logging.info("🎬 Starting video generation process")

    if file:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        text = extract_text(file_path)
        logging.info(f"📄 Extracted text from {file.filename}")

    if not text:
        return JSONResponse(content={"error": "No text provided"}, status_code=400)

    keywords = extract_keywords(text)
    video_url, audio_url = None, None
    for keyword in keywords:
        video_url = video_url or await fetch_video(keyword)
        audio_url = audio_url or await fetch_background_audio(keyword)
        if video_url and audio_url:
            break

    if not video_url or not audio_url:
        return JSONResponse(content={"error": "Could not find video or background audio"}, status_code=500)

    video_path, background_audio_path, speech_audio_path = await asyncio.gather(
        cached_download(video_url, VIDEO_DIR),
        cached_download(audio_url, AUDIO_DIR),
        asyncio.to_thread(text_to_speech, text)
    )

    return FileResponse(video_path, media_type="video/mp4", filename="output.mp4")

@app.get("/")
def root():
    return {"message": "🚀 API is running successfully!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

