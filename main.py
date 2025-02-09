import os
import aiohttp
import asyncio
import subprocess
import logging
import hashlib
from collections import Counter
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from docx import Document
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from TTS.api import TTS
import nltk

# Download necessary NLTK data
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
UPLOAD_DIR = "data"
AUDIO_DIR = "audio"
VIDEO_DIR = "public/videos"
CACHE_DIR = "cache"
for directory in [UPLOAD_DIR, AUDIO_DIR, VIDEO_DIR, CACHE_DIR]:
    os.makedirs(directory, exist_ok=True)

# API Keys
PIXABAY_API_KEY = "48738698-6ae327a6f8a04d813fa6c6101"
FREESOUND_API_KEY = "YG213ZwV2kJwHtcXAfs3Ik6s0lTJzjwI45WnFA23"

# Load Coqui TTS Model
MODEL_NAME = "tts_models/en/ljspeech/tacotron2-DDC"
coqui_tts = TTS(MODEL_NAME)

# Generate cache paths
def get_cache_path(key, directory):
    return os.path.join(directory, hashlib.md5(key.encode()).hexdigest())

# Caching mechanism for downloads
async def cached_download(url, directory):
    cache_path = get_cache_path(url, directory)
    if os.path.exists(cache_path):
        logging.info(f"Using cached file: {cache_path}")
        return cache_path
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                content = await response.read()
                with open(cache_path, "wb") as f:
                    f.write(content)
                return cache_path
            else:
                logging.error(f"Failed to download {url} - Status {response.status}")
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

# Extract keywords from text
def extract_keywords(text, num_keywords=3):
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    return [word for word, _ in Counter(filtered_words).most_common(num_keywords)]

# Fetch media (generic function)
async def fetch_media(keyword, api_key, base_url, params):
    url = f"{base_url}?key={api_key}&q={keyword}&" + "&".join(f"{k}={v}" for k, v in params.items())
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("hits", [])
    return []

# Fetch video from Pixabay
async def fetch_video(keyword):
    videos = await fetch_media(keyword, PIXABAY_API_KEY, "https://pixabay.com/api/videos/", {"per_page": 3})
    return videos[0]["videos"]["medium"]["url"] if videos else None

# Fetch background audio from Freesound (Improved with better error handling)
async def fetch_background_audio(keyword):
    url = f"https://freesound.org/apiv2/search/text/?query={keyword}&token={FREESOUND_API_KEY}&fields=previews"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                logging.error(f"Freesound API request failed with status {response.status}")
                return None
            
            try:
                data = await response.json()
                sounds = data.get("results", [])
                
                if not sounds:
                    logging.warning(f"No background audio found for keyword: {keyword}")
                    return None
                
                for sound in sounds:
                    preview = sound.get("previews", {}).get("preview-hq-mp3")
                    if preview:
                        logging.info(f"Found background audio: {preview}")
                        return preview
                
                logging.warning(f"No suitable audio preview found for keyword: {keyword}")
                return None
            
            except Exception as e:
                logging.error(f"Error parsing Freesound API response: {str(e)}")
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
    logging.info("Starting video generation process")
    
    if file:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        text = extract_text(file_path)
        logging.info(f"Extracted text from file: {file.filename}")
    
    if not text:
        logging.error("No text provided")
        return JSONResponse(content={"error": "No text provided"}, status_code=400)
    
    keywords = extract_keywords(text)
    logging.info(f"Extracted keywords: {keywords}")
    
    video_url, audio_url = None, None
    for keyword in keywords:
        video_url = video_url or await fetch_video(keyword)
        audio_url = audio_url or await fetch_background_audio(keyword)
        if video_url and audio_url:
            break
    
    if not video_url or not audio_url:
        return JSONResponse(content={"error": "Could not find video or background audio"}, status_code=500)

    logging.info(f"Found video URL: {video_url}")
    logging.info(f"Found audio URL: {audio_url}")
    
    video_path, background_audio_path, speech_audio_path = await asyncio.gather(
        cached_download(video_url, VIDEO_DIR),
        cached_download(audio_url, AUDIO_DIR),
        asyncio.to_thread(text_to_speech, text)
    )

    output_video = os.path.join(VIDEO_DIR, "output.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", video_path, "-i", speech_audio_path, "-i", background_audio_path,
        "-filter_complex", "[1:a][2:a]amix=inputs=2:duration=first:weights=4 1[aout]",
        "-map", "0:v", "-map", "[aout]", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k", "-shortest", output_video
    ]

    await asyncio.create_subprocess_exec(*cmd)
    return FileResponse(output_video, media_type="video/mp4", filename="output.mp4")

@app.get("/")
def root():
    return {"message": "API is running successfully!"}
