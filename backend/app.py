# backend/app.py

import asyncio
import base64
import uuid
from io import BytesIO
from typing import Dict

import requests
from PIL import Image
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Import our modularized functions
from scraper import fetch_post_data_from_any_url
from ml_models import (
    load_models, 
    predict_caption, 
    predict_image, 
    predict_audio,
    predict_caption_full_dist,  # NEW
    predict_image_full_dist   # NEW
)
# NEW: Import the fusion function
from fusion import fuse_predictions

# ----------------- App Setup -----------------
app = FastAPI(title="Instagram Emotion Analysis API")
results: Dict[str, Dict] = {}

@app.on_event("startup")
def startup_event():
    load_models()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- (API Request / Response Models are unchanged) ---
class UrlRequest(BaseModel):
    url: str

class TextRequest(BaseModel):
    text: str
    
# ---------- UPDATED: Background Task Logic ----------

def run_analysis_task(task_id: str, url: str):
    """This function runs in the background and contains all our original logic."""
    try:
        # Step 1: Scrape
        scraped_data = fetch_post_data_from_any_url(url)
        caption = scraped_data.get("caption", "")
        media_url = scraped_data.get("media_urls", [None])[0]
        
        # --- Step 2: Individual Predictions (for separate display) ---
        text_preds = []
        image_preds = []
        media_data_url = None
        
        # --- NEW: Placeholders for full probability distributions ---
        text_probs = {}
        image_probs = {}

        if caption and caption != "No caption found.":
            # For separate display (uses original function)
            text_preds = predict_caption(caption)
            # For fusion (uses NEW function)
            text_probs = predict_caption_full_dist(caption)

        if media_url:
            try:
                response = requests.get(media_url, timeout=15)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
                
                # Run both image prediction functions
                image_preds = predict_image(image)           # For separate display
                image_probs = predict_image_full_dist(image) # For fusion
                
                encoded_image = base64.b64encode(response.content).decode("utf-8")
                media_data_url = f"data:image/jpeg;base64,{encoded_image}"
            except Exception as e:
                print(f"Image processing failed: {e}")
        
        # --- Step 3: NEW Fused Prediction ---
        fused_predictions = []
        # Run fusion if we have results from at least one model
        if text_probs or image_probs:
            fused_predictions = fuse_predictions(text_probs, image_probs)
        
        # --- Step 4: Store the final, successful result ---
        results[task_id] = {
            "status": "complete",
            "data": {
                "caption": caption,
                "media_data_url": media_data_url,
                "fused_predictions": fused_predictions, # NEW combined results
                "text_predictions": text_preds,         # Separate text results
                "image_predictions": image_preds        # Separate image results
            }
        }
    except Exception as e:
        print(f"Task {task_id} failed: {e}")
        results[task_id] = {"status": "failed", "error": str(e)}

# --- (All other endpoints are 100% UNCHANGED) ---

@app.post("/api/predict_url")
async def predict_from_url(req: UrlRequest, background_tasks: BackgroundTasks):
    # (No changes)
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="Empty URL provided.")
    task_id = str(uuid.uuid4())
    results[task_id] = {"status": "pending"} 
    background_tasks.add_task(run_analysis_task, task_id, url)
    return {"message": "Analysis started", "task_id": task_id}

@app.get("/api/result/{task_id}")
def get_result(task_id: str):
    # (No changes)
    result = results.get(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Task ID not found.")
    return result
    
@app.post("/api/predict_text")
async def predict_from_text(req: TextRequest):
    # (No changes - this still works perfectly for the text-only section)
    text = req.text.strip()
    if not text: raise HTTPException(status_code=400, detail="Empty text provided.")
    try:
        text_preds = await asyncio.to_thread(predict_caption, text)
        return {"caption": text, "text_predictions": text_preds}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.post("/api/predict_audio")
async def predict_from_audio(file: UploadFile = File(...)):
    # (No changes - this still works perfectly for the audio-only section)
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file provided.")
        audio_preds = await asyncio.to_thread(predict_audio, audio_bytes)
        return {"audio_predictions": audio_preds}
    except Exception as e:
        print(f"Error in audio endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Audio prediction failed: {e}")

@app.get("/api/health")
def health():
    # (No changes)
    return {"status": "ok"}