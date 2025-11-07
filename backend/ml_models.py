# backend/ml_models.py
# (All your existing imports are correct)

import os
from typing import List, Dict, Any
from PIL import Image
from io import BytesIO 

# --- PyTorch & Transformers Imports (Text/Image) ---
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoImageProcessor, AutoModelForImageClassification as AutoImageModel

# --- TensorFlow/Keras & Audio Imports (Audio) ---
import pickle
import numpy as np
import librosa
import tensorflow.keras.models as tf_models

# --- (All your Configuration constants are correct) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEXT_MODEL_DIR = "../goemotions_model_complete"
IMAGE_MODEL_DIR = "../fer2013_emotion_model_final"
AUDIO_MODEL_FOLDER = "../audio_model" 
AUDIO_MODEL_PATH = os.path.join(AUDIO_MODEL_FOLDER, "ravdess_crnn_model.h5")
AUDIO_ENCODER_PATH = os.path.join(AUDIO_MODEL_FOLDER, "label_encoder.pkl")
SAMPLE_RATE = 22050
N_MFCC = 40
MAX_PAD_LEN = 174

# --- Global Model Storage ---
models = {}

# --- (load_models function is correct and loads all 3 models) ---
def load_models():
    """Loads ALL models (PyTorch and TensorFlow) into the 'models' dictionary."""
    
    # --- Load Text Model (PyTorch) ---
    try:
        print(f"Loading text model from: {TEXT_MODEL_DIR}")
        text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_DIR)
        text_model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_DIR)
        text_model.to(DEVICE)
        text_model.eval()
        models['text_tokenizer'] = text_tokenizer
        models['text_model'] = text_model
        models['text_id2label'] = text_model.config.id2label
        print("Text model loaded.")
    except Exception as e:
        print(f"Error loading text model: {e}")

    # --- Load Image Model (PyTorch) ---
    try:
        print(f"Loading image model from: {IMAGE_MODEL_DIR}")
        image_processor = AutoImageProcessor.from_pretrained(IMAGE_MODEL_DIR)
        image_model = AutoImageModel.from_pretrained(IMAGE_MODEL_DIR)
        image_model.to(DEVICE)
        image_model.eval()
        models['image_processor'] = image_processor
        models['image_model'] = image_model
        models['image_id2label'] = image_model.config.id2label
        print("Image model loaded.")
    except Exception as e:
        print(f"Error loading image model: {e}")

    # --- Load Audio Model (TensorFlow/Keras) ---
    try:
        print(f"Loading audio model from: {AUDIO_MODEL_PATH}")
        audio_model = tf_models.load_model(AUDIO_MODEL_PATH)
        models['audio_model'] = audio_model
        print("Audio model loaded.")
        
        print(f"Loading audio label encoder from: {AUDIO_ENCODER_PATH}")
        with open(AUDIO_ENCODER_PATH, 'rb') as f:
            audio_label_encoder = pickle.load(f)
        models['audio_label_encoder'] = audio_label_encoder
        print("Audio label encoder loaded.")
    except Exception as e:
        print(f"Error loading audio model or encoder: {e}")

# --- Text Prediction (NO CHANGE) ---
# This function is still used by /api/predict_text and for the separate display
def predict_caption(caption: str) -> List[Dict[str, Any]]:
    """Predicts emotions from a text caption. (Returns TOP 7 LIST)"""
    inputs = models['text_tokenizer'](caption, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = models['text_model'](**inputs).logits
    
    probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
    
    # Get all labels and scores, then sort
    all_preds = [{"label": models['text_id2label'][i], "score": float(probs[i])} for i in range(len(probs))]
    all_preds.sort(key=lambda x: x["score"], reverse=True)
    
    # Return the top 7
    return all_preds[:7]

# --- Image Prediction (NO CHANGE) ---
# This function is still used for the separate display
def predict_image(image: Image.Image) -> List[Dict[str, Any]]:
    """Predicts emotions from a PIL image. (Returns TOP 7 LIST)"""
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    inputs = models['image_processor'](images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = models['image_model'](**inputs).logits

    probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

    # Get all labels and scores, then sort
    all_preds = [{"label": models['image_id2label'][i], "score": float(probs[i])} for i in range(len(probs))]
    all_preds.sort(key=lambda x: x["score"], reverse=True)

    return all_preds[:7]

# --- (Audio functions are correct, NO CHANGE) ---
def extract_mfcc_from_bytes(audio_bytes: bytes) -> np.ndarray:
    # ... (function as before)
    try:
        y, sr = librosa.load(BytesIO(audio_bytes), sr=SAMPLE_RATE, duration=4.0)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        if mfcc.shape[1] < MAX_PAD_LEN:
            mfcc = np.pad(mfcc, ((0, 0), (0, MAX_PAD_LEN - mfcc.shape[1])), 'constant')
        else:
            mfcc = mfcc[:, :MAX_PAD_LEN]
        mfcc = mfcc[..., np.newaxis]
        return mfcc
    except Exception as e:
        print(f"Error processing audio bytes: {e}")
        raise

def predict_audio(audio_bytes: bytes) -> List[Dict[str, Any]]:
    # ... (function as before)
    try:
        features = extract_mfcc_from_bytes(audio_bytes)
        sample = features[np.newaxis, ...]
        model = models['audio_model']
        encoder = models['audio_label_encoder']
        probs = model.predict(sample)[0]
        top_preds_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:7]
        results = [
            {"label": encoder.classes_[i], "score": float(probs[i])}
            for i in top_preds_indices
        ]
        return results
    except Exception as e:
        print(f"Audio prediction failed: {e}")
        return []

# --- NEW: Functions for Fusion (Return FULL dictionary) ---

def predict_caption_full_dist(caption: str) -> Dict[str, float]:
    """
    Predicts emotions from text and returns a FULL dictionary 
    of all labels and their probabilities.
    """
    inputs = models['text_tokenizer'](caption, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = models['text_model'](**inputs).logits
    
    # Get probabilities for ALL classes
    probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
    
    # Create the full dictionary, ensuring labels are lowercase for mapping
    return {models['text_id2label'][i].lower(): float(probs[i]) for i in range(len(probs))}

def predict_image_full_dist(image: Image.Image) -> Dict[str, float]:
    """
    Predicts emotions from an image and returns a FULL dictionary 
    of all labels and their probabilities.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    inputs = models['image_processor'](images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = models['image_model'](**inputs).logits

    # Get probabilities for ALL classes
    probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
    
    # Create the full dictionary, ensuring labels are lowercase for mapping
    return {models['image_id2label'][i].lower(): float(probs[i]) for i in range(len(probs))}