# 🧠 NLP-Powered Emotion Classifier for Mental Health Monitoring

## 📘 Overview
This project presents a **multimodal emotion recognition system** designed to analyze and understand human emotions from social media content.  
Emotions play a crucial role in communication, but most sentiment analysis systems focus only on text, failing to capture the emotional depth expressed through other modalities.

To overcome this limitation, our system fuses insights from **text**, **images**, and **audio** to provide a holistic and accurate understanding of user emotions.  
The web application, built using **React (frontend)** and **FastAPI (backend)**, enables users to analyze social media posts or upload media for real-time emotion detection.

---

## 🏗️ System Architecture
The system employs a **tri-modal framework** integrating Natural Language Processing (NLP), Computer Vision (CV), and Speech Processing.  
It follows a **service-oriented architecture**, ensuring modularity and scalability.

### 1️⃣ Text Model (NLP)
- **Model:** DistilBERT — a distilled version of BERT optimized for performance and efficiency.  
- **Dataset:** Fine-tuned on **GoEmotions**, a 58k human-annotated Reddit comment dataset.  
- **Purpose:** Classifies text into 7 core emotions:  
  **Joy**, **Sadness**, **Anger**, **Surprise**, **Fear**, **Disgust**, and **Neutral**.

### 2️⃣ Image Model (Computer Vision)
- **Model:** Vision Transformer (**ViT**) — applies the transformer mechanism to image patches for global feature extraction.  
- **Dataset:** Fine-tuned on **FER-2013**, a widely used facial emotion dataset.  
- **Purpose:** Detects human facial expressions and infers the dominant emotion.

### 3️⃣ Audio Model (Speech Processing)
- **Model:** Hybrid **CNN-LSTM** (Convolutional + Recurrent network).  
- **Dataset:** Trained on **RAVDESS**, a dataset of emotional speech recordings.  
- **Features:** Extracts **Mel-Frequency Cepstral Coefficients (MFCCs)** to represent tone, energy, and pitch variations.  
- **Purpose:** Predicts emotional tone from voice recordings.

---

## ⚙️ How to Run

Follow the steps below to set up and run the project locally.

### 🔹 1. Setup Virtual Environment
It is recommended to use a virtual environment for dependency management.

```bash
# Create a new virtual environment
python -m venv venv
```
```bash
# Activate the virtual environment
# On Windows:
.\venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```
```bash
# Navigate to the backend directory
cd backend
```
```bash
# Install all dependencies
pip install -r requirements.txt
```
```bash
#Run the Backend Server
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
```bash
#Run the Frontend
index.html
