# NLP-Powered Emotion Classifier for Mental Health Monitoring

## Overview
[cite_start]This project presents a multimodal emotion recognition system designed to analyze and understand human emotions from social media content. [cite: 8] [cite_start]Emotions play a crucial role in communication, but most sentiment analysis systems focus only on text, failing to capture the depth conveyed through other modalities. [cite: 6, 7]

[cite_start]To address this, our system analyzes and fuses insights from **text**, **images**, and **audio** to provide a more holistic and accurate understanding of user emotion. [cite: 8, 21] [cite_start]The web application, built using React and FastAPI, allows users to analyze social media posts or directly upload media for real-time emotion prediction. [cite: 13]

---

## System Architecture
[cite_start]The system uses a tri-modal framework that integrates Natural Language Processing (NLP), Computer Vision (CV), and Speech Processing. [cite: 24] [cite_start]It is built on a service-oriented architecture with a React frontend and a FastAPI (Python) backend. [cite: 37, 73, 75]

### 1. Text Model (NLP)
* [cite_start]**Model:** **DistilBERT**, a distilled version of BERT chosen for its balance of performance and efficiency. [cite: 58]
* [cite_start]**Dataset:** Fine-tuned on the **GoEmotions** dataset, a large-scale corpus of 58k human-annotated Reddit comments. [cite: 59]
* [cite_start]**Purpose:** Classifies text into 7 core emotions: Joy, Sadness, Anger, Surprise, Fear, Disgust, and Neutral. [cite: 60]

### 2. Image Model (Computer Vision)
* [cite_start]**Model:** **Vision Transformer (ViT)**, which applies the transformer architecture to image patches to capture global context. [cite: 63]
* [cite_start]**Dataset:** Fine-tuned on the **FER-2013** dataset. [cite: 64]
* [cite_start]**Purpose:** Identifies human facial expressions from images. [cite: 62]

### 3. Audio Model (Speech Processing)
* [cite_start]**Model:** A hybrid **CNN-LSTM** (Convolutional Neural Network + Long Short-Term Memory) architecture. [cite: 51]
* [cite_start]**Dataset:** Trained on the **RAVDESS** dataset of emotional speech recordings. [cite: 71]
* [cite_start]**Features:** Uses **Mel-Frequency Cepstral Coefficients (MFCCs)** as features to represent the audio's timbre, pitch, and energy. [cite: 69]

---

## How to Run

Follow these steps to set up and run the project locally.

### 1. Setup Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.

```bash
# Create a new virtual environment
python -m venv venv
