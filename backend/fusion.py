# backend/fusion.py

from typing import Dict, List

# --- Label Mapping ---
# We map the Image Model's labels (keys) to the Text Model's labels (values).
# Based on your sample, 'happy' maps to 'joy', 'sad' to 'sadness', etc.
# Please VERIFY these are your 7 text model labels.
LABEL_MAPPING_IMAGE_TO_TEXT = {
    # Key = Image Model Label, Value = Text Model Label
    "happy": "joy",
    "sad": "sadness",
    "angry": "anger",
    "fearful": "fear",
    "disgust": "disgust",
    "surprised": "surprise",
    "neutral": "neutral"
}

# Get the 7 canonical labels we will use for the final output
CANONICAL_LABELS = sorted(list(set(LABEL_MAPPING_IMAGE_TO_TEXT.values())))
# This should result in: ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']


def fuse_predictions(
    text_probs: Dict[str, float], 
    image_probs: Dict[str, float], 
    text_weight: float = 0.6, 
    image_weight: float = 0.4
) -> List[Dict[str, any]]:
    """
    Fuses predictions from text and image models using weighted averaging
    based on a common label mapping.
    """
    
    # 1. Map image probabilities to the common text label space
    #    e.g., {"happy": 0.9, "sad": 0.1} -> {"joy": 0.9, "sadness": 0.1}
    mapped_image_probs = {}
    for img_label, score in image_probs.items():
        common_label = LABEL_MAPPING_IMAGE_TO_TEXT.get(img_label.lower())
        if common_label:
            mapped_image_probs[common_label] = score

    # 2. Create combined scores using the weights
    combined_scores = {}
    
    for label in CANONICAL_LABELS:
        text_score = text_probs.get(label, 0)
        image_score = mapped_image_probs.get(label, 0)
        
        # Apply the weights
        combined_score = (text_score * text_weight) + (image_score * image_weight)
        combined_scores[label] = combined_score
    
    # 3. Normalize the final scores so they add up to 1 (optional but good practice)
    total_score = sum(combined_scores.values())
    if total_score == 0:
        # Fallback if no scores are present
        return [{"label": "neutral", "score": 1.0}]
    
    normalized_scores = {label: score / total_score for label, score in combined_scores.items()}

    # 4. Format for the frontend (list of dicts, sorted)
    final_preds = [{"label": label, "score": score} for label, score in normalized_scores.items()]
    final_preds.sort(key=lambda x: x["score"], reverse=True)
    
    return final_preds