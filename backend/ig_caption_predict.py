# ig_caption_predict.py
import re
import os
import zipfile
from playwright.sync_api import sync_playwright
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

def extract_caption(raw_caption: str) -> str:
    # Try to extract text inside the outermost quotes
    quoted_text = re.findall(r'"([^"]+)"', raw_caption)
    if quoted_text:
        caption = " ".join(quoted_text)
    else:
        parts = raw_caption.split(":")
        if len(parts) > 1:
            caption = parts[-1]
        else:
            caption = raw_caption
    caption = caption.strip()
    caption = re.sub(r'\s+', ' ', caption)
    return caption

def clean_caption(text: str) -> str:
    # remove emojis and non-ascii characters; keep it simple
    return re.sub(r'[^\x00-\x7F]+', '', text).strip()

def scrape_instagram_post(post_url: str, headless: bool = True, timeout: int = 15000):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context()
        page = context.new_page()
        page.goto(post_url, wait_until='networkidle')
        # Accept cookies if the banner appears (best-effort)
        try:
            page.locator('text=Only allow essential cookies').click(timeout=5000)
        except Exception:
            pass
        # Wait for article to load
        try:
            page.wait_for_selector('article', timeout=timeout)
        except Exception:
            # fallback to waiting for any img
            page.wait_for_selector('img', timeout=timeout)
        # Try meta description first
        caption = None
        try:
            meta = page.locator('meta[name="description"]').first
            content = meta.get_attribute('content')
            if content:
                caption = content
        except Exception:
            caption = None
        # As fallback try to find the first visible caption-like element inside article
        if not caption:
            try:
                # Instagram renders captions inside <div role="button"> elements in Recent DOM variants,
                # but this is brittle across UI versions. We'll attempt multiple selectors.
                possible = page.locator('article').locator('div')
                # pick the longest visible text node
                texts = []
                for i in range(min(60, possible.count())):
                    try:
                        t = possible.nth(i).inner_text().strip()
                        if t:
                            texts.append(t)
                    except Exception:
                        pass
                if texts:
                    # heuristic: the longest text likely contains the caption + username etc.
                    caption = max(texts, key=len)
            except Exception:
                caption = None
        if not caption:
            caption = "No caption found"
        caption = clean_caption(caption)
        caption = extract_caption(caption)
        # Get media URLs (best-effort)
        try:
            media_urls = page.eval_on_selector_all('article img', 'elements => elements.map(el => el.src)')
        except Exception:
            media_urls = []
        browser.close()
        return {'caption': caption, 'media_urls': media_urls}

def load_label_map_from_dir(model_dir: str):
    """
    Try to load id2label from model config if available.
    If not available, look for a labels.txt or labels.json in the folder.
    Otherwise return a numeric mapping.
    """
    # attempt via transformers config (when loading model below it will populate config.id2label)
    labels_path_txt = os.path.join(model_dir, "labels.txt")
    if os.path.exists(labels_path_txt):
        with open(labels_path_txt, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]
        return {i: lab for i, lab in enumerate(labels)}
    # fallback: return None and we'll rely on model.config.id2label or numeric indices
    return None

def predict_caption_emotions(caption: str, model_dir: str, top_k: int = 5, device: str = None):
    """
    Loads model/tokenizer from model_dir and returns top_k predicted labels and probabilities.
    device: "cpu" or "cuda" or None (auto)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # Prepare label mapping
    label_map = None
    if hasattr(model.config, "id2label") and model.config.id2label:
        # id2label is usually a dict mapping "0":"joy" or 0:"joy"
        id2label = {int(k): v for k, v in model.config.id2label.items()}
        label_map = id2label
    else:
        label_map = load_label_map_from_dir(model_dir)
        if label_map is None:
            # fallback numeric labels
            label_map = {i: str(i) for i in range(model.config.num_labels)}

    # Tokenize (single example)
    inputs = tokenizer(
        caption,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # forward
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(0)  # shape: (num_labels,)
        # if multi-label (sigmoid) or single-label (softmax)? We'll guess:
        # If the last activation used for training was BCEWithLogitsLoss (multi-label), you'll want sigmoid.
        # If CrossEntropyLoss (single-label) then softmax. We attempt to infer from problem type by checking config.
        probs = None
        # Common hint: if model was trained multi-label, often config.problem_type = "multi_label_classification"
        problem_type = getattr(model.config, "problem_type", None)
        if problem_type == "multi_label_classification":
            probs = torch.sigmoid(logits)
        else:
            # default to softmax
            probs = F.softmax(logits, dim=-1)
        probs = probs.cpu().numpy()

    # If multi-label, return labels with prob > threshold and sorted; if single-label return top_k softmax labels.
    results = []
    if problem_type == "multi_label_classification":
        # use threshold 0.3 as default, but still return top_k by probability if none pass threshold
        thresh = 0.3
        indices = [i for i, p in enumerate(probs) if p >= thresh]
        if not indices:
            indices = list(reversed(probs.argsort()[-top_k:]))  # highest first
        for i in indices:
            results.append((label_map.get(i, str(i)), float(probs[i])))
        # sort by prob desc
        results = sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
    else:
        topk = min(top_k, len(probs))
        idxs = probs.argsort()[-topk:][::-1]
        for i in idxs:
            results.append((label_map.get(i, str(i)), float(probs[i])))

    return results

if __name__ == "__main__":
    # Example usage:
    # 1) Make sure the model dir points to your unzipped model folder.
    MODEL_DIR = "../goemotions_model_complete"  # <--- put your unzipped model folder here
    INSTAGRAM_URL = "https://www.instagram.com/p/DO0YiXSEpJk/?utm_source=ig_web_copy_link"

    # 2) Scrape
    print("Scraping Instagram post...")
    scraped = scrape_instagram_post(INSTAGRAM_URL, headless=True)
    caption = scraped.get("caption", "").strip()
    print("Raw caption:", caption)

    # 3) Predict
    print("\nLoading model and predicting...")
    preds = predict_caption_emotions(caption, MODEL_DIR, top_k=6)
    print("\nTop predictions:")
    for label, prob in preds:
        print(f"{label}: {prob:.4f}")
