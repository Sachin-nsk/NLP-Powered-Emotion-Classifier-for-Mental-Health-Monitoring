const API_BASE_URL = "http://127.0.0.1:8000";

const els = {
  url: document.getElementById("ig-url"),
  processUrl: document.getElementById("process-url"),
  clearUrl: document.getElementById("clear-url"),
  text: document.getElementById("caption-text"),
  processText: document.getElementById("process-text"),
  clearText: document.getElementById("clear-text"),
  audioFile: document.getElementById("audio-file"),
  processAudio: document.getElementById("process-audio"),
  clearAudio: document.getElementById("clear-audio"),
  captionPreview: document.getElementById("caption-preview"),
  finalPredList: document.getElementById("final-pred-list"),
  loaderArea: document.getElementById("loader-area"),
  loaderText: document.querySelector("#loader-area p"),
  scrapedImage: document.getElementById("scraped-image"),
  mediaPlaceholder: document.getElementById("media-placeholder"),
};

// ---------- Helper functions ----------
const showLoader = (msg = "Analyzing sentiment...") => {
  els.loaderText.textContent = msg;
  els.loaderArea.classList.remove("hidden");
};
const hideLoader = () => els.loaderArea.classList.add("hidden");

const resetResults = () => {
  els.captionPreview.textContent = "No content processed yet.";
  els.scrapedImage.classList.add("hidden");
  els.mediaPlaceholder.classList.remove("hidden");
  els.scrapedImage.src = "";
  els.finalPredList.innerHTML = `<div class="text-dark-muted">No predictions yet.</div>`;
};

const makeGradientForIndex = (i) => {
  const hue = (i * 47) % 360;
  return `linear-gradient(90deg, hsl(${hue} 95% 65%), hsl(${(hue + 40) % 360} 85% 56%))`;
};

const renderPredictions = (predictions, placeholder) => {
  const container = els.finalPredList;
  container.innerHTML = "";
  if (!predictions || predictions.length === 0) {
    container.innerHTML = `<div class="text-dark-muted">${placeholder}</div>`;
    return;
  }
  predictions.forEach((pred, idx) => {
    const pct = Math.round(pred.score * 100);
    const bar = document.createElement("div");
    bar.className = "flex items-center space-x-3";
    bar.innerHTML = `
      <div class="flex-1 bg-dark-border rounded-full overflow-hidden">
        <div class="h-3 rounded-full" style="width:${pct}%; background:${makeGradientForIndex(idx)}"></div>
      </div>
      <span class="w-24 text-sm font-medium">${pred.label} (${pct}%)</span>
    `;
    container.appendChild(bar);
  });
};

const renderMedia = (imageUrl) => {
  if (imageUrl) {
    els.scrapedImage.src = imageUrl;
    els.scrapedImage.classList.remove("hidden");
    els.mediaPlaceholder.classList.add("hidden");
  } else {
    els.scrapedImage.classList.add("hidden");
    els.mediaPlaceholder.classList.remove("hidden");
  }
};

// ---------- URL ----------
els.processUrl.addEventListener("click", async () => {
  const url = els.url.value.trim();
  if (!url) return alert("Please enter a valid post URL.");
  resetResults();
  showLoader("Fetching and analyzing post...");

  const response = await fetch(`${API_BASE_URL}/api/predict_url`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
  const { task_id } = await response.json();

  const poll = async () => {
    const res = await fetch(`${API_BASE_URL}/api/result/${task_id}`);
    const data = await res.json();
    if (data.status === "pending") return setTimeout(poll, 2000);

    hideLoader();
    if (data.status === "failed") return alert("Processing failed: " + data.error);

    const result = data.data;
    els.captionPreview.textContent = result.caption || "No caption found.";
    renderMedia(result.media_data_url);
    renderPredictions(result.fused_predictions, "No predictions yet.");
  };
  poll();
});

// ---------- Text ----------
els.processText.addEventListener("click", async () => {
  const text = els.text.value.trim();
  if (!text) return alert("Please enter caption text.");
  resetResults();
  showLoader("Analyzing text...");

  const res = await fetch(`${API_BASE_URL}/api/predict_text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  const data = await res.json();

  hideLoader();
  els.captionPreview.textContent = data.caption;
  renderPredictions(data.text_predictions, "No predictions yet.");
});

// ---------- Audio ----------
els.processAudio.addEventListener("click", async () => {
  const file = els.audioFile.files[0];
  if (!file) return alert("Please upload an audio file.");
  resetResults();
  showLoader("Analyzing audio...");

  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_BASE_URL}/api/predict_audio`, { method: "POST", body: formData });
  const data = await res.json();

  hideLoader();
  renderPredictions(data.audio_predictions, "No predictions yet.");
});

// ---------- Clear ----------
els.clearUrl.addEventListener("click", () => (els.url.value = ""));
els.clearText.addEventListener("click", () => (els.text.value = ""));
els.clearAudio.addEventListener("click", () => (els.audioFile.value = ""));
