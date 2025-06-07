from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
import logging
import sys
import os

app = Flask(__name__)

# --- Konfiguration ---
class Config:
    MODEL_NAME = "roberta-base-openai-detector"
    AI_DETECTION_THRESHOLD = 0.5
    MAX_SEQUENCE_LENGTH = 512
    DEBUG = False  # Viktigt: sätt False i produktion
    LOG_LEVEL = logging.INFO

app.config.from_object(Config)

# --- Loggning ---
logging.basicConfig(level=app.config["LOG_LEVEL"])
app.logger.setLevel(app.config["LOG_LEVEL"])

if not app.debug:
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler('app.log', maxBytes=10 * 1024 * 1024, backupCount=5)
    file_handler.setLevel(logging.WARNING)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    app.logger.addHandler(file_handler)

# --- Ladda AI-detektormodellen ---
tokenizer = None
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    app.logger.info(f"Laddar modell '{app.config['MODEL_NAME']}' på enhet: {device}")
    tokenizer = AutoTokenizer.from_pretrained(app.config["MODEL_NAME"])
    model = AutoModelForSequenceClassification.from_pretrained(app.config["MODEL_NAME"])
    model.to(device)
    model.eval()
    app.logger.info("Modell laddad.")
except Exception as e:
    app.logger.critical(f"Kunde inte ladda modell: {e}")
    sys.exit(1)

def detect_ai_prob_batch(texts: list[str]) -> list[float]:
    if not texts:
        return []

    non_empty_indices = [i for i, text in enumerate(texts) if text.strip()]
    non_empty_texts = [texts[i] for i in non_empty_indices]

    if not non_empty_texts:
        return [0.0] * len(texts)

    inputs = tokenizer(
        non_empty_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=app.config["MAX_SEQUENCE_LENGTH"]
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)

    full_probs = [0.0] * len(texts)
    for i, original_idx in enumerate(non_empty_indices):
        full_probs[original_idx] = probs[i][1].item()

    return full_probs

def highlight_ai_segments(text: str) -> tuple[str, float]:
    if not text.strip():
        return "", 0.0

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    all_probs = detect_ai_prob_batch(sentences)

    highlighted = []
    valid_probs = []

    for i, sentence in enumerate(sentences):
        if not sentence.strip():
            highlighted.append(sentence)
            continue

        prob = all_probs[i] if i < len(all_probs) else 0.0
        valid_probs.append(prob)

        if prob >= app.config["AI_DETECTION_THRESHOLD"]:
            highlighted.append(
                f"<span class='highlight' data-tooltip='AI-likhet: {prob:.2f}'>{sentence}</span>"
            )
        else:
            highlighted.append(sentence)

    return_text = " ".join(highlighted)
    avg_prob = sum(valid_probs) / len(valid_probs) if valid_probs else 0.0
    return return_text, avg_prob

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    try:
        data = request.get_json(force=True)
        text = data.get("text", "")

        if not text.strip():
            app.logger.warning("Ingen text mottagen.")
            return jsonify({"error": "Ingen text angiven."}), 400

        marked_text, ai_score = highlight_ai_segments(text)

        return jsonify({
            "highlighted_text": marked_text,
            "ai_score": ai_score
        })

    except Exception as e:
        app.logger.error(f"Fel i /detect: {e}", exc_info=True)
        return jsonify({"error": "Internt serverfel."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=app.config["DEBUG"])
