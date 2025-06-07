from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
import os

app = Flask(__name__)
CORS(app)  # Remove if CORS is not needed

MODEL_NAME = "microsoft/xtremedistil-l6-h256-uncased"
MAX_LEN = 64
THRESHOLD = 0.7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SENTENCE_SPLITTER = re.compile(r'(?<=[.!?])\s+(?=\S)')

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

def get_ai_prob(text):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN
    ).to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
    return probs[0][1].item()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Ingen text angiven."}), 400

    sentences = SENTENCE_SPLITTER.split(text)
    scores = [get_ai_prob(sentence) for sentence in sentences]

    highlighted = []
    for sentence, score in zip(sentences, scores):
        if score >= THRESHOLD:
            # Use data-tooltip for custom CSS tooltip
            highlighted.append(f"<span class='highlight' data-tooltip='AI: {score:.2f}'>{sentence}</span>")
        else:
            highlighted.append(sentence)

    avg_score = sum(scores) / len(scores) if scores else 0.0

    return jsonify({
        "highlighted_text": " ".join(highlighted),
        "ai_score": round(avg_score, 4)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
