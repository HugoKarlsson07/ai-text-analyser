from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

app = Flask(__name__)

MODEL_NAME = "microsoft/xtremedistil-l6-h256-uncased"  # Extremt liten modell
MAX_LEN = 128
THRESHOLD = 0.7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SENTENCE_SPLITTER = re.compile(r'(?<=[.!?])\s+(?=\S)')

# Ladda modell
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

def get_ai_prob(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN).to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
    return probs[0][1].item()  # OBS! Utan finetuning är detta inte tillförlitligt!

@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Ingen text angiven."}), 400

    sentences = SENTENCE_SPLITTER.split(text)
    scores = [get_ai_prob(s) for s in sentences]
    total_score = sum(scores)
    highlighted = []

    for s, score in zip(sentences, scores):
        if score >= THRESHOLD:
            highlighted.append(f"<span class='highlight' title='AI: {score:.2f}'>{s}</span>")
        else:
            highlighted.append(s)

    avg_score = total_score / len(scores) if scores else 0.0
    return jsonify({
        "highlighted_text": " ".join(highlighted),
        "ai_score": avg_score
    })

if __name__ == "__main__":
    app.run(port=10000)
