from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
import logging
import sys # För att hantera kritiska fel vid uppstart

app = Flask(__name__)

# --- Konfiguration ---
class Config:
    MODEL_NAME = "roberta-base-openai-detector"
    AI_DETECTION_THRESHOLD = 0.5
    MAX_SEQUENCE_LENGTH = 512 # Maximal längd för tokenizern
    DEBUG = True # Sätt till False i produktion
    LOG_LEVEL = logging.INFO

app.config.from_object(Config)

# --- Loggning ---
logging.basicConfig(level=app.config["LOG_LEVEL"])
app.logger.setLevel(app.config["LOG_LEVEL"])

# För produktion: Lägg till filhantering för loggar
if not app.debug:
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler('app.log', maxBytes=1024 * 1024 * 10, backupCount=5)
    file_handler.setLevel(logging.WARNING)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    app.logger.addHandler(file_handler)

# --- Ladda AI-detektormodellen ---
tokenizer = None
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    app.logger.info(f"Laddar AI-detektormodell '{app.config['MODEL_NAME']}' till enhet: {device}")
    tokenizer = AutoTokenizer.from_pretrained(app.config["MODEL_NAME"])
    model = AutoModelForSequenceClassification.from_pretrained(app.config["MODEL_NAME"])
    model.to(device) # Flytta modellen till GPU om tillgänglig
    model.eval() # Sätt modellen i eval-läge för inferens
    app.logger.info("Modell laddad framgångsrikt.")
except Exception as e:
    app.logger.critical(f"Kritiskt fel: Kunde inte ladda AI-detektormodellen. Applikationen kan inte starta utan den. Fel: {e}")
    sys.exit(1) # Avsluta applikationen om modellen inte kan laddas

def detect_ai_prob_batch(texts: list[str]) -> list[float]:
    """
    Returnerar en lista med sannolikheten att varje text i batchen är AI-genererad.
    Hanterar tomma strängar korrekt.
    """
    if not texts:
        return []

    # Filtrera bort tomma strängar men behåll ordningen för att matcha tillbaka resultaten
    non_empty_indices = [i for i, text in enumerate(texts) if text.strip()]
    non_empty_texts = [texts[i] for i in non_empty_indices]

    if not non_empty_texts:
        return [0.0] * len(texts) # Alla var tomma strängar

    inputs = tokenizer(
        non_empty_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=app.config["MAX_SEQUENCE_LENGTH"]
    ).to(device) # Flytta input till rätt enhet

    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)

    full_probs = [0.0] * len(texts) # Initialisera med nollor
    for i, original_idx in enumerate(non_empty_indices):
        full_probs[original_idx] = probs[i][1].item() # Sannolikhet för AI-genererad

    return full_probs


def highlight_ai_segments(text: str) -> tuple[str, float]:
    """
    Delar upp text i meningar och markerar AI-liknande meningar.
    Returnerar markerad text och medelvärdet av AI-sannolikheten för icke-tomma meningar.
    """
    if not text.strip():
        return "", 0.0

    # Dela upp i meningar med regex
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    # Hämta AI-sannolikheter för varje mening
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
    """Rendera startsidan."""
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    """API-slutpunkt för att detektera AI-genererad text."""
    try:
        data = request.get_json(force=True)
        text = data.get("text", "")

        if not text.strip():
            app.logger.warning("Mottog en detekteringsförfrågan utan text.")
            return jsonify({"error": "Ingen text angiven för detektering."}), 400

        app.logger.info(f"Mottog text för detektering (längd: {len(text)}).")
        marked_text, ai_score = highlight_ai_segments(text)

        return jsonify({
            "highlighted_text": marked_text,
            "ai_score": ai_score
        })

    except Exception as e:
        app.logger.error(f"Fel i /detect-slutpunkten: {e}", exc_info=True)
        return jsonify({"error": "Internt serverfel uppstod vid detektering."}), 500

if __name__ == "__main__":
    app.run(debug=app.config["DEBUG"])