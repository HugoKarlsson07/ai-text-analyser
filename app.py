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
    AI_DETECTION_THRESHOLD = 0.7
    MAX_SEQUENCE_LENGTH = 64
    DEBUG = False  # Viktigt: sätt False i produktion
    LOG_LEVEL = logging.INFO

app.config.from_object(Config)

# --- Loggning ---
# Använd en global logger för att undvika redundans
logger = app.logger
logging.basicConfig(level=app.config["LOG_LEVEL"], stream=sys.stdout) # Logga till stdout i debug-läge

if not app.debug:
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler('app.log', maxBytes=10 * 1024 * 1024, backupCount=5)
    file_handler.setLevel(logging.WARNING)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("Produktionsloggning aktiverad.")
else:
    logger.info("Debug-läge aktiverat. Loggning till konsol.")


# --- Ladda AI-detektormodellen ---
tokenizer = None
model = None
# Bestäm enhet en gång vid uppstart
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    logger.info(f"Laddar modell '{app.config['MODEL_NAME']}' på enhet: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(app.config["MODEL_NAME"])
    model = AutoModelForSequenceClassification.from_pretrained(app.config["MODEL_NAME"])
    model.to(DEVICE)
    model.eval() # Sätt modellen i utvärderingsläge en gång
    logger.info("Modell laddad.")
except Exception as e:
    logger.critical(f"Kunde inte ladda modell: {e}")
    sys.exit(1)

# Kompilera regex för prestanda
SENTENCE_SPLITTER = re.compile(r'(?<=[.!?])\s+')

def detect_ai_prob_batch(texts: list[str]) -> list[float]:
    """
    Beräknar AI-detektionssannolikheten för en batch av texter.
    Optimerad för att hantera tomma eller endast blankstegstexter effektivt.
    """
    if not texts:
        return []

    # Filtrera bort tomma strängar och strängar med endast blanksteg tidigt
    processed_texts = []
    original_indices = []
    for i, text in enumerate(texts):
        stripped_text = text.strip()
        if stripped_text:
            processed_texts.append(stripped_text)
            original_indices.append(i)

    if not processed_texts:
        return [0.0] * len(texts) # Alla texter var tomma eller blanksteg

    inputs = tokenizer(
        processed_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=app.config["MAX_SEQUENCE_LENGTH"]
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1) # Sannolikheter för båda klasserna (mänsklig/AI)

    # Skapa en lista med nollor i samma storlek som ursprungliga indata
    full_probs = [0.0] * len(texts)
    # Fyll i sannolikheterna för de bearbetade texterna
    for i, original_idx in enumerate(original_indices):
        full_probs[original_idx] = probs[i][1].item() # Ta sannolikheten för AI-klassen (index 1)

    return full_probs

def highlight_ai_segments(text: str) -> tuple[str, float]:
    """
    Delar upp text i meningar, detekterar AI-sannolikhet för varje mening
    och markerar meningar över en viss tröskel.
    Returnerar den markerade texten och den genomsnittliga AI-sannolikheten.
    """
    stripped_text = text.strip()
    if not stripped_text:
        return "", 0.0

    # Använd kompilerad regex
    sentences = SENTENCE_SPLITTER.split(stripped_text)
    
    # Ta bort eventuella tomma strängar som kan uppstå från split i slutet
    sentences = [s for s in sentences if s.strip()]

    if not sentences:
        return "", 0.0 # Hantera fall där endast blanksteg finns efter strip och split

    all_probs = detect_ai_prob_batch(sentences)

    highlighted_segments = []
    total_prob = 0.0
    valid_sentence_count = 0

    for i, sentence in enumerate(sentences):
        # Använd proben direkt från all_probs; detect_ai_prob_batch garanterar korrekt längd/ordning
        prob = all_probs[i]

        # Inkludera endast meningar med faktiskt innehåll i genomsnittsberäkningen
        if sentence.strip():
            total_prob += prob
            valid_sentence_count += 1
            if prob >= app.config["AI_DETECTION_THRESHOLD"]:
                highlighted_segments.append(
                    f"<span class='highlight' data-tooltip='AI-likhet: {prob:.2f}'>{sentence}</span>"
                )
            else:
                highlighted_segments.append(sentence)
        else:
            highlighted_segments.append(sentence) # Lägg till tomma meningar tillbaka för korrekt formattering

    return_text = " ".join(highlighted_segments)
    avg_prob = total_prob / valid_sentence_count if valid_sentence_count else 0.0
    return return_text, avg_prob

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    """
    API-slutpunkt för att detektera AI-genererad text.
    Förväntar sig JSON med en 'text'-nyckel.
    """
    try:
        # Använd get_json(silent=True) för att hantera icke-JSON-förfrågningar utan att kasta fel
        data = request.get_json(silent=True)

        if not data:
            logger.warning("Ingen JSON-data mottagen eller ogiltig JSON.")
            return jsonify({"error": "Ogiltigt format. Förväntade JSON-data."}), 400

        text = data.get("text", "")

        if not text.strip():
            logger.warning("Ingen text mottagen.")
            return jsonify({"error": "Ingen text angiven."}), 400

        marked_text, ai_score = highlight_ai_segments(text)

        return jsonify({
            "highlighted_text": marked_text,
            "ai_score": ai_score
        })

    except Exception as e:
        logger.error(f"Fel i /detect: {e}", exc_info=True)
        return jsonify({"error": "Internt serverfel."}), 500

if __name__ == "__main__":
    # Flask körs i utvecklingsläge när debug=True.
    # För produktion, använd en WSGI-server som Gunicorn.
    port = int(os.environ.get("PORT", 10000))
    # app.run(host="0.0.0.0", port=port, debug=app.config["DEBUG"])
    # I en produktionsmiljö, kör inte med app.run(debug=True)
    # Använd gunicorn app:app -b 0.0.0.0:10000
    if app.config["DEBUG"]:
        app.run(host="0.0.0.0", port=port, debug=True)
    else:
        # Logga instruktioner för produktionskörning
        logger.info(f"Kör i produktionsläge. Använd en WSGI-server, t.ex. 'gunicorn app:app -b 0.0.0.0:{port}'")
        # app.run() anropas inte direkt här, då en WSGI-server tar över
        # Men för att koden ska kunna köras direkt i en utvecklingsmiljö utan gunicorn
        # kan man ha en fallback:
        app.run(host="0.0.0.0", port=port) # Utan debug=True