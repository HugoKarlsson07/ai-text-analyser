from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
import logging
import sys
import os
from functools import lru_cache # För cache-funktioner

app = Flask(__name__)

# --- Konfiguration ---
class Config:
    # Byt till en mindre modell som troligen ligger under 400MB
    # Exempel på mindre modeller:
    # 'distilroberta-base' - Cirka 328 MB (en bra start)
    # 'ProsusAI/finbert' - Cirka 420 MB (troligen för stor, men specialiserad)
    # 'pysentimiento/robertuito-base-cased-sentiment' - Cirka 440 MB (också troligen för stor)
    # 'finiteautomata/bertweet-base-sentiment-analysis' - Cirka 300 MB (BERT-baserad, inte RoBERTa, men liten)

    # Vi väljer 'distilroberta-base' som ett troligt alternativ
    MODEL_NAME = "distilroberta-base"
    AI_DETECTION_THRESHOLD = 0.7
    MAX_SEQUENCE_LENGTH = 128 # Fortfarande en rimlig längd för kortare meningar
    DEBUG = False
    LOG_LEVEL = logging.INFO

app.config.from_object(Config)

# --- Loggning ---
logger = app.logger
logging.basicConfig(level=app.config["LOG_LEVEL"], stream=sys.stdout)

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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    logger.info(f"Laddar modell '{app.config['MODEL_NAME']}' på enhet: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(app.config["MODEL_NAME"])

    # Kontrollera modellens storlek innan den laddas in helt i minnet, om möjligt.
    # Detta är inte alltid direkt möjligt via Hugging Face Transformers utan att ladda ner vikterna först.
    # Det bästa sättet att kontrollera är att manuellt ladda ner modellen och se filstorleken,
    # eller att titta på modellkortet på Hugging Face Hub.
    
    model = AutoModelForSequenceClassification.from_pretrained(app.config["MODEL_NAME"])
    model.to(DEVICE)
    model.eval()
    logger.info("Modell laddad.")

    # Validera minnesanvändning (endast för informativt syfte, inte en hård gräns)
    if DEVICE.type == 'cpu':
        # För CPU är det svårare att få exakt VRAM-användning. Vi kan approximera.
        # Detta är inte en perfekt mätning, men kan ge en indikation.
        # torch.cuda.memory_allocated() är endast för GPU.
        # För CPU måste vi lita på att modellen i sig är under gränsen.
        # En grov uppskattning: modell.state_dict() storlek i bytes
        total_params = sum(p.numel() for p in model.parameters())
        model_size_bytes = total_params * 4 # Uppskatta 4 bytes per float32 parameter
        model_size_mb = model_size_bytes / (1024 * 1024)
        logger.info(f"Uppskattad CPU-modellstorlek i minne: {model_size_mb:.2f} MB")
        if model_size_mb > 400:
            logger.critical(f"VARNING: Uppskattad CPU-modellstorlek ({model_size_mb:.2f} MB) överstiger 400 MB gränsen.")
            # Här kan du välja att avsluta appen om gränsen är absolut kritisk.
            # sys.exit(1)
    elif DEVICE.type == 'cuda':
        allocated_memory = torch.cuda.memory_allocated(DEVICE) / (1024 * 1024)
        logger.info(f"Allokerat GPU-minne efter modelladdning: {allocated_memory:.2f} MB")
        if allocated_memory > 400:
            logger.critical(f"VARNING: GPU-minnesanvändning ({allocated_memory:.2f} MB) överstiger 400 MB gränsen.")
            # Här kan du välja att avsluta appen om gränsen är absolut kritisk.
            # sys.exit(1)

except Exception as e:
    logger.critical(f"Kunde inte ladda modell: {e}")
    sys.exit(1)

# Kompilera regex för prestanda
SENTENCE_SPLITTER = re.compile(r'(?<=[.!?])\s*(?=\S|$)')

@lru_cache(maxsize=256) # Ökad cache-storlek för fler unika meningar
def _detect_ai_prob_single(text: str) -> float:
    """
    Beräknar AI-detektionssannolikheten för en enskild textsträng.
    Denna funktion är cachad för prestanda.
    """
    stripped_text = text.strip()
    if not stripped_text:
        return 0.0

    inputs = tokenizer(
        stripped_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=app.config["MAX_SEQUENCE_LENGTH"]
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)
    
    # OBS: `distilroberta-base` är inte tränad för att detektera AI-genererad text.
    # Den är tränad för maskerad språkanaly och nästa mening prediction.
    # Utgången (logits) kan därför inte tolkas direkt som "mänsklig" vs "AI".
    # För att använda en sådan modell för AI-detektion skulle den behöva finjusteras (fine-tuned)
    # på ett dataset för AI-detektion, annars kommer resultaten inte vara meningsfulla.
    # För demonstrationens skull antar vi att probs[0][1] representerar en 'AI-sannolikhet'
    # men i en verklig applikation måste du fine-tuna modellen eller välja en modell
    # som är specifikt tränad för AI-detektion (som den ursprungliga 'roberta-base-openai-detector').
    # Om du byter modell, måste du också validera vad index 1 (eller 0) representerar.
    
    return probs[0][1].item() # Returnera sannolikheten för index 1


def detect_ai_prob_batch(texts: list[str]) -> list[float]:
    """
    Beräknar AI-detektionssannolikheten för en batch av texter.
    Använder den cachade singeldetekteringsfunktionen.
    """
    return [_detect_ai_prob_single(text) for text in texts]

def highlight_ai_segments(text: str) -> tuple[str, float]:
    stripped_text = text.strip()
    if not stripped_text:
        return "", 0.0

    sentences = SENTENCE_SPLITTER.split(stripped_text)
    sentences = [s for s in sentences if s.strip()]

    if not sentences:
        return "", 0.0

    all_probs = detect_ai_prob_batch(sentences)

    highlighted_segments = []
    total_prob = 0.0
    valid_sentence_count = 0

    for i, sentence in enumerate(sentences):
        prob = all_probs[i]

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
            highlighted_segments.append(sentence)

    return_text = " ".join(highlighted_segments)
    avg_prob = total_prob / valid_sentence_count if valid_sentence_count else 0.0
    return return_text, avg_prob

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    try:
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
    port = int(os.environ.get("PORT", 10000))
    if app.config["DEBUG"]:
        app.run(host="0.0.0.0", port=port, debug=True)
    else:
        logger.info(f"Kör i produktionsläge. Använd en WSGI-server, t.ex. 'gunicorn app:app -b 0.0.0.0:{port}'")
        app.run(host="0.0.0.0", port=port)