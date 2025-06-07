from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

app = Flask(__name__)

MODEL_PATH = "model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"

def train_and_save_model():
    # Dummy data, byt ut mot riktiga träningsdata
    texts = [
        "This is AI generated text example.",
        "Det här är en mänskligt skriven mening.",
        "Another AI generated sentence.",
        "Mänskligt innehåll här."
    ]
    labels = [1, 0, 1, 0]  # 1 = AI, 0 = människa

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression()
    model.fit(X, labels)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print("Modell tränad och sparad.")

# Träna modellen om den inte finns
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    train_and_save_model()

# Ladda modell och vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Ingen text angiven."}), 400

    # Dela gärna text i meningar om du vill
    # Här analyserar vi hela texten som en enda enhet
    X = vectorizer.transform([text])
    prob = model.predict_proba(X)[0][1]

    # Highlighting kan implementeras i frontend eller enkelt som här
    # Här returnerar vi endast sannolikheten och låter frontend markera

    return jsonify({
        "highlighted_text": text,  # Skicka tillbaka text oförändrad, frontend kan markera efter behov
        "ai_score": round(prob, 4)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
