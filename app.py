from flask import Flask, request, jsonify, render_template
import joblib
import os

app = Flask(__name__)

MODEL_PATH = "model.joblib"

# Kontrollera att filen finns, annars kasta fel
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modellfil saknas: {MODEL_PATH}")

# Ladda modellen och vectorizern från samma fil (en tuple)
model, vectorizer = joblib.load(MODEL_PATH)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Ingen text angiven."}), 400

    try:
        X = vectorizer.transform([text])
        prob = model.predict_proba(X)[0][1]
    except Exception as e:
        return jsonify({"error": f"Något gick fel vid prediktion: {str(e)}"}), 500

    return jsonify({
        "highlighted_text": text,
        "ai_score": round(prob, 4)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
