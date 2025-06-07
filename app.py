from flask import Flask, request, jsonify, render_template
import onnxruntime as ort
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = "model.onnx"

# Kontrollera att filen finns
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modellfil saknas: {MODEL_PATH}")

# Initiera onnxruntime session
ort_session = ort.InferenceSession(MODEL_PATH)

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
        # ONNX-modellen förväntar sig input som en array av strängar med form (batch_size, 1)
        inputs = {ort_session.get_inputs()[0].name: np.array([[text]], dtype=object)}
        outputs = ort_session.run(None, inputs)

        # Antag att output[1] är sannolikheten för klassen "AI"
        # Det kan variera beroende på modell, kolla din ONNX-export
        probs = outputs[1]  # Om modellen ger predict_proba som andra output
        prob_ai = float(probs[0][1])  # sannolikhet för "AI"-klass
    except Exception as e:
        return jsonify({"error": f"Något gick fel vid prediktion: {str(e)}"}), 500

    return jsonify({
        "highlighted_text": text,
        "ai_score": round(prob_ai, 4)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
