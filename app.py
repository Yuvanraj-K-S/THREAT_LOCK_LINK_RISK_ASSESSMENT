from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import joblib
import pickle
import re

print("Loading model...")
# Load model
model = load_model(r"E:\Threat_lock_link_risk_assessment\outputs\deep_url_classifier.h5")
print("Model loaded.")
print("Loading tokenizer...")
# Load tokenizer
with open(r"E:\Threat_lock_link_risk_assessment\outputs\url_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
print("Tokenizer loaded.")
# Must match the max_len used during training
MAX_LEN = 100

from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow all origins (for dev)


@app.route("/check", methods=["POST"])
def check_url():
    data = request.get_json()
    url = data.get("url", "")
    max_len = 200
    url_pattern = r'https?://(?:www\.)?([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
    match = re.search(url_pattern, url)
    # Check if a match is found and extract the matching part
    if match:
        url = match.group(1)
        #url = url[:-1]
    # Tokenize and pad the input URL
    sequence = tokenizer.texts_to_sequences([url])
    padded = pad_sequences(sequence, maxlen=max_len)
    
    # Predict
    prob = model.predict(padded)[0][0]
    label = True if prob > 0.15 else False
    print(label,prob,url,type(url))
    return jsonify({"is_malicious": bool(label)})

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)