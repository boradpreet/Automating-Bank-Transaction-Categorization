from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Initialize Flask app
app = Flask(__name__)

# Load models and tokenizers
model_name = load_model("lstm_matched_name.h5")
model_acc = load_model("lstm_account.h5")

# Load label encoders
label_enc_name = LabelEncoder()
label_enc_name.classes_ = np.load("label_enc_name_classes.npy", allow_pickle=True)  # Allow pickle for loading

label_enc_acc = LabelEncoder()
label_enc_acc.classes_ = np.load("label_enc_acc_classes.npy", allow_pickle=True)  # Allow pickle for loading

# Tokenizer initialization
MAX_VOCAB_SIZE = 10000
OOV_TOKEN = "<OOV>"
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token=OOV_TOKEN)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    return text.strip()

# Load the dataset and apply the clean_text function
data = pd.read_csv("ABC.csv")  # Load the dataset
data["Cleaned_Description"] = data["Description"].apply(lambda x: clean_text(x))  # Apply text cleaning

# Fit the tokenizer on the cleaned descriptions
tokenizer.fit_on_texts(data["Cleaned_Description"])

# Prediction function
def classify_transaction(description, threshold=50):
    cleaned_desc = clean_text(description)
    seq = tokenizer.texts_to_sequences([cleaned_desc])
    padded_seq = pad_sequences(seq, maxlen=model_name.layers[0].input.shape[1], padding='post')

    pred_name = model_name.predict(padded_seq)[0]
    pred_acc = model_acc.predict(padded_seq)[0]

    best_name_index = np.argmax(pred_name)
    best_acc_index = np.argmax(pred_acc)

    best_matched_name = label_enc_name.inverse_transform([best_name_index])[0] if pred_name.size > 0 else "Not Found"
    best_account = label_enc_acc.inverse_transform([best_acc_index])[0] if pred_acc.size > 0 else "Not Found"

    best_matched_conf = pred_name[best_name_index] * 100 if pred_name.size > 0 else 0.0
    best_account_conf = pred_acc[best_acc_index] * 100 if pred_acc.size > 0 else 0.0

    if best_matched_conf < threshold:
        best_matched_name = "Not Found"

    if best_account_conf < threshold:
        best_account = "Not Found"

    results = {
        "description": description,
        "predicted_matched_name": f"{best_matched_name} ({best_matched_conf:.2f}%)",
        "predicted_account": f"{best_account} ({best_account_conf:.2f}%)",
    }

    return results

# Flask route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Get description from the request
    description = data.get("description", "")

    if not description:
        return jsonify({"error": "Description is required"}), 400

    # Make prediction
    results = classify_transaction(description)

    return jsonify(results)

# Main route to render the HTML form
@app.route('/')
def home():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
