import streamlit as st
import torch
import torch.nn as nn
import pickle
from transformers import AutoTokenizer, AutoModel
import os

# ========= CONFIG ==========
MODEL_NAME = "xlm-roberta-base"
MAX_LEN = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "message_classifier.pth"
ENCODER_PATH = "label_encoder.pkl"
# ===========================

# ====== Load Tokenizer ======
@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)

tokenizer = load_tokenizer()

# ====== Load Label Encoder ======
with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)
labels = list(label_encoder.classes_)

# ====== Model Definition (same as training) ======
class WhatsAppClassifier(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        return self.classifier(self.dropout(pooled))

# ====== Load Trained Model ======
@st.cache_resource
def load_model():
    model = WhatsAppClassifier(MODEL_NAME, num_classes=len(labels))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ===== Prediction Function =====
def predict_message(text):
    enc = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LEN)
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()

    return labels[pred_idx], confidence

# ===== Streamlit UI =====
st.title("📦 WhatsApp Message Classifier")
st.write("This app tells whether a message is **delivery-related** or not.")

msg = st.text_area("Enter a WhatsApp message:")

if st.button("Classify"):
    if not msg.strip():
        st.warning("Please enter a message.")
    else:
        label, conf = predict_message(msg)
        st.success(f"**Prediction:** `{label}`")
        st.info(f"Confidence: `{conf*100:.2f}%`")
