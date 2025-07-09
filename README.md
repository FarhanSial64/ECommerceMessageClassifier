# 🤖 WhatsApp Message Intent Classifier for eCommerce

This project builds a multilingual AI model that classifies customer messages received via **WhatsApp Web** for **eCommerce businesses**. It enables automated detection of intent such as **order delivery inquiries**, **pricing questions**, and more — allowing for instant and intelligent replies via a Chrome extension or integrated backend service.

## 🚀 Project Purpose

Modern eCommerce businesses rely heavily on **WhatsApp** for customer support. However, manual response to repetitive queries like:

- "Where is my order?"
- "Kitne ka hai ye?"
- "Order kab milega?"
- "Batao price please"

…can overwhelm support teams.

This project solves that problem by:

- 🧠 Using **transformer-based NLP** models to classify message intent
- 🌍 Supporting **multilingual and casual message patterns** (e.g., Urdu + English mix)
- 🤝 Enabling **real-time automation** for WhatsApp customer service
- 📦 Designed to integrate with eCommerce backend APIs for actions like checking delivery status or fetching product prices

---

## 🧱 Tech Stack

| Component | Description |
|----------|-------------|
| 🧠 Model | [XLM-RoBERTa](https://huggingface.co/xlm-roberta-base) (Transformer for multilingual understanding) |
| 💬 Dataset | 15,000 WhatsApp messages across 4 intent labels |
| 🔧 Backend | Python, PyTorch, Transformers |
| 📊 Data Handling | Pandas, Scikit-learn |
| 💾 Storage | Trained model saved as `.pth` and `.pkl` |
| 🧪 Evaluation | Accuracy on 10% hold-out test set |

---

## 🧩 Dataset

The dataset consists of **15,000 messages** labeled with four key intent categories:

- `delivery` – Order tracking, ETA, etc.
- `pricing` – Asking about prices
- `product_inquiry` – Specs, availability, etc.
- `others` – General inquiries not classifiable

Data is preprocessed and split into:

- ✅ 70% for training  
- 🔍 20% for validation  
- 🧪 10% for final testing  

All splits are stratified for label balance and stored as Excel files (`train_data.xlsx`, `val_data.xlsx`, `test_data.xlsx`).

---

## 🏋️‍♂️ Model Training

The model architecture uses:

- **XLM-RoBERTa Base** as the encoder
- A custom classification head
- Cross-entropy loss
- AdamW optimizer
- GPU acceleration if available

Training is performed using 3 epochs, with batch size 32, and supports parallel data loading via `num_workers`.

### 📂 Output Artifacts

After training, the following files are saved:

- `message_classifier.pth` – Trained model weights
- `message_classifier_model.pkl` – Serialized full model (optional)
- `label_encoder.pkl` – Label encoder for inference use

---

## 📊 Evaluation

After training, the model is evaluated on the **test set** and provides accuracy feedback. You can optionally add:

- `sklearn` classification report
- Confusion matrix
- Precision, Recall, F1 metrics

---

## 🔮 Example Use Case

Once trained, this model can be deployed to a **Chrome Extension** or **backend service** that:

1. Reads incoming WhatsApp Web messages using DOM injection
2. Feeds the message to the model
3. Identifies the intent (e.g., delivery query)
4. Responds with a relevant automated message
5. Optionally fetches data from eCommerce backend (order status, price)

---

## 🧪 Inference Example

```python
from transformers import AutoTokenizer
import torch
import pickle

# Load model + label encoder
with open("message_classifier_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
text = "Mera order kab tak ayega?"

inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
model.eval()
with torch.no_grad():
    logits = model(inputs['input_ids'], inputs['attention_mask'])
    pred = torch.argmax(logits, dim=1)
    intent = le.inverse_transform(pred.numpy())[0]

print(f"Predicted intent: {intent}")
````

---

## 📦 Setup & Installation

```bash
# Clone repository
git clone https://github.com/yourusername/whatsapp-intent-classifier.git
cd whatsapp-intent-classifier

# Install dependencies
pip install -r requirements.txt

# Train the model
python train_message_classifier.py
```

---

## 🛠 Requirements

* Python 3.8+
* PyTorch
* Transformers (Hugging Face)
* Pandas, TQDM, Scikit-learn
* Excel support via openpyxl or xlrd

Install via:

```bash
pip install torch transformers pandas scikit-learn openpyxl tqdm
```

---

## 📁 Folder Structure

```
.
├── train_message_classifier.py     # Training script
├── train_data.xlsx
├── val_data.xlsx
├── test_data.xlsx
├── message_classifier.pth         # Trained weights
├── message_classifier_model.pkl   # Optional full model
├── label_encoder.pkl              # Label encoder
├── README.md
```

---

## 📌 Future Enhancements

* Add FastAPI server for real-time API inference
* Integrate into Chrome Extension DOM
* Add response templates for auto-reply
* Support more intent classes (returns, complaints, support)

---

## 💡 Author

**Farhan Sial**
Software Engineering Student @ FAST University
GitHub: [FarhanSial64](https://github.com/FarhanSial64)

