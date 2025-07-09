import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from tqdm import tqdm
import pickle
import logging
import os

# ========== CONFIG ==========
MODEL_NAME = "xlm-roberta-base"  # or "bert-base-multilingual-cased"
MAX_LEN = 64
BATCH_SIZE = 32
EPOCHS = 3
NUM_WORKERS = os.cpu_count() - 1 or 2  # Use available CPU cores
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ============================

# ======= Logging Setup =======
logging.basicConfig(
    format='[%(levelname)s] %(asctime)s - %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S'
)
log = logging.getLogger()

# ======= Tokenizer (Global Scope for Dataset Access) =======
log.info(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ======= Dataset Class =======
class WhatsAppDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=MAX_LEN,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ======= Model Class =======
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

# =============================
# MAIN TRAINING FUNCTION ENTRY
# =============================
if __name__ == "__main__":
    log.info("Loading dataset...")
    df = pd.read_excel("training_data.xlsx")  # You can change to training_data_15000.xlsx
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(str).tolist()

    # Label encoding
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    num_classes = len(le.classes_)

    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    # Split Dataset
    log.info("Splitting dataset into training and validation...")
    X_train, X_val, y_train, y_val = train_test_split(texts, encoded_labels, test_size=0.2, random_state=42)

    train_dataset = WhatsAppDataset(X_train, y_train)
    val_dataset = WhatsAppDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # Load Model
    log.info("Initializing model...")
    model = WhatsAppClassifier(MODEL_NAME, num_classes).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    log.info("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

        for batch in loop:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        log.info(f"[Epoch {epoch+1}] Training Loss: {avg_loss:.4f}")

    # Save model weights
    log.info("Saving model weights to message_classifier.pth...")
    torch.save(model.state_dict(), "message_classifier.pth")

    # Save full model (optional)
    log.info("Saving full model to message_classifier_model.pkl...")
    with open("message_classifier_model.pkl", "wb") as f:
        pickle.dump(model, f)

    log.info("✅ Training complete. Files saved: model (.pth), full model (.pkl), label encoder (.pkl)")
