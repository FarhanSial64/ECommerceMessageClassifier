import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from tqdm import tqdm
import pickle
import logging
import os

# ========== CONFIG ==========
MODEL_NAME = "xlm-roberta-base"
MAX_LEN = 64
BATCH_SIZE = 32
EPOCHS = 3
NUM_WORKERS = os.cpu_count() - 1 or 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ============================

# ======= Logging Setup =======
logging.basicConfig(
    format='[%(levelname)s] %(asctime)s - %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S'
)
log = logging.getLogger()

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

# ======= Main =======
if __name__ == "__main__":
    log.info("Loading datasets...")
    train_df = pd.read_excel("train_data.xlsx")
    val_df = pd.read_excel("val_data.xlsx")
    test_df = pd.read_excel("test_data.xlsx")

    all_labels = pd.concat([train_df["label"], val_df["label"], test_df["label"]])
    le = LabelEncoder()
    le.fit(all_labels)

    # Save label encoder
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    # Encode labels
    y_train = le.transform(train_df["label"])
    y_val = le.transform(val_df["label"])
    y_test = le.transform(test_df["label"])

    train_dataset = WhatsAppDataset(train_df["text"].tolist(), y_train)
    val_dataset = WhatsAppDataset(val_df["text"].tolist(), y_val)
    test_dataset = WhatsAppDataset(test_df["text"].tolist(), y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    num_classes = len(le.classes_)
    model = WhatsAppClassifier(MODEL_NAME, num_classes).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    # ===== Training =====
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

    # ===== Save Model =====
    log.info("Saving model...")
    torch.save(model.state_dict(), "message_classifier.pth")
    with open("message_classifier_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # ===== Evaluation on Test Set =====
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    log.info(f"🧪 Test Accuracy: {accuracy * 100:.2f}%")
