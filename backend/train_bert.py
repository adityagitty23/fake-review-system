import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm   # ✅ progress bar

MODEL_NAME = "bert-base-uncased"

# ================= LOAD DATA =================
df = pd.read_csv("reviews.csv")

texts = df["review_text"].tolist()
labels = df["label"].tolist()

# ================= TOKENIZER & MODEL =================
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

# ================= DATASET =================
class ReviewDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

dataset = ReviewDataset(texts, labels)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

optimizer = AdamW(model.parameters(), lr=5e-5)

# ================= TRAINING =================
EPOCHS = 1   # ⭐ recommended for i3 CPU

model.train()

for epoch in range(EPOCHS):
    print(f"\n🚀 Epoch {epoch + 1}/{EPOCHS}")

    progress_bar = tqdm(loader, desc="Training", unit="batch")

    total_loss = 0

    for batch in progress_bar:
        optimizer.zero_grad()

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)
    print(f"✅ Epoch {epoch + 1} completed | Avg Loss: {avg_loss:.4f}")

# ================= SAVE MODEL =================
model.save_pretrained("bert")
tokenizer.save_pretrained("bert")

print("\n🎉 BERT model trained and saved successfully")
