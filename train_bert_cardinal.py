import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
import numpy as np

# Cardinal loss implementation
class CardinalLoss(torch.nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
    def forward(self, logits, targets):
        # logits: [batch, num_classes], targets: [batch]
        prob = torch.nn.functional.softmax(logits, dim=1)
        targets = targets.view(-1, 1)
        # Cardinal loss: penalize distance between predicted and true class
        class_indices = torch.arange(self.num_classes, device=logits.device).float()
        expected = (prob * class_indices).sum(dim=1)
        return torch.mean((expected - targets.squeeze().float()) ** 2)

class ConsensusDataset(Dataset):
    def __init__(self, folder, tokenizer, max_length=128):
        self.samples = []
        for file in os.listdir(folder):
            if file.endswith('.json'):
                with open(os.path.join(folder, file), encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        text = item['data']['text']
                        ann = item['annotations'][0]['result'] if item['annotations'] and item['annotations'][0]['result'] else None
                        if ann:
                            label_str = ann[0]['value']['choices'][0]
                            label = int(label_str.split('-')[0]) - 1  # 0-based
                            self.samples.append((text, label))
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        text, label = self.samples[idx]
        enc = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        item = {key: val.squeeze(0) for key, val in enc.items()}
        item['labels'] = torch.tensor(label)
        return item

def train():
    folder = 'consensus/consensus'
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    dataset = ConsensusDataset(folder, tokenizer)
    train_idx, val_idx = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=42)
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=8, num_workers=0)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=5)
    model.train()
    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = CardinalLoss(num_classes=5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    for epoch in range(2):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * input_ids.size(0)
        print(f'Epoch {epoch+1} train loss: {total_loss/len(train_set):.4f}')
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            val_loss += loss.item() * input_ids.size(0)
    print(f'Validation cardinal loss: {val_loss/len(val_set):.4f}')

if __name__ == '__main__':
    train()
