import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class CardinalLoss(torch.nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
    def forward(self, logits, targets):
        prob = torch.nn.functional.softmax(logits, dim=1)
        targets = targets.view(-1, 1)
        class_indices = torch.arange(self.num_classes, device=logits.device).float()
        expected = (prob * class_indices).sum(dim=1)
        return torch.mean((expected - targets.squeeze().float()) ** 2)

class ConsensusDataset(Dataset):
    def __init__(self, folder, vocab=None, max_length=50, max_samples=32):
        self.samples = []
        self.vocab = vocab or {}
        self.max_length = max_length
        # Only load WFXBHI.json
        file_path = os.path.join(folder, 'WFXBHI.json')
        if os.path.exists(file_path):
            with open(file_path, encoding='utf-8') as f:
                data = json.load(f)
                for item in data:  # Use all samples
                    text = item['data']['text']
                    ann = item['annotations'][0]['result'] if item['annotations'] and item['annotations'][0]['result'] else None
                    if ann:
                        label_str = ann[0]['value']['choices'][0]
                        label = int(label_str.split('-')[0]) - 1  # 0-based
                        self.samples.append((text, label))
        if not vocab:
            self.build_vocab()
    def build_vocab(self):
        words = set()
        for text, _ in self.samples:
            words.update(text.lower().split())
        self.vocab = {word: i+1 for i, word in enumerate(words)}  # 0 for padding
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        text, label = self.samples[idx]
        tokens = [self.vocab.get(word, 0) for word in text.lower().split()][:self.max_length]
        tokens += [0] * (self.max_length - len(tokens))
        return torch.tensor(tokens), torch.tensor(label, dtype=torch.float)

class MinimalLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=20, hidden_dim=4, num_classes=5):  # Normal dimensions for few thousand params
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        return self.fc(h.squeeze(0))

def train_overfit_single_batch():
    folder = 'notebook'  # Use notebook folder
    dataset = ConsensusDataset(folder)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    # Take first batch for overfitting
    single_batch = next(iter(train_loader))
    x_batch, y_batch = single_batch
    vocab_size = len(dataset.vocab) + 1
    model = MinimalLSTM(vocab_size)
    # Print model parameters and layer dimensions
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {total_params}')
    print(f'Embedding: {model.embedding}')
    print(f'LSTM: {model.lstm}')
    print(f'FC: {model.fc}')
    print(f'FC: {model.fc}')
    print(f'Embedding dim: {model.embedding.embedding_dim}, Hidden dim: {model.lstm.hidden_size}, Output dim: {model.fc.out_features}')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = CardinalLoss(num_classes=5)
    losses = []
    print('Starting overfitting on single batch with steps until loss < 0.001...')
    step = 0
    max_steps = 10000
    while step < max_steps:
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if loss.item() < 0.001:
            print(f'Loss < 0.001 reached at step {step+1}, Loss: {loss.item():.6f}')
            break
        step += 1
        if step % 50 == 0:
            print(f'Step {step}, Loss: {loss.item():.6f}')
    else:
        print(f'Max steps reached, final loss: {loss.item():.6f}')
    # Final predictions
    model.eval()
    with torch.no_grad():
        final_outputs = model(x_batch)
        pred_classes = torch.argmax(final_outputs, dim=1)
        print(f'True labels: {y_batch.tolist()}')
        print(f'Predicted classes: {pred_classes.tolist()}')
    # Plot loss
    plt.plot(losses)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Overfitting on Single Batch with Steps')
    plt.savefig('lstm_overfit.png')
    print(f'Final loss: {losses[-1]:.4f}')
    print('Overfitting complete. Plot saved as lstm_overfit.png')

if __name__ == '__main__':
    train_overfit_single_batch()
