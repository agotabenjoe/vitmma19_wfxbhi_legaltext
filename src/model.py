import torch
import torch.nn as nn
from torch.utils.data import Dataset

# --- Custom Loss ---
class OrdinalLoss(torch.nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        
    def forward(self, logits, targets):
        prob = torch.nn.functional.softmax(logits, dim=1)
        class_indices = torch.arange(self.num_classes, device=logits.device).float()
        expected = (prob * class_indices).sum(dim=1)
        return torch.mean((expected - targets.float()) ** 2)

# --- Dataset Wrapper ---
class TextDataset(Dataset):
    def __init__(self, hf_dataset, vocab=None, max_length=50):
        self.data = hf_dataset
        self.max_length = max_length
        # If vocab is provided (Testing), use it. Otherwise build it (Training).
        self.vocab = vocab if vocab else self._build_vocab()

    def _build_vocab(self):
        words = set()
        for item in self.data:
            if item.get('text'):
                words.update(item['text'].lower().split())
        # CRITICAL: Sort to ensure ID 5 is always the same word across runs
        return {word: i + 1 for i, word in enumerate(sorted(words))} 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get('text', '') or ''
        label_str = item.get('label', '')
        
        label_val = item.get('label', '')
        if isinstance(label_val, int):
            label = label_val
        else:
            try:
                label = int(str(label_val).split('-')[0]) - 1
            except (ValueError, IndexError, AttributeError):
                label = 0
            
        tokens = [self.vocab.get(word, 0) for word in text.lower().split()][:self.max_length]
        tokens += [0] * (self.max_length - len(tokens))
        
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.float)

# --- Model Architecture ---
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes=5, num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Dropout is only valid if num_layers > 1
        lstm_dropout = dropout if num_layers > 1 else 0
        
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            batch_first=True, 
            num_layers=num_layers, 
            bidirectional=True, 
            dropout=lstm_dropout
        )
        # Input to FC is hidden_dim * 2 because of Bidirectional LSTM
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        # h shape: (num_layers * num_directions, batch, hidden_dim)
        _, (h, _) = self.lstm(x)
        
        # Concat the last forward and backward hidden states
        cat_hidden = torch.cat((h[-2], h[-1]), dim=1)
        return self.fc(cat_hidden)