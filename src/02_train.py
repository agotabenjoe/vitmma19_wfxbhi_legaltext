import os
import json
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from collections import Counter
import sys
from utils import logger
from model import LSTMClassifier, TextDataset, OrdinalLoss
import config

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y.long()).sum().item()
    return total_loss / len(loader), correct / total

def train():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(base_dir, '..')
    model_dir = os.path.join(root_dir, config.MODEL_DIR)
    os.makedirs(model_dir, exist_ok=True)
    global_best_model_path = os.path.join(root_dir, config.BEST_MODEL_PATH)
    global_best_metric_path = os.path.join(root_dir, config.BEST_ACC_PATH)
    
    logger.info("========== TRAINING CONFIGURATION ==========")
    logger.info(f"EPOCHS: {config.EPOCHS}")
    logger.info(f"BATCH_SIZE: {config.BATCH_SIZE}")
    logger.info(f"LEARNING_RATE: {config.LEARNING_RATE}")
    logger.info(f"EMBED_DIM: {config.EMBED_DIM}")
    logger.info(f"HIDDEN_DIM: {config.HIDDEN_DIM}")
    logger.info(f"NUM_LAYERS: {config.NUM_LAYERS}")
    logger.info(f"EARLY_STOPPING_PATIENCE: {config.EARLY_STOPPING_PATIENCE}")
    logger.info("============================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    dataset_path = os.path.join(root_dir, config.DATASET_PATH)
    dataset = load_from_disk(dataset_path)
    logger.info("Data loaded from %s", dataset_path)
    logger.info("Train size: %d | Validation size: %d", len(dataset['train']), len(dataset['validation']))
    train_counts = Counter([x['label'] for x in dataset['train']])
    val_counts = Counter([x['label'] for x in dataset['validation']])
    logger.info(f"Train class distribution: {dict(train_counts)}")
    logger.info(f"Validation class distribution: {dict(val_counts)}")
    logger.info("Sample training examples:")
    for i in range(min(3, len(dataset['train']))):
        sample = dataset['train'][i]
        logger.info(f"Sample {i+1}: Label={sample['label']} | Text={sample['text'][:100]}...")
    full_vocab_source = TextDataset(dataset['train'])
    vocab = full_vocab_source.vocab
    vocab_size = len(vocab) + 1
    train_data = TextDataset(dataset['train'], vocab=vocab)
    val_data = TextDataset(dataset['validation'], vocab=vocab)
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE, shuffle=False)
    model = LSTMClassifier(
        vocab_size=vocab_size,
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_classes=config.NUM_CLASSES,
        num_layers=config.NUM_LAYERS,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model architecture: %s", model)
    logger.info(f"Total parameters: {n_params} | Trainable: {n_trainable}")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = OrdinalLoss(num_classes=config.NUM_CLASSES)
    best_run_acc = 0.0
    best_val_loss_for_early_stopping = float('inf')
    epochs_no_improve = 0
    best_run_state = None
    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        if val_acc > best_run_acc:
            best_run_acc = val_acc
            best_run_state = model.state_dict()
        if val_loss < best_val_loss_for_early_stopping:
            best_val_loss_for_early_stopping = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    logger.info(f"Training Finished. Best Validation Accuracy: {best_run_acc:.4f}")
    logger.info("Saving best model...")
    torch.save(best_run_state, global_best_model_path)
    config_dict = {
        "batch_size": config.BATCH_SIZE,
        "embed_dim": config.EMBED_DIM,
        "hidden_dim": config.HIDDEN_DIM,
        "num_layers": config.NUM_LAYERS,
        "learning_rate": config.LEARNING_RATE,
        "epochs": config.EPOCHS
    }
    with open(global_best_model_path.replace('.pt', '_config.json'), 'w') as f:
        json.dump(config_dict, f)
    with open(global_best_model_path.replace('.pt', '_vocab.json'), 'w') as f:
        json.dump(vocab, f)
    logger.info(f"Model saved to {global_best_model_path}")

if __name__ == '__main__':
    train()
    sys.exit(0)