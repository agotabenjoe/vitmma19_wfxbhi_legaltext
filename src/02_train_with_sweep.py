import os
import json
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
import wandb
import sys
from utils import logger
from model import LSTMClassifier, TextDataset, OrdinalLoss

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
    with wandb.init() as run:
        config = run.config
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, '..', 'models')
        os.makedirs(model_dir, exist_ok=True)
        global_best_model_path = os.path.join(model_dir, "best_sweep_model.pt")
        global_best_metric_path = os.path.join(model_dir, "best_sweep_acc.txt")
        logger.info("========== TRAINING CONFIGURATION ==========")
        for k, v in dict(config).items():
            logger.info(f"{k}: {v}")
        logger.info("============================================")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device} ({'CUDA available' if torch.cuda.is_available() else 'CPU only'})")
        dataset_path = os.path.join(base_dir, '..', 'data', 'final_split_dataset')
        dataset = load_from_disk(dataset_path)
        logger.info("Data loaded from %s", dataset_path)
        logger.info("Train size: %d | Validation size: %d", len(dataset['train']), len(dataset['validation']))
        logger.info("Sample training examples:")
        for i in range(min(3, len(dataset['train']))):
            sample = dataset['train'][i]
            logger.info(f"Sample {i+1}: Label={sample['label']} | Text={sample['text'][:100]}...")
        full_vocab_source = TextDataset(dataset['train'])
        vocab = full_vocab_source.vocab
        vocab_size = len(vocab) + 1
        train_data = TextDataset(dataset['train'], vocab=vocab)
        val_data = TextDataset(dataset['validation'], vocab=vocab)
        train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)
        model = LSTMClassifier(
            vocab_size=vocab_size,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            num_classes=5,
            num_layers=config.num_layers,
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Model architecture: %s", model)
        logger.info(f"Total parameters: {n_params} | Trainable: {n_trainable}")
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = OrdinalLoss(num_classes=5)
        patience = 10
        best_run_acc = 0.0
        best_val_loss_for_early_stopping = float('inf')
        epochs_no_improve = 0
        best_run_state = None
        for epoch in range(config.epochs):
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
            wandb.log({
                "epoch": epoch+1, 
                "train_loss": avg_train_loss, 
                "val_loss": val_loss, 
                "val_acc": val_acc
            })
            logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            if val_acc > best_run_acc:
                best_run_acc = val_acc
                best_run_state = model.state_dict()
            if val_loss < best_val_loss_for_early_stopping:
                best_val_loss_for_early_stopping = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        global_best_acc = 0.0
        if os.path.exists(global_best_metric_path):
            with open(global_best_metric_path, 'r') as f:
                try:
                    global_best_acc = float(f.read().strip())
                except ValueError: pass
        logger.info(f"Run Finished. Local Best Acc: {best_run_acc:.4f} vs Global Best Acc: {global_best_acc:.4f}")
        if best_run_acc > global_best_acc:
            logger.info(f"NEW RECORD! Overwriting best model.")
            with open(global_best_metric_path, 'w') as f:
                f.write(str(best_run_acc))
            torch.save(best_run_state, global_best_model_path)
            with open(global_best_model_path.replace('.pt', '_config.json'), 'w') as f:
                json.dump(dict(config), f)
            with open(global_best_model_path.replace('.pt', '_vocab.json'), 'w') as f:
                json.dump(vocab, f)

if __name__ == '__main__':
    sweep_config = {
        "method": "bayes",
        "metric": {"name": "val_acc", "goal": "maximize"},
        "parameters": {
            "learning_rate": {"min": 0.0001, "max": 0.05},
            "batch_size": {"values": [32, 64, 128]},
            "epochs": {"value": 30},
            "embed_dim": {"values": [50, 100]},
            "hidden_dim": {"values": [64, 128, 256]},
            "num_layers": {"values": [1, 2]}
        }
    }
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
    tracker_file = os.path.join(model_dir, "best_sweep_acc.txt")
    if os.path.exists(tracker_file):
        os.remove(tracker_file)
    sweep_id = wandb.sweep(sweep_config, project="legaltext")
    wandb.agent(sweep_id, function=train, count=100)
    sys.exit(0)