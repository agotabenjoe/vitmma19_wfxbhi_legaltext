import os
import json
import torch
import re
import torch.nn.functional as F
from model import LSTMClassifier
from utils import logger  # Import the unified logger

MODEL_PATH = "models/best_sweep_model.pt"
CONFIG_PATH = MODEL_PATH.replace('.pt', '_config.json')
VOCAB_PATH = MODEL_PATH.replace('.pt', '_vocab.json')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = {
    0: "1 (Low)",
    1: "2",
    2: "3",
    3: "4",
    4: "5 (High)"
}

def load_artifacts():
    logger.info(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    with open(VOCAB_PATH, 'r') as f:
        vocab = json.load(f)
    is_bidirectional = config.get('bidirectional', True)
    model = LSTMClassifier(
        vocab_size=len(vocab) + 1,
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        num_classes=5,
        num_layers=config.get('num_layers', 1),
    )
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model, vocab, config

def preprocess_text(text, vocab):
    words = re.findall(r'\w+', text.lower())
    indices = [vocab.get(w, 0) for w in words]
    if not indices:
        return torch.tensor([0], device=DEVICE).unsqueeze(0)
    return torch.tensor(indices, device=DEVICE).unsqueeze(0)

def classify_text(text, model, vocab):
    if not text or not text.strip():
        return None
    inputs = preprocess_text(text, vocab)
    with torch.no_grad():
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)
    probs = probs.squeeze().cpu().numpy()
    results = {}
    for idx, score in enumerate(probs):
        label_name = LABELS[idx]
        results[label_name] = float(score)
    return results

# Sample Hungarian ÁSZF texts (generic examples for inference)
sample_texts = [
    "A szolgáltató fenntartja a jogot a szolgáltatás módosítására bármikor.",
    "A felhasználó köteles a személyes adatokat pontosan megadni.",
    "A szerződés megszűnésével a felek közötti jogviszony véglegesen megszűnik.",
    "A vitás kérdések esetén a bíróság illetékes.",
    "A jelen feltételek bármely módosítása írásban történik."
]

if __name__ == "__main__":
    try:
        model, vocab, config = load_artifacts()
        logger.info("Model loaded successfully for inference!")
        
        logger.info("\n--- Sample Inference on Hungarian ÁSZF Texts ---")
        for i, text in enumerate(sample_texts, 1):
            results = classify_text(text, model, vocab)
            if results:
                top_label = max(results, key=results.get)
                logger.info(f"Sample {i}: '{text[:50]}...' -> Predicted: {top_label} (Probabilities: {results})")
            else:
                logger.info(f"Sample {i}: '{text[:50]}...' -> No valid prediction")
        
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        logger.error("Please ensure the model is trained and paths are correct.")
