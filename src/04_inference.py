import os
import json
import torch
import re
import gradio as gr
import torch.nn.functional as F

# Import your model class
# Ensure model.py is in the same directory
from model import LSTMClassifier

# --- CONFIGURATION ---
MODEL_PATH = "models/best_sweep_model.pt"  # Adjust path if needed
CONFIG_PATH = MODEL_PATH.replace('.pt', '_config.json')
VOCAB_PATH = MODEL_PATH.replace('.pt', '_vocab.json')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CLASS MAP ---
# Map the output indices (0-4) to your actual labels
LABELS = {
    0: "1 (Low)",
    1: "2",
    2: "3",
    3: "4",
    4: "5 (High)"
}

def load_artifacts():
    """Loads the model, configuration, and vocabulary."""
    print(f"Loading model from {MODEL_PATH}...")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    # 1. Load Config & Vocab
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    with open(VOCAB_PATH, 'r') as f:
        vocab = json.load(f)
        
    # 2. Initialize Model
    # Determine if model was trained as bidirectional based on config or default logic
    # (If your config doesn't have 'bidirectional' key, you might need to hardcode True/False based on what you trained)
    is_bidirectional = config.get('bidirectional', True) 

    model = LSTMClassifier(
        vocab_size=len(vocab) + 1,
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        num_classes=5,
        num_layers=config.get('num_layers', 1),
        # Note: If you haven't updated model.py to accept 'bidirectional' in __init__, 
        # you might need to remove this arg or update model.py first.
    )
    
    # 3. Load Weights
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    
    return model, vocab, config

# Load artifacts once at startup
try:
    model, vocab, config = load_artifacts()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Please ensure you have trained the model and the paths are correct.")
    exit(1)

def preprocess_text(text, vocab):
    """Converts raw text to a tensor of indices using the vocab."""
    # Simple tokenization matching your training extraction
    # Adjust regex if you used different preprocessing
    words = re.findall(r'\w+', text.lower()) 
    
    indices = [vocab.get(w, 0) for w in words] # 0 is usually padding/unknown
    
    if not indices:
        return torch.tensor([0], device=DEVICE).unsqueeze(0)
        
    return torch.tensor(indices, device=DEVICE).unsqueeze(0) # Add batch dim

def classify_text(text):
    """Gradio prediction function."""
    if not text or not text.strip():
        return None

    # Preprocess
    inputs = preprocess_text(text, vocab)
    
    # Inference
    with torch.no_grad():
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)
    
    # Format for Gradio (returns a dict of {label: probability})
    # probs is shape [1, 5], squeeze to [5]
    probs = probs.squeeze().cpu().numpy()
    
    results = {}
    for idx, score in enumerate(probs):
        label_name = LABELS[idx]
        results[label_name] = float(score)
    
    return results

# --- GRADIO INTERFACE ---
if __name__ == "__main__":
    demo = gr.Interface(
        fn=classify_text,
        inputs=gr.Textbox(lines=5, placeholder="Enter legal text here..."),
        outputs=gr.Label(num_top_classes=5),
        title="Legal Text Classifier",
        description="Enter text to classify its complexity/grade on a scale of 1 (Low) to 5 (High).",
        examples=[
            ["The tenant shall pay the rent on the first day of every month."],
            ["Notwithstanding anything to the contrary contained herein, the indemnification obligations shall survive the termination of this agreement."]
        ]
    )
    
    print("🚀 Launching Gradio on http://localhost:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)