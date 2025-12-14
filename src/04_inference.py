import os
import json
import torch
import re
import gradio as gr
import torch.nn.functional as F
from model import LSTMClassifier

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
    print(f"Loading model from {MODEL_PATH}...")
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

try:
    model, vocab, config = load_artifacts()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure you have trained the model and the paths are correct.")
    exit(1)

def preprocess_text(text, vocab):
    words = re.findall(r'\w+', text.lower())
    indices = [vocab.get(w, 0) for w in words]
    if not indices:
        return torch.tensor([0], device=DEVICE).unsqueeze(0)
    return torch.tensor(indices, device=DEVICE).unsqueeze(0)

def classify_text(text):
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

if __name__ == "__main__":
    demo = gr.Interface(
        fn=classify_text,
        inputs=gr.Textbox(lines=5, placeholder="Írja be a jogi szöveget ide..."),
        outputs=gr.Label(num_top_classes=5),
        title="Jogi Szöveg Osztályozó",
        description="Írjon be egy szöveget, hogy besorolja annak komplexitását/nehézségét 1 (alacsony) és 5 (magas) között.",
        examples=[
            ["A bérlő köteles a bérleti díjat minden hónap első napján megfizetni."],
            ["A jelen szerződés bármely rendelkezésével ellentétes kikötés hiányában a kártérítési kötelezettségek a szerződés megszűnését követően is fennmaradnak."]
        ]
    )
    print("Launching Gradio on http://localhost:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)