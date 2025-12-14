import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import logger
import os
import json
import glob
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from torch.utils.data import DataLoader
from datasets import load_from_disk
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import LSTMClassifier, TextDataset
import config

def analyze_consensus_quality(consensus_folder, output_dir):
    logger.info("\n" + "="*40)
    logger.info("STEP 1: ANALYZING CONSENSUS DATA QUALITY")
    logger.info("="*40)
    json_files = glob.glob(os.path.join(consensus_folder, "*.json"))
    if not json_files:
        logger.warning(f"No consensus files found in {consensus_folder}. Skipping analysis.")
        return
    stats = {
        'total_tasks': 0,
        'perfect_alignment': 0,
        'disagreement': 0,
        'ties': 0,
        'avg_agreement_rates': []
    }
    logger.info(f"   Processing {len(json_files)} annotator files...")
    text_map = defaultdict(list)
    for jf in json_files:
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, list): data = [data]
                for item in data:
                    text = item.get('data', {}).get('text', '').strip()
                    if not text: continue
                    for ann in item.get('annotations', []):
                        if ann.get('result'):
                            val = ann['result'][0].get('value', {}).get('choices', [])
                            if val:
                                text_map[text].append(val[0])
        except Exception as e:
            logger.warning(f"Error reading {jf}: {e}")
    for text, labels in text_map.items():
        stats['total_tasks'] += 1
        n_votes = len(labels)
        if n_votes < 2: continue
        counts = Counter(labels)
        most_common = counts.most_common()
        majority_count = most_common[0][1]
        agreement_rate = majority_count / n_votes
        stats['avg_agreement_rates'].append(agreement_rate)
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            stats['ties'] += 1
        elif agreement_rate == 1.0:
            stats['perfect_alignment'] += 1
        else:
            stats['disagreement'] += 1
    avg_agreement = np.mean(stats['avg_agreement_rates']) * 100 if stats['avg_agreement_rates'] else 0
    logger.info(f"\n   Total Unique Tasks Analyzed: {stats['total_tasks']}")
    logger.info(f"   Perfect Alignment (100% Agree): {stats['perfect_alignment']} ({(stats['perfect_alignment']/stats['total_tasks']*100):.1f}%)")
    logger.info(f"   Disagreement (Majority Exists): {stats['disagreement']} ({(stats['disagreement']/stats['total_tasks']*100):.1f}%)")
    logger.info(f"   Ties (Ambiguous/Dropped):       {stats['ties']} ({(stats['ties']/stats['total_tasks']*100):.1f}%)")
    logger.info(f"   Average Agreement Rate:         {avg_agreement:.2f}%")
    if stats['total_tasks'] > 0:
        plt.figure(figsize=(8, 6))
        labels = ['Perfect Alignment', 'Minor Disagreement', 'Ties (Dropped)']
        sizes = [stats['perfect_alignment'], stats['disagreement'], stats['ties']]
        colors = ['#2ecc71', '#f1c40f', '#e74c3c']
        labels = [l for l, s in zip(labels, sizes) if s > 0]
        colors = [c for c, s in zip(colors, sizes) if s > 0]
        sizes = [s for s in sizes if s > 0]
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
        plt.title('Consensus Data Quality: Annotator Agreement')
        save_path = os.path.join(output_dir, 'consensus_quality_chart.png')
        plt.savefig(save_path)
        logger.info(f"   Saved Data Quality Chart: {save_path}")
        plt.close()

def extract_features(text):
    if not text: return [0, 0, 0]
    sentences = re.split(r'[.!?\n]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = re.findall(r'\w+', text)
    return [
        len(text),
        np.mean([len(s.split()) for s in sentences]) if sentences else 0,
        np.mean([len(w) for w in words]) if words else 0
    ]

def plot_confusion_matrix(y_true, y_pred, title, save_path, labels=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label (Consensus)')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"   Saved Matrix: {save_path}")
    plt.close()

def get_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1 (Weighted)": f1_score(y_true, y_pred, average='weighted', zero_division=0),
        "Precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_true, y_pred, average='weighted', zero_division=0)
    }

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(base_dir, '..')
    model_dir = os.path.join(root_dir, 'models')
    data_dir = os.path.join(root_dir, 'data')
    dataset_path = os.path.join(data_dir, 'final_split_dataset')
    consensus_raw_path = os.path.join(data_dir, 'legaltextdecoder', 'consensus')
    output_dir = os.path.join(root_dir, 'log')
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"[DEBUG] Checking consensus folder at: {consensus_raw_path}")
    if os.path.exists(consensus_raw_path):
        logger.info(f"[DEBUG] Consensus folder found.")
        analyze_consensus_quality(consensus_raw_path, output_dir)
    else:
        logger.warning(f"Consensus raw folder not found at {consensus_raw_path}. Skipping data quality analysis.")
    logger.info("\n" + "="*40)
    logger.info("STEP 2: LOADING DATA & MODELS")
    logger.info("="*40)
    if not os.path.exists(dataset_path):
        logger.error(f"Error: Dataset not found at {dataset_path}")
        return
    dataset = load_from_disk(dataset_path)
    train_ds = dataset['train']
    test_ds = dataset['test']
    logger.info(f"   Train Set: {len(train_ds)} samples")
    logger.info(f"   Test Set (Consensus): {len(test_ds)} samples")
    class_labels = ["1 (Low)", "2", "3", "4", "5 (High)"]
    logger.info("\n   --- A. Training Baseline (Logistic Regression) ---")
    X_train = np.array([extract_features(x['text']) for x in train_ds])
    y_train = np.array([int(x['label']) - 1 for x in train_ds])
    X_test = np.array([extract_features(x['text']) for x in test_ds])
    y_test = np.array([int(x['label']) - 1 for x in test_ds])
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    clf = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)
    base_preds = clf.predict(X_test_scaled)
    base_metrics = get_metrics(y_test, base_preds)
    plot_confusion_matrix(y_test, base_preds, "Baseline Confusion Matrix",
                          os.path.join(output_dir, "baseline_confusion_matrix.png"), class_labels)
    logger.info("\nExample Baseline Predictions:")
    for i in range(min(5, len(test_ds))):
        logger.info(f"Sample {i+1}:")
        logger.info(f"   Ground Truth (0-4): {y_test[i]}, Predicted: {base_preds[i]}")
        logger.info("-" * 30)
    logger.info("\n   --- B. Evaluating Best LSTM ---")
    weights_path = os.path.join(root_dir, config.BEST_MODEL_PATH)
    if os.path.exists(weights_path):
        config_path = weights_path.replace('.pt', '_config.json')
        vocab_path = weights_path.replace('.pt', '_vocab.json')
        with open(config_path, 'r') as f: model_config = json.load(f)
        with open(vocab_path, 'r') as f: vocab = json.load(f)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_data_objs = [{'text': x['text'], 'label': x['label']} for x in test_ds]
        lstm_test_ds = TextDataset(test_data_objs, vocab=vocab)
        test_loader = DataLoader(lstm_test_ds, batch_size=32, shuffle=False)
        model = LSTMClassifier(
            vocab_size=len(vocab) + 1,
            embed_dim=model_config['embed_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_classes=5,
            num_layers=model_config['num_layers']
        ).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        lstm_preds = []
        lstm_targets = []
        first_batch = True
        with torch.no_grad():
            for x, y in test_loader:
                if first_batch:
                    logger.info(f"[DEBUG] LSTM First Batch Labels (Raw from Loader): {y[:10]}")
                    first_batch = False
                x = x.to(device)
                outputs = model(x)
                _, predicted = torch.max(outputs.data, 1)
                lstm_preds.extend(predicted.cpu().numpy())
                lstm_targets.extend(y.numpy())
        lstm_metrics = get_metrics(lstm_targets, lstm_preds)
        plot_confusion_matrix(lstm_targets, lstm_preds, "LSTM Confusion Matrix",
                              os.path.join(output_dir, "lstm_confusion_matrix.png"), class_labels)
        logger.info("\nExample LSTM Predictions:")
        for i in range(min(5, len(lstm_targets))):
            logger.info(f"Sample {i+1}:")
            logger.info(f"   Ground Truth: {lstm_targets[i]}, Predicted: {lstm_preds[i]}")
            logger.info("-" * 30)
    else:
        logger.error(f"LSTM weights not found at: {weights_path}")
        lstm_metrics = None
    logger.info("\n" + "="*60)
    logger.info("FINAL EVALUATION REPORT")
    logger.info("="*60)
    logger.info(f"{'Metric':<25} | {'Baseline':<12} | {'LSTM':<12}")
    logger.info("-" * 60)
    if lstm_metrics:
        for key in base_metrics:
            logger.info(f"{key:<25} | {base_metrics[key]:.4f}       | {lstm_metrics[key]:.4f}")
    logger.info("="*60)
    logger.info(f"Analysis complete. Results saved in: {output_dir}")

if __name__ == "__main__":
    main()