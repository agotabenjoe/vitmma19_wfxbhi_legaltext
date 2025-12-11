import os
import json
import numpy as np
import glob
import re
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

# Ordinal cross-entropy loss
def ordinal_cross_entropy(y_true, y_pred, num_classes=5):
	# y_pred: shape (n_samples, num_classes), softmax probabilities
	# y_true: shape (n_samples,), integer labels 1-5
	y_true = np.array(y_true) - 1  # 0-based
	y_true_onehot = np.eye(num_classes)[y_true]
	return log_loss(y_true_onehot, y_pred)

def extract_features(text):
	sentences = re.split(r'[.!?\n]', text)
	sentences = [s.strip() for s in sentences if s.strip()]
	words = re.findall(r'\w+', text)
	text_len = len(text)
	avg_sentence_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0
	avg_word_len = np.mean([len(w) for w in words]) if words else 0
	return [text_len, avg_sentence_len, avg_word_len]

def load_consensus_data(folder):
	X, y = [], []
	for file in glob.glob(os.path.join(folder, '*.json')):
		with open(file, encoding='utf-8') as f:
			data = json.load(f)
			for item in data:
				text = item['data']['text']
				# Find comprehensibility label
				ann = item['annotations'][0]['result'] if item['annotations'] and item['annotations'][0]['result'] else None
				if ann:
					label_str = ann[0]['value']['choices'][0]
					# Extract number from label (e.g. '3-Többé/kevésbé megértem' -> 3)
					label = int(label_str.split('-')[0])
					X.append(extract_features(text))
					y.append(label)
	return np.array(X), np.array(y)

def main():
	consensus_folder = 'consensus/consensus'
	x, y = load_consensus_data(consensus_folder)
	scaler = StandardScaler()
	x_scaled = scaler.fit_transform(x)
	# Logistic regression for ordinal prediction (baseline)
	clf = LogisticRegression(multi_class='multinomial', max_iter=200)
	clf.fit(x_scaled, y)
	y_pred_proba = clf.predict_proba(x_scaled)
	loss = ordinal_cross_entropy(y, y_pred_proba, num_classes=5)
	print(f'Baseline ordinal cross-entropy loss: {loss:.4f}')

if __name__ == '__main__':
	main()
