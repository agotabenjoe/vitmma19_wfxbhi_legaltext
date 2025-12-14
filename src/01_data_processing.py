import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import logger
import os
import json
import requests
import zipfile
import re
import shutil
import glob
from collections import Counter
from datasets import Dataset, DatasetDict
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import string

logger.info("Setting up NLTK resources...")

def download_nltk_resource(resource):
    try:
        nltk.data.find(resource)
    except LookupError:
        logger.info(f"Downloading NLTK resource: {resource}")
        nltk.download(resource.split('/')[-1])

download_nltk_resource('tokenizers/punkt')
download_nltk_resource('corpora/stopwords')

stemmer = SnowballStemmer('hungarian')
stop_words = set(stopwords.words('hungarian'))

def download_and_extract_zip(url, extract_to):
    zip_path = os.path.join(extract_to, 'downloaded_data.zip')
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    if any(os.scandir(extract_to)) and not os.path.exists(zip_path):
        logger.info(f"Data directory {extract_to} is not empty. Skipping download.")
        return
    logger.info(f"Downloading data from {url}...")
    session = requests.Session()
    response = session.get(url, stream=True)
    if response.status_code != 200:
        raise Exception(f"Failed to download: HTTP {response.status_code}")
    with open(zip_path, 'wb') as out_file:
        for chunk in response.iter_content(chunk_size=8192):
            out_file.write(chunk)
    logger.info("Extracting zip file...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info("Download and extraction complete.")
    except zipfile.BadZipFile:
        logger.error("Error: Invalid zip file.")
        raise
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)

def parse_label(label_str):
    try:
        digit = int(re.search(r'\d+', str(label_str)).group())
        return max(0, min(4, digit - 1))
    except Exception as e:
        logger.error(f"Label parsing error: {label_str} | {e}")
        return None

def load_consensus_data(consensus_folder):
    json_files = glob.glob(os.path.join(consensus_folder, "*.json"))
    logger.info(f"Found {len(json_files)} consensus files.")
    text_to_annotations = {}
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, list): data = [data]
                for item in data:
                    raw_text = item.get('data', {}).get('text', '').strip()
                    if not raw_text: continue
                    for ann in item.get('annotations', []):
                        if ann.get('result'):
                            for res in ann['result']:
                                if res.get('type') == 'choices':
                                    val = res.get('value', {}).get('choices', [])
                                    if val:
                                        if raw_text not in text_to_annotations:
                                            text_to_annotations[raw_text] = []
                                        text_to_annotations[raw_text].append(val[0])
        except Exception as e:
            logger.warning(f"Error reading {json_file}: {e}")
    test_samples = []
    forbidden_texts = set()
    for raw_text, labels in text_to_annotations.items():
        counts = Counter(labels)
        most_common = counts.most_common()
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            continue
        majority_label_str = most_common[0][0]
        label_idx = parse_label(majority_label_str)
        if label_idx is not None:
            test_samples.append({'text': raw_text, 'label': label_idx})
            forbidden_texts.add(raw_text)
    logger.info(f"Consensus (Test) Set created: {len(test_samples)} samples.")
    return test_samples, forbidden_texts

def load_raw_training_data(data_dir, forbidden_texts):
    clean_samples = []
    seen_texts_in_train = set()
    skipped_test_leakage = 0
    skipped_duplicate = 0
    for root, dirs, files in os.walk(data_dir):
        if 'consensus' in root.split(os.sep):
            continue
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        items = json.load(f)
                        if isinstance(items, dict): items = [items]
                        for item in items:
                            if not isinstance(item, dict): continue
                            raw_text = item.get('data', {}).get('text', '')
                            if not raw_text: continue
                            raw_text = raw_text.strip()
                            if raw_text in forbidden_texts:
                                skipped_test_leakage += 1
                                continue
                            if raw_text in seen_texts_in_train:
                                skipped_duplicate += 1
                                continue
                            annotations = item.get('annotations', [])
                            if annotations and annotations[0].get('result'):
                                result = annotations[0]['result'][0]
                                if 'value' in result and 'choices' in result['value']:
                                    label_str = result['value']['choices'][0]
                                    label_idx = parse_label(label_str)
                                    if label_idx is not None:
                                        clean_samples.append({'text': raw_text, 'label': label_idx})
                                        seen_texts_in_train.add(raw_text)
                except Exception:
                    pass
    logger.info(f"Raw Data Loaded: {len(clean_samples)} valid samples.")
    logger.info(f"Skipped {skipped_test_leakage} samples overlapping with Test Set.")
    logger.info(f"Skipped {skipped_duplicate} duplicate samples within Training Set.")
    return clean_samples

def preprocess_batch(batch):
    processed_texts = []
    for text in batch['text']:
        tokens = text.split()
        filtered_tokens = [t for t in tokens if t.lower() not in stop_words]
        processed_texts.append(" ".join(filtered_tokens))
    return {"text": processed_texts}

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '..', 'data')
    final_output_dir = os.path.join(base_dir, '..', 'data', 'final_split_dataset')
    zip_url = "https://bmeedu-my.sharepoint.com/:u:/g/personal/gyires-toth_balint_vik_bme_hu/IQDYwXUJcB_jQYr0bDfNT5RKARYgfKoH97zho3rxZ46KA1I?e=iFp3iz&download=1"
    try:
        download_and_extract_zip(zip_url, data_dir)
        actual_cons_dir = None
        for root, dirs, files in os.walk(data_dir):
            if 'consensus' in dirs:
                actual_cons_dir = os.path.join(root, 'consensus')
                break
        if not actual_cons_dir:
            print("Warning: Consensus folder not found. Creating empty test set.")
            test_samples, forbidden_texts = [], set()
        else:
            test_samples, forbidden_texts = load_consensus_data(actual_cons_dir)
        raw_samples = load_raw_training_data(data_dir, forbidden_texts)
        if not raw_samples:
            print("No training data found.")
            exit(1)
        logger.info("Applying NLTK preprocessing (Stemming + Stopwords)...")
        full_train_ds = Dataset.from_list(raw_samples)
        test_ds = Dataset.from_list(test_samples)
        full_train_ds = full_train_ds.map(preprocess_batch, batched=True)
        test_ds = test_ds.map(preprocess_batch, batched=True)
        logger.info("Preprocessing complete.")
        splits = full_train_ds.train_test_split(test_size=0.1, seed=42)
        train_ds = splits['train']
        val_ds = splits['test']
        print("\n" + "="*40)
        print("VERIFYING & REMOVING OVERLAPS BETWEEN SPLITS")
        print("="*40)
        train_texts = set(train_ds['text'])
        val_texts = set(val_ds['text'])
        test_texts = set(test_ds['text'])
        train_ds = train_ds.filter(lambda ex: ex['text'] not in test_texts)
        val_ds = val_ds.filter(lambda ex: ex['text'] not in test_texts and ex['text'] not in set(train_ds['text']))
        train_texts = set(train_ds['text'])
        val_texts = set(val_ds['text'])
        test_texts = set(test_ds['text'])
        train_val_overlap = train_texts.intersection(val_texts)
        train_test_overlap = train_texts.intersection(test_texts)
        val_test_overlap = val_texts.intersection(test_texts)
        if train_val_overlap or train_test_overlap or val_test_overlap:
            print(f"Overlap removal failed. Still found overlaps.")
            print(f"Train/Val: {len(train_val_overlap)}, Train/Test: {len(train_test_overlap)}, Val/Test: {len(val_test_overlap)}")
            exit(1)
        else:
            print("VERIFIED: Zero overlap between Train, Validation, and Test sets after removal.")
        final_dataset = DatasetDict({
            'train': train_ds,
            'validation': val_ds,
            'test': test_ds
        })
        def print_class_dist(ds, name):
            from collections import Counter
            counts = Counter(ds['label'])
            logger.info(f"Class distribution in {name} set:")
            for k in sorted(counts):
                logger.info(f"  Class {k}: {counts[k]}")
        print_class_dist(train_ds, 'train')
        print_class_dist(val_ds, 'validation')
        print_class_dist(test_ds, 'test')
        final_dataset.save_to_disk(final_output_dir)
        logger.info("\n" + "="*40)
        logger.info("DATASET SAVED SUCCESSFULLY")
        logger.info(f"Location: {final_output_dir}")
        logger.info("="*40)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())