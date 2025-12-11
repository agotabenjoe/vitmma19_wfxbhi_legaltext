import os
import json
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch

class LegalTextDataset(Dataset):
    def __init__(self, folder, tokenizer, max_length=128):
        self.samples = []
        for file in os.listdir(folder):
            if file.endswith('.json'):
                with open(os.path.join(folder, file), encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        text = item['data']['text']
                        ann = item['annotations'][0]['result'] if item['annotations'] and item['annotations'][0]['result'] else None
                        if ann:
                            label_str = ann[0]['value']['choices'][0]
                            label = int(label_str.split('-')[0]) - 1  # 0-based
                            self.samples.append((text, label))
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        text, label = self.samples[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item

def main():
    folder = 'consensus/consensus'
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    dataset = LegalTextDataset(folder, tokenizer)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=5)
    training_args = TrainingArguments(
        output_dir='./bert_output',
        num_train_epochs=1,
        per_device_train_batch_size=8,
        logging_steps=10,
        save_steps=50,
        evaluation_strategy='no',
        learning_rate=2e-5,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    print('Training finished. Model saved to ./bert_output')

if __name__ == '__main__':
    main()
