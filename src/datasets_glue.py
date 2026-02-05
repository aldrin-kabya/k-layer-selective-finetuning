import torch
import pandas as pd
from torch.utils.data import Dataset
import os

class GLUEDataset(Dataset):
    def __init__(self, file_path, task_name, tokenizer, max_len):
        self.task_name = task_name
        self.tokenizer = tokenizer
        self.max_len = max_len

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if task_name == 'rte':
            self.data = pd.read_csv(file_path, sep='\t', quoting=3)
            self.data['label'] = self.data['label'].map({'entailment': 0, 'not_entailment': 1})
        elif task_name == 'mrpc':
            self.data = pd.read_csv(file_path, sep='\t', quoting=3)
            # MRPC usually has header, but quoting=3 can mess it up if not careful. 
            # Assuming standard GLUE format:
            self.data.columns = ['label', 'id1', 'id2', 'sentence1', 'sentence2']
            self.data['label'] = self.data['label'].astype(int)
        elif task_name == 'cola':
            self.data = pd.read_csv(file_path, sep='\t', quoting=3, header=None, names=['id', 'label', 'misc', 'sentence'])
        elif task_name == 'qnli':
            self.data = pd.read_csv(file_path, sep='\t', quoting=3)
            self.data['label'] = self.data['label'].map({'entailment': 0, 'not_entailment': 1})
        elif task_name == 'sst2':
            self.data = pd.read_csv(file_path, sep='\t', quoting=3)
            self.data.columns = ['sentence', 'label']
            self.data['label'] = self.data['label'].astype(int)
        elif task_name in ['sst5', 'mr', 'cr', 'mpqa', 'subj', 'trec']:
            self.data = pd.read_csv(file_path, header=None, names=['label', 'sentence'])
            self.data['label'] = self.data['label'].astype(int)
        elif task_name in ['mnli', 'mnli-mm']:
            self.data = pd.read_csv(file_path, sep='\t', quoting=3)
            self.data = self.data[['sentence1', 'sentence2', 'gold_label']]
            self.data['label'] = self.data['gold_label'].map({'entailment': 0, 'neutral': 1, 'contradiction': 2})
        elif task_name == 'snli':
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines[1:]:  # Skip header
                    parts = line.strip().split('\t')
                    if len(parts) >= 8:
                        data.append({
                            'sentence1': parts[7],
                            'sentence2': parts[8],
                            'gold_label': parts[-1]
                        })
            self.data = pd.DataFrame(data)
            self.data['label'] = self.data['gold_label'].map({'entailment': 0, 'neutral': 1, 'contradiction': 2})
        elif task_name == 'qqp':
            self.data = pd.read_csv(file_path, sep='\t', quoting=3)
            self.data.columns = ['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate']
            self.data['label'] = self.data['is_duplicate'].astype(int)
        else:
            raise ValueError(f"Unsupported task: {task_name}")

        # Handle NaN values
        cols_to_check = ['sentence', 'sentence1', 'sentence2']
        for col in cols_to_check:
            if col in self.data.columns:
                self.data[col] = self.data[col].fillna('')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        row = self.data.iloc[item]

        if self.task_name in ['rte', 'mrpc', 'mnli', 'mnli-mm', 'snli']:
            text_a = row['sentence1']
            text_b = row['sentence2']
        elif self.task_name == 'qqp':
            text_a = row['question1']
            text_b = row['question2']
        elif self.task_name == 'qnli':
            text_a = row['question']
            text_b = row['sentence']
        elif self.task_name in ['cola', 'sst2', 'sst5', 'mr', 'cr', 'mpqa', 'subj', 'trec']:
            text_a = row['sentence']
            text_b = None

        label = row['label']

        encoding = self.tokenizer.encode_plus(
            text_a,
            text_b,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }