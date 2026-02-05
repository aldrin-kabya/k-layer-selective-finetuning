import torch
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset

class E2EDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = pd.read_csv(file_path)
        self.data = self.data.dropna()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        row = self.data.iloc[item]
        mr = str(row['mr'])
        ref = str(row['ref'])

        mr_tokens = self.tokenizer(f"{mr} {self.tokenizer.bos_token}", add_special_tokens=False)['input_ids']
        ref_tokens = self.tokenizer(f"{ref} {self.tokenizer.eos_token}", add_special_tokens=False)['input_ids']

        combined_ids = mr_tokens + ref_tokens
        combined_labels = [-100] * len(mr_tokens) + ref_tokens

        if len(combined_ids) > self.max_len:
            combined_ids = combined_ids[:self.max_len]
            combined_labels = combined_labels[:self.max_len]
            attention_mask = [1] * self.max_len
        else:
            pad_len = self.max_len - len(combined_ids)
            combined_ids += [self.tokenizer.pad_token_id] * pad_len
            combined_labels += [-100] * pad_len
            attention_mask = [1] * (len(combined_ids) - pad_len) + [0] * pad_len

        return {
            'input_ids': torch.tensor(combined_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(combined_labels, dtype=torch.long)
        }

class E2EEvalDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        df = pd.read_csv(file_path)
        df = df.dropna()
        self.grouped_data = df.groupby('mr')['ref'].apply(list).reset_index()

    def __len__(self):
        return len(self.grouped_data)

    def __getitem__(self, item):
        row = self.grouped_data.iloc[item]
        mr = str(row['mr'])
        refs = row['ref']

        text = f"{mr} {self.tokenizer.bos_token}"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'mr': mr,
            'refs': refs
        }

def eval_collate_fn(batch):
    max_len = max([b['input_ids'].shape[0] for b in batch])
    input_ids_list = []
    attention_mask_list = []
    mrs = []
    refs = []
    
    for b in batch:
        pad_len = max_len - b['input_ids'].shape[0]
        # Left Padding for Generation
        input_ids = F.pad(b['input_ids'], (pad_len, 0), value=50256) 
        attention_mask = F.pad(b['attention_mask'], (pad_len, 0), value=0)
        
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        mrs.append(b['mr'])
        refs.append(b['refs'])
        
    return {
        'input_ids': torch.stack(input_ids_list),
        'attention_mask': torch.stack(attention_mask_list),
        'mr': mrs,
        'refs': refs
    }