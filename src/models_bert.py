import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from tqdm import tqdm
import time
import numpy as np
import random

class BertLayerSelection(nn.Module):
    def __init__(self, n_classes, k=3, finetune_method='highest', use_all_layers=True):
        super(BertLayerSelection, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.num_layers = self.bert.config.num_hidden_layers
        self.hidden_size = self.bert.config.hidden_size
        self.k = min(k, self.num_layers)
        self.finetune_method = finetune_method
        self.use_all_layers = use_all_layers
        
        self.layer_selection_time = 0
        self.finetuning_time = 0
        self.dropout = nn.Dropout(0.3)
        
        if self.use_all_layers:
            self.out = nn.Sequential(
                nn.Linear(self.hidden_size * self.num_layers, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, n_classes)
            )
        else:
            self.out = nn.Sequential(
                nn.Linear(self.hidden_size * self.k, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, n_classes)
            )
        
        for param in self.bert.parameters():
            param.requires_grad = False
        self.selected_indices = None

    def select_layers(self, dataloader):
        start_time = time.time()
        self.bert.eval()
        device = next(self.parameters()).device
        
        layer_similarity_sums = torch.zeros(self.num_layers).to(device)
        total_samples = 0
        pbar = tqdm(total=len(dataloader), desc="Selecting Layers")

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                hidden_states = outputs.hidden_states[1:]
                
                cls_tokens = torch.stack([hs[:, 0, :] for hs in hidden_states])
                
                for i in range(self.num_layers):
                    similarity_sum = 0
                    count = 0
                    for j in range(self.num_layers):
                        if i != j:
                            similarity = F.cosine_similarity(cls_tokens[i], cls_tokens[j], dim=1).mean()
                            similarity_sum += similarity
                            count += 1
                    layer_similarity_sums[i] += similarity_sum / count
                
                total_samples += 1
                pbar.update(1)
        
        pbar.close()
        self.layer_scores = layer_similarity_sums / total_samples

        if self.finetune_method == 'highest':
            _, self.selected_indices = torch.topk(self.layer_scores, k=self.k)
        else:
            _, self.selected_indices = torch.topk(-self.layer_scores, k=self.k)
        
        self.selected_indices = self.selected_indices.cpu().numpy()
        self.layer_selection_time = time.time() - start_time

    def finetune_selected_layers(self):
        start_time = time.time()
        for i in range(self.num_layers):
            if i in self.selected_indices:
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = True
            else:
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = False
        self.finetuning_time = time.time() - start_time

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states[1:]
        cls_tokens = torch.stack([hs[:, 0, :] for hs in hidden_states], dim=1)
        batch_size = cls_tokens.size(0)
        
        if self.use_all_layers:
            all_cls = cls_tokens.view(batch_size, -1)
            output = self.out(self.dropout(all_cls))
        else:
            selected_cls = cls_tokens[:, self.selected_indices, :]
            selected_cls = selected_cls.view(batch_size, -1)
            output = self.out(self.dropout(selected_cls))
        return output

class BertBlockwiseLayerSelection(BertLayerSelection):
    def __init__(self, n_classes, k=4, B=4, finetune_method='blockwise_highest', use_all_layers=True):
        # We inherit initialization partially but need to override logic
        nn.Module.__init__(self) # Call grand-parent init
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.num_layers = self.bert.config.num_hidden_layers
        self.hidden_size = self.bert.config.hidden_size
        self.k = k
        self.B = B
        
        if self.num_layers % B != 0:
            raise ValueError(f"Layers ({self.num_layers}) must be divisible by B ({B}).")
        if self.k % B != 0:
            raise ValueError(f"k ({self.k}) must be divisible by B ({B}).")

        self.finetune_method = finetune_method
        self.use_all_layers = use_all_layers
        self.layer_selection_time = 0
        self.finetuning_time = 0
        self.dropout = nn.Dropout(0.3)
        
        if self.use_all_layers:
            self.out = nn.Sequential(
                nn.Linear(self.hidden_size * self.num_layers, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, n_classes)
            )
        else:
            self.out = nn.Sequential(
                nn.Linear(self.hidden_size * self.k, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, n_classes)
            )
        
        for param in self.bert.parameters():
            param.requires_grad = False
        self.selected_indices = None

    def select_layers(self, dataloader):
        # Re-use the similarity calculation from Parent, but change selection logic
        super().select_layers(dataloader)
        
        # Override selection logic based on blocks
        block_size = self.num_layers // self.B
        k_per_block = self.k // self.B
        selected_indices_list = []
        
        for i in range(self.B):
            start_idx = i * block_size
            end_idx = start_idx + block_size
            block_scores = self.layer_scores[start_idx:end_idx]
            
            if 'blockwise_highest' in self.finetune_method:
                _, relative_indices = torch.topk(block_scores, k=k_per_block)
            else: # blockwise_lowest
                _, relative_indices = torch.topk(-block_scores, k=k_per_block)
            
            absolute_indices = relative_indices + start_idx
            selected_indices_list.append(absolute_indices)
            
        self.selected_indices = torch.cat(selected_indices_list).cpu().numpy()
        self.selected_indices = np.sort(self.selected_indices)

class BertRandomLayerSelection(nn.Module):
    def __init__(self, n_classes, k):
        super(BertRandomLayerSelection, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.num_layers = self.bert.config.num_hidden_layers
        self.hidden_size = self.bert.config.hidden_size
        self.k = k
        self.selected_layers = random.sample(range(self.num_layers), k)
        self.dropout = nn.Dropout(0.3)
        
        self.out = nn.Sequential(
            nn.Linear(self.hidden_size * self.num_layers, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, n_classes)
        )
        
        for param in self.bert.parameters():
            param.requires_grad = False
        
        for layer_idx in self.selected_layers:
            for param in self.bert.encoder.layer[layer_idx].parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states[1:]
        cls_tokens = torch.stack([hs[:, 0, :] for hs in hidden_states], dim=1)
        batch_size = cls_tokens.size(0)
        all_cls = cls_tokens.view(batch_size, -1)
        output = self.out(self.dropout(all_cls))
        return output

class BERTFinetune(nn.Module):
    def __init__(self, n_classes):
        super(BERTFinetune, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.num_layers = self.bert.config.num_hidden_layers
        self.hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * self.num_layers, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, n_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states[1:]
        cls_tokens = torch.stack([hs[:, 0, :] for hs in hidden_states], dim=1)
        all_cls = cls_tokens.view(cls_tokens.size(0), -1)
        logits = self.classifier(self.dropout(all_cls))
        return logits