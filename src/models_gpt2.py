import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Config
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from tqdm import tqdm
import time
import numpy as np
import random

class GPT2LayerSelection(nn.Module):
    def __init__(self, k=3, finetune_method='highest'):
        super(GPT2LayerSelection, self).__init__()
        self.config = GPT2Config.from_pretrained('gpt2')
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', config=self.config)
        self.num_layers = self.config.n_layer
        self.k = min(k, self.num_layers)
        self.finetune_method = finetune_method
        
        for param in self.gpt2.parameters():
            param.requires_grad = False
        self.selected_indices = None
        self.layer_selection_time = 0

    def select_layers(self, dataloader):
        start_time = time.time()
        self.gpt2.eval()
        device = next(self.parameters()).device
        
        layer_similarity_sums = torch.zeros(self.num_layers).to(device)
        total_samples = 0
        # Limit to 50 batches for selection speed
        pbar = tqdm(total=min(len(dataloader), 50), desc="Selecting Layers (Last Token)") 

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= 50: break 
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                hidden_states = outputs.hidden_states[1:] 
                
                last_token_idxs = attention_mask.sum(dim=1) - 1
                layer_reps_list = []
                for hs in hidden_states:
                    last_tokens = hs[torch.arange(hs.size(0)), last_token_idxs]
                    layer_reps_list.append(last_tokens)
                layer_reps = torch.stack(layer_reps_list)
                
                for l_i in range(self.num_layers):
                    similarity_sum = 0
                    count = 0
                    for l_j in range(self.num_layers):
                        if l_i != l_j:
                            sim = F.cosine_similarity(layer_reps[l_i], layer_reps[l_j], dim=1).mean()
                            similarity_sum += sim
                            count += 1
                    layer_similarity_sums[l_i] += similarity_sum / count
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
        for i in self.selected_indices:
            for param in self.gpt2.transformer.h[i].parameters():
                param.requires_grad = True
        for param in self.gpt2.transformer.ln_f.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return CausalLMOutputWithCrossAttentions(loss=loss, logits=logits)
    
    def generate(self, input_ids, attention_mask, **kwargs):
        return self.gpt2.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

class GPT2BlockwiseLayerSelection(GPT2LayerSelection):
    def __init__(self, k=4, B=4, finetune_method='blockwise_highest'):
        nn.Module.__init__(self)
        self.config = GPT2Config.from_pretrained('gpt2')
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', config=self.config)
        self.num_layers = self.config.n_layer
        self.k = k
        self.B = B
        self.finetune_method = finetune_method
        
        if self.num_layers % B != 0:
            raise ValueError(f"Layers ({self.num_layers}) must be divisible by blocks B ({B}).")
        
        for param in self.gpt2.parameters():
            param.requires_grad = False
        self.selected_indices = None
        self.layer_selection_time = 0

    def select_layers(self, dataloader):
        super().select_layers(dataloader)
        
        block_size = self.num_layers // self.B
        k_per_block = self.k // self.B
        selected_indices_list = []
        
        for i in range(self.B):
            start_idx = i * block_size
            end_idx = start_idx + block_size
            block_scores = self.layer_scores[start_idx:end_idx]
            
            if 'highest' in self.finetune_method:
                _, relative_indices = torch.topk(block_scores, k=k_per_block)
            else:
                _, relative_indices = torch.topk(-block_scores, k=k_per_block)
            
            absolute_indices = relative_indices + start_idx
            selected_indices_list.append(absolute_indices)
            
        self.selected_indices = torch.cat(selected_indices_list).cpu().numpy()
        self.selected_indices = np.sort(self.selected_indices)

class GPT2RandomLayerSelection(nn.Module):
    def __init__(self, k):
        super(GPT2RandomLayerSelection, self).__init__()
        self.config = GPT2Config.from_pretrained('gpt2')
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', config=self.config)
        self.num_layers = self.config.n_layer
        self.k = k
        self.selected_layers = sorted(random.sample(range(self.num_layers), k))
        
        for param in self.gpt2.parameters():
            param.requires_grad = False
        
        for layer_idx in self.selected_layers:
            for param in self.gpt2.transformer.h[layer_idx].parameters():
                param.requires_grad = True

        for param in self.gpt2.transformer.ln_f.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return CausalLMOutputWithCrossAttentions(loss=loss, logits=logits)
    
    def generate(self, input_ids, attention_mask, **kwargs):
        return self.gpt2.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

class GPT2Finetune(nn.Module):
    def __init__(self):
        super(GPT2Finetune, self).__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return CausalLMOutputWithCrossAttentions(loss=loss, logits=logits)
    
    def generate(self, input_ids, attention_mask, **kwargs):
        return self.gpt2.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)