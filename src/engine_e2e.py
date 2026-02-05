import torch
import numpy as np
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.nist_score import corpus_nist
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

class NLGEvaluator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.cider_scorer = Cider()
        try:
            self.meteor_scorer = Meteor()
        except Exception as e:
            print(f"Warning: METEOR initialization failed (Java missing?): {e}")
            self.meteor_scorer = None
        
    def compute_metrics(self, predictions, references):
        gts = {i: r_list for i, r_list in enumerate(references)}
        res = {i: [p] for i, p in enumerate(predictions)}

        meteor_val = 0.0
        if self.meteor_scorer:
            try:
                meteor_val, _ = self.meteor_scorer.compute_score(gts, res)
                meteor_val = meteor_val * 100
            except:
                meteor_val = 0.0

        refs_tokenized = [[nltk.word_tokenize(r) for r in ref_list] for ref_list in references]
        preds_tokenized = [nltk.word_tokenize(p) for p in predictions]
        
        bleu = corpus_bleu(refs_tokenized, preds_tokenized, smoothing_function=SmoothingFunction().method1) * 100
        try:
            nist = corpus_nist(refs_tokenized, preds_tokenized)
        except:
            nist = 0.0

        rouge_l_scores = []
        for p, r_list in zip(predictions, references):
            scores = [self.rouge_scorer.score(r, p)['rougeL'].fmeasure for r in r_list]
            rouge_l_scores.append(max(scores) if scores else 0)
        rouge_l = np.mean(rouge_l_scores) * 100
        
        cider_score, _ = self.cider_scorer.compute_score(gts, res)
        
        return {
            'BLEU': bleu,
            'NIST': nist,
            'METEOR': meteor_val,
            'ROUGE-L': rouge_l,
            'CIDEr': cider_score
        }

def train_epoch(model, data_loader, optimizer, device, scheduler):
    model.train()
    losses = []
    progress_bar = tqdm(total=len(data_loader), desc="Training", leave=False)
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        progress_bar.update(1)
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    progress_bar.close()
    return np.mean(losses)

def evaluate_generation(model, data_loader, device, tokenizer, evaluator, desc="Evaluating"):
    model.eval()
    all_preds = []
    all_refs = []
    all_mrs = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=desc):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            full_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=64,
                num_beams=10,
                length_penalty=0.8,
                no_repeat_ngram_size=0,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            gen_only = full_outputs[:, input_ids.shape[1]:]
            generated_text = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
            
            for i, text in enumerate(generated_text):
                clean_gen = text.strip()
                all_preds.append(clean_gen)
                all_refs.append(batch['refs'][i])
                all_mrs.append(batch['mr'][i])

    metrics = evaluator.compute_metrics(all_preds, all_refs)
    return metrics, all_preds, all_refs, all_mrs