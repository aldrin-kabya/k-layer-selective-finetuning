import argparse
import os
import torch
import time
import csv
import copy
from tabulate import tabulate
from transformers import GPT2Tokenizer, AdamW, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader

from src.datasets_e2e import E2EDataset, E2EEvalDataset, eval_collate_fn
from src.models_gpt2 import GPT2LayerSelection, GPT2BlockwiseLayerSelection, GPT2RandomLayerSelection, GPT2Finetune
from src.engine_e2e import train_epoch, evaluate_generation, NLGEvaluator
from src.utils import set_seed, count_parameters

def main():
    parser = argparse.ArgumentParser(description="Layer Selection for GPT-2 on E2E NLG")
    parser.add_argument('--data_dir', type=str, default='./data/e2e_data', help="Path to E2E data csv files")
    parser.add_argument('--method', type=str, choices=['highest', 'lowest', 'blockwise_highest', 'blockwise_lowest', 'random', 'full_ft'], default='highest')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--B', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./results_e2e')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading datasets...")
    train_dataset = E2EDataset(os.path.join(args.data_dir, 'trainset.csv'), tokenizer, 128)
    dev_dataset = E2EEvalDataset(os.path.join(args.data_dir, 'devset.csv'), tokenizer, 128)
    test_dataset = E2EEvalDataset(os.path.join(args.data_dir, 'testset_w_refs.csv'), tokenizer, 128)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=eval_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=eval_collate_fn)
    
    if args.method == 'full_ft':
        model = GPT2Finetune()
    elif args.method == 'random':
        model = GPT2RandomLayerSelection(k=args.k)
    elif args.method in ['highest', 'lowest']:
        model = GPT2LayerSelection(k=args.k, finetune_method=args.method)
    elif 'blockwise' in args.method:
        model = GPT2BlockwiseLayerSelection(k=args.k, B=args.B, finetune_method=args.method)
        
    model = model.to(device)
    
    if hasattr(model, 'select_layers'):
        print("Selecting layers...")
        model.select_layers(train_loader)
        print(f"Selected: {model.selected_indices}")
        model.finetune_selected_layers()

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(len(train_loader)*args.epochs*0.1), num_training_steps=len(train_loader)*args.epochs)
    
    evaluator = NLGEvaluator()
    best_bleu = 0
    best_state = None
    
    print("Starting training...")
    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, optimizer, device, scheduler)
        metrics, _, _, _ = evaluate_generation(model, dev_loader, device, tokenizer, evaluator, desc="Validating")
        print(f"Epoch {epoch+1} Loss: {loss:.4f} | Dev BLEU: {metrics['BLEU']:.2f}")
        
        if metrics['BLEU'] > best_bleu:
            best_bleu = metrics['BLEU']
            best_state = copy.deepcopy(model.state_dict())
            
    print("Evaluating Best Model on Test Set...")
    model.load_state_dict(best_state)
    metrics, _, _, _ = evaluate_generation(model, test_loader, device, tokenizer, evaluator, desc="Testing")
    
    results = [args.method, args.k, metrics['BLEU'], metrics['NIST'], metrics['METEOR'], metrics['ROUGE-L'], metrics['CIDEr']]
    headers = ["Method", "k", "BLEU", "NIST", "METEOR", "ROUGE-L", "CIDEr"]
    
    print("\n" + tabulate([results], headers=headers, tablefmt="grid"))
    
    out_file = os.path.join(args.output_dir, f"e2e_{args.method}_k{args.k}.csv")
    with open(out_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerow(results)

if __name__ == "__main__":
    main()