import argparse
import os
import torch
import time
import csv
from tabulate import tabulate
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

from src.datasets_glue import GLUEDataset
from src.models_bert import BertLayerSelection, BertBlockwiseLayerSelection, BertRandomLayerSelection, BERTFinetune
from src.engine_glue import train_epoch, eval_model
from src.utils import set_seed, count_parameters

def main():
    parser = argparse.ArgumentParser(description="Training-Free Layer Selection for BERT on GLUE")
    parser.add_argument('--dataset', type=str, required=True, help="GLUE task name (e.g., sst2, mrpc)")
    parser.add_argument('--data_dir', type=str, default='./data/glue_data', help="Path to GLUE data")
    parser.add_argument('--method', type=str, choices=['highest', 'lowest', 'blockwise_highest', 'blockwise_lowest', 'random', 'full_ft'], default='highest')
    parser.add_argument('--k', type=int, default=3, help="Number of layers to fine-tune")
    parser.add_argument('--B', type=int, default=3, help="Number of blocks (for blockwise methods)")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./results')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Running {args.method} on {args.dataset} (k={args.k})")

    # Dataset configs (N_CLASSES, MAX_LEN)
    task_configs = {
        'rte': (2, 256), 'mrpc': (2, 256), 'cola': (2, 256), 'qnli': (2, 256),
        'sst2': (2, 256), 'sst5': (5, 256), 'mr': (2, 256), 'cr': (2, 256),
        'mpqa': (2, 128), 'subj': (2, 256), 'trec': (6, 128), 
        'mnli': (3, 256), 'mnli-mm': (3, 256), 'snli': (3, 256), 'qqp': (2, 256)
    }

    if args.dataset not in task_configs:
        raise ValueError(f"Task {args.dataset} not supported")

    n_classes, max_len = task_configs[args.dataset]
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Path handling
    if args.dataset == 'mnli':
        train_path = os.path.join(args.data_dir, 'MNLI/train.tsv')
        dev_path = os.path.join(args.data_dir, 'MNLI/dev_matched.tsv')
    elif args.dataset == 'mnli-mm':
        train_path = os.path.join(args.data_dir, 'MNLI/train.tsv')
        dev_path = os.path.join(args.data_dir, 'MNLI/dev_mismatched.tsv')
    else:
        # Standard folder names usually match task name uppercase or specific cases
        # Adjust mapping as needed based on extraction
        folder_map = {'sst2': 'SST-2', 'cola': 'CoLA', 'qnli': 'QNLI', 'rte': 'RTE', 'mrpc': 'MRPC', 'qqp': 'QQP', 'snli': 'SNLI'}
        folder_name = folder_map.get(args.dataset, args.dataset)
        train_path = os.path.join(args.data_dir, folder_name, 'train.tsv' if args.dataset in folder_map else 'train.csv')
        dev_path = os.path.join(args.data_dir, folder_name, 'dev.tsv' if args.dataset in folder_map else 'test.csv')

    train_dataset = GLUEDataset(train_path, args.dataset, tokenizer, max_len)
    dev_dataset = GLUEDataset(dev_path, args.dataset, tokenizer, max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize Model
    if args.method == 'full_ft':
        model = BERTFinetune(n_classes)
    elif args.method == 'random':
        model = BertRandomLayerSelection(n_classes, args.k)
    elif args.method in ['highest', 'lowest']:
        model = BertLayerSelection(n_classes, k=args.k, finetune_method=args.method)
    elif 'blockwise' in args.method:
        model = BertBlockwiseLayerSelection(n_classes, k=args.k, B=args.B, finetune_method=args.method)
    
    model = model.to(device)
    
    sel_time = 0
    if hasattr(model, 'select_layers'):
        model.select_layers(train_loader)
        sel_time = model.layer_selection_time
        print(f"Selected Layers: {model.selected_indices}")
        model.finetune_selected_layers()
        
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*args.epochs)
    
    total_params, trainable_params = count_parameters(model)
    print(f"Total Params: {total_params:,} | Trainable: {trainable_params:,}")

    best_acc = 0
    best_f1 = 0
    start_ft_time = time.time()
    
    for epoch in range(args.epochs):
        train_acc, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, scheduler, len(train_dataset))
        test_acc, test_loss, prec, rec, f1 = eval_model(model, dev_loader, loss_fn, device, len(dev_dataset))
        
        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f} | Dev Acc {test_acc:.4f} | F1 {f1:.4f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_f1 = f1

    ft_time = time.time() - start_ft_time
    
    results = [args.dataset, args.method, args.k, f"{sel_time:.2f}s", f"{ft_time:.2f}s", f"{best_acc*100:.2f}", f"{best_f1*100:.2f}"]
    headers = ["Task", "Method", "k", "Sel Time", "FT Time", "Acc", "F1"]
    
    print("\n" + tabulate([results], headers=headers, tablefmt="grid"))
    
    # Save CSV
    out_file = os.path.join(args.output_dir, f"{args.dataset}_{args.method}_k{args.k}.csv")
    with open(out_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerow(results)

if __name__ == "__main__":
    main()