import os
import torch
import torch.nn as nn
import argparse
from pathlib import Path
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, CosineAnnealingLR
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter
from torch.amp import autocast, GradScaler

# Local imports
from med_vqa_model import MedVQAModel
from data_utils import get_data_loaders, get_combined_data_loaders, get_vqarad_data_loaders

# -----------------------------------------------------------------------------
# UTILS & METRICS
# -----------------------------------------------------------------------------
def compute_class_weights(train_loader, device):
    """
    Egitim setindeki sinif frekanslarindan ters agirliklar hesaplar.
    Nadiren gorulen acik-uclu cevap siniflarini yukselterek
    yes/no baskinligini azaltir ve open-ended accuracy'yi arttirir.
    """
    from collections import Counter
    counts = Counter()
    for item in train_loader.dataset.data_list:
        ans = item.get('answer', '')
        idx = train_loader.dataset.answer2idx.get(ans, 0)
        counts[idx] += 1
    num_classes = len(train_loader.dataset.answer2idx)
    total = sum(counts.values())
    weights = torch.ones(num_classes, device=device)
    for idx, cnt in counts.items():
        if cnt > 0:
            weights[idx] = total / (num_classes * cnt)
    # [0.1, 10.0] araligina sinirla — asiri agirlik etkisini engelle
    weights = torch.clamp(weights, 0.1, 10.0)
    return weights


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[SEED] Seed set to: {seed}")

def is_closed_ended(answer_text: str) -> bool:
    if not isinstance(answer_text, str): return False
    return answer_text.lower().strip() in ['yes', 'no', 'evet', 'hayır']

def calculate_metrics(predictions, true_answers, dataset, answer_types=None):
    correct_overall = 0
    correct_closed = 0
    total_closed = 0
    correct_open = 0
    total_open = 0

    for i, (pred_idx, true_idx) in enumerate(zip(predictions, true_answers)):
        if pred_idx == true_idx: correct_overall += 1
        ans_text = dataset.idx2answer[true_idx]
        atype = answer_types[i] if answer_types else 'UNKNOWN'
        if atype == 'CLOSED':
            closed = True
        elif atype == 'OPEN':
            closed = False
        else:
            closed = is_closed_ended(ans_text)
        if closed:
            total_closed += 1
            if pred_idx == true_idx: correct_closed += 1
        else:
            total_open += 1
            if pred_idx == true_idx: correct_open += 1

    return {
        'overall': correct_overall / len(predictions) if predictions else 0,
        'closed': correct_closed / total_closed if total_closed > 0 else 0,
        'open': correct_open / total_open if total_open > 0 else 0
    }

def plot_metrics(history, output_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.title('Loss')
    plt.legend(); plt.grid(True)
    plt.savefig(output_dir / 'loss.png'); plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['val_acc'], 'k-', linewidth=2, label='Overall Acc')
    plt.plot(epochs, history['closed_acc'], 'b--', label='Closed-Ended')
    plt.plot(epochs, history['open_acc'], 'r--', label='Open-Ended')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy')
    plt.legend(); plt.grid(True)
    plt.savefig(output_dir / 'accuracy_breakdown.png'); plt.close()

# -----------------------------------------------------------------------------
# TRAINING LOOP (STANDARD CONFIG)
# -----------------------------------------------------------------------------
def run_eval(model, data_loader, device):
    model.eval()
    preds, true, types = [], [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Eval', mininterval=60, ncols=60):
            img = batch['image'].to(device).float()
            inp = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            preds.extend(model(img, inp, mask).argmax(dim=1).cpu().tolist())
            true.extend(batch['answer'].tolist())
            if 'answer_type' in batch:
                types.extend(batch['answer_type'])
    return calculate_metrics(preds, true, data_loader.dataset,
                             answer_types=types if types else None)

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # Load Data
    if args.dataset == 'vqarad':
        train_loader, val_loader, test_loader = get_vqarad_data_loaders(
            batch_size=args.batch_size,
            min_answer_freq=args.min_answer_freq,
            unknown_token=args.unknown_token,
        )
    else:
        vocab_splits = ['train', 'val'] if args.vocab_from_train else None
        if args.use_combined:
            train_loader, val_loader, test_loader = get_combined_data_loaders(
                args.data_dir,
                args.batch_size,
                vocab_splits=vocab_splits,
                min_answer_freq=args.min_answer_freq,
                unknown_token=args.unknown_token,
            )
        else:
            train_loader, val_loader, test_loader = get_data_loaders(
                args.data_dir,
                args.batch_size,
                vocab_splits=vocab_splits,
                min_answer_freq=args.min_answer_freq,
                unknown_token=args.unknown_token,
            )
    num_classes = len(train_loader.dataset.answer2idx)

    # Initialize Model
    model = MedVQAModel(num_classes=num_classes).to(device)

    # Transfer learning: load encoder + fusion weights, skip classifier
    if args.pretrained_path:
        print(f"🔀 Transfer Learning from: {args.pretrained_path}")
        ckpt = torch.load(args.pretrained_path, map_location=device)
        src = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        transfer = {k: v for k, v in src.items() if not k.startswith('classifier.')}
        missing, unexpected = model.load_state_dict(transfer, strict=False)
        print(f"   ✓ Loaded {len(transfer)} layers (skipped classifier)")
        print(f"   Missing : {[k for k in missing if not k.startswith('classifier.')]}")

    # Loss — sinif agirlikli veya standart
    if args.use_class_weights:
        class_weights = compute_class_weights(train_loader, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1).to(device)
        print(f"📊 Class-weighted CrossEntropy aktif "
              f"(min={class_weights.min():.2f}, max={class_weights.max():.2f})")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
        print("🚀 Training with STANDARD CrossEntropy (Label Smoothing=0.1)")

    optimizer = AdamW([
        {'params': model.vit.parameters(),        'lr': 1e-5},
        {'params': model.bert.parameters(),       'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': 5e-4},
        {'params': model.fusion.parameters(),     'lr': 5e-4}
    ], weight_decay=0.01)

    # initial_lr'yi sakla: warmup dogrusal hesaplama icin baz deger
    for pg in optimizer.param_groups:
        pg['initial_lr'] = pg['lr']

    # Tek doğrusal cosine azalım — restart yok, kararlı yakınsama.
    # T_max = warmup sonrası kalan epoch sayısı.
    cosine_epochs = max(args.num_epochs - args.warmup_epochs, 1)
    scheduler = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=1e-7)
    warmup_epochs = args.warmup_epochs
    scaler = GradScaler('cuda')
    
    # Tracking
    best_val_acc = 0.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = {'train_loss': [], 'val_acc': [], 'open_acc': [], 'closed_acc': []}
    
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0

        # Linear Warmup: ilk warmup_epochs epoch boyunca LR'yi lineer artir
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['initial_lr'] * warmup_factor

        # Freezing logic
        if epoch == args.stage2_epoch:
            print("❄️ Freezing Encoders (Stage 2)...")
            for param in model.vit.parameters(): param.requires_grad = False
            for param in model.bert.parameters(): param.requires_grad = False
            
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}', mininterval=30, ncols=80)
        for i, batch in enumerate(pbar):
            img = batch['image'].to(device).float()
            inp = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            ans = batch['answer'].to(device)
            
            with autocast('cuda', enabled=True):
                out = model(img, inp, mask)
                loss = criterion(out, ans)
                loss = loss / args.gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            total_loss += loss.item() * args.gradient_accumulation_steps
            
            if (i + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        
        # Validation
        metrics = run_eval(model, val_loader, device)
        history['train_loss'].append(total_loss / len(train_loader))
        history['val_acc'].append(metrics['overall'])
        history['closed_acc'].append(metrics['closed'])
        history['open_acc'].append(metrics['open'])
        
        print(f"📊 Epoch {epoch+1}: Overall: {metrics['overall']:.4f} | Closed: {metrics['closed']:.4f} | Open: {metrics['open']:.4f}")
        
        if epoch >= warmup_epochs:
            scheduler.step()
        
        if metrics['overall'] > best_val_acc:
            best_val_acc = metrics['overall']
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': best_val_acc,
                'epoch': epoch,
                'num_classes': num_classes
            }, output_dir / 'model_best.pth')
            print("⭐ New Best Model Saved!")
            
    plot_metrics(history, output_dir)

    if args.eval_test:
        best_path = output_dir / 'model_best.pth'
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=device)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            model.load_state_dict(state_dict)
        test_metrics = run_eval(model, test_loader, device)
        print("\n" + "="*50)
        print("🏆 FINAL TEST RESULTS")
        print("="*50)
        print(f"   Overall Accuracy : {test_metrics['overall']*100:.2f}%")
        print(f"   Closed-Ended Acc : {test_metrics['closed']*100:.2f}%")
        print(f"   Open-Ended Acc   : {test_metrics['open']*100:.2f}%")
        print("="*50)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/datasets/amoghdumbre/slakemedvqa/Slake1.0')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--num_epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--stage2_epoch', type=int, default=15)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default='slake', choices=['slake', 'vqarad'],
                        help='Dataset to train and evaluate on (slake or vqarad)')
    parser.add_argument('--use_combined', action='store_true', default=False)
    parser.add_argument('--eval_test', action='store_true', default=False)
    parser.add_argument('--vocab_from_train', action='store_true', default=False)
    parser.add_argument('--min_answer_freq', type=int, default=1)
    parser.add_argument('--unknown_token', type=str, default=None)
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Checkpoint to load encoder+fusion weights from (classifier skipped)')
    parser.add_argument('--use_class_weights', action='store_true', default=False,
                        help='Sinif frekansindan ters agirlik hesapla (acik-uclu dengesizligi)')
    parser.add_argument('--warmup_epochs', type=int, default=3,
                        help='Linear LR warmup epoch sayisi')
    args = parser.parse_args()
    
    set_seed(args.seed)
    train(args)

if __name__ == '__main__':
    main()
