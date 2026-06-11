import os
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F

# Local imports
from med_vqa_model import MedVQAModel
from data_utils import get_data_loaders, get_vqarad_data_loaders
from answer_normalization import normalize_answer

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

def report_vocab_coverage(dataset, unknown_token=None, top_k=10):
    total = len(dataset.data_list)
    missing_counts = {}
    for item in dataset.data_list:
        ans = item.get('answer')
        if ans not in dataset.answer2idx:
            missing_counts[ans] = missing_counts.get(ans, 0) + 1
    missing = sum(missing_counts.values())
    print("\n" + "="*50)
    print("📌 Vocabulary Coverage Report")
    print("="*50)
    print(f"Total samples: {total}")
    print(f"Missing answers: {missing} ({(missing/total*100) if total else 0:.2f}%)")
    if unknown_token is not None:
        print(f"Unknown token: {unknown_token}")
    if missing_counts:
        top = sorted(missing_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
        print("Top missing answers:")
        for ans, cnt in top:
            print(f"  {ans}: {cnt}")

def test_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    print(f"📦 Loading Checkpoint: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # Determine num_classes from checkpoint or data
    if 'num_classes' in checkpoint:
        num_classes = checkpoint['num_classes']
    else:
        # Fallback detection
        keys = list(state_dict.keys())
        last_key = [k for k in keys if 'classifier' in k and 'weight' in k][-1]
        num_classes = state_dict[last_key].shape[0]
        
    print(f"ℹ️ Detected Classes: {num_classes}")
    
    # Load Model
    model = MedVQAModel(num_classes=num_classes, image_size=448).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Load Data
    print(f"📂 Loading Data ({args.dataset.upper()})...")
    if args.dataset == 'vqarad':
        train_loader, val_loader, test_loader = get_vqarad_data_loaders(
            batch_size=args.batch_size,
            image_size=448,
            min_answer_freq=args.min_answer_freq,
            unknown_token=args.unknown_token,
        )
    else:
        vocab_splits = ['train', 'val'] if args.vocab_from_train else None
        train_loader, val_loader, test_loader = get_data_loaders(
            args.data_dir,
            args.batch_size,
            image_size=448,
            vocab_splits=vocab_splits,
            min_answer_freq=args.min_answer_freq,
            unknown_token=args.unknown_token,
        )
    split = args.split.lower()
    if split == 'train':
        eval_loader = train_loader
    elif split == 'val':
        eval_loader = val_loader
    else:
        eval_loader = test_loader
    
    print(f"\n🚀 Evaluation Started — dataset={args.dataset.upper()}, split={split}...")
    if args.report_vocab_coverage:
        report_vocab_coverage(eval_loader.dataset, unknown_token=args.unknown_token)
    
    all_preds = []
    all_true = []
    all_types = []

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc='Testing'):
            img = batch['image'].to(device).float()
            inp = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            ans = batch['answer']

            out = model(img, inp, mask)
            preds = out.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_true.extend(ans.tolist())
            if 'answer_type' in batch:
                all_types.extend(batch['answer_type'])

    metrics = calculate_metrics(all_preds, all_true, eval_loader.dataset,
                                answer_types=all_types if all_types else None)
    
    print("\n" + "="*50)
    print(f"🏆 FINAL TEST RESULTS ({args.dataset.upper()})")
    print("="*50)
    print(f"   Overall Accuracy : {metrics['overall']*100:.2f}%")
    print(f"   Closed-Ended Acc : {metrics['closed']*100:.2f}%")
    print(f"   Open-Ended Acc   : {metrics['open']*100:.2f}%")
    print("="*50)

    if args.save_preds:
        import json
        preds_data = {
            'dataset': args.dataset,
            'split': args.split,
            'model_path': args.model_path,
            'predictions': all_preds,
            'true_labels': all_true,
            'answer_types': all_types,
            'metrics': {
                'overall': metrics['overall'],
                'closed': metrics['closed'],
                'open': metrics['open'],
            }
        }
        with open(args.preds_output, 'w') as f:
            json.dump(preds_data, f, indent=2)
        print(f"💾 Tahminler kaydedildi: {args.preds_output}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='slake', choices=['slake', 'vqarad'])
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/datasets/amoghdumbre/slakemedvqa/Slake1.0')
    parser.add_argument('--model_path', type=str, required=True, help="Path to .pth file")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--report_vocab_coverage', action='store_true', default=False)
    parser.add_argument('--vocab_from_train', action='store_true', default=False)
    parser.add_argument('--min_answer_freq', type=int, default=1)
    parser.add_argument('--unknown_token', type=str, default=None)
    parser.add_argument('--save_preds', action='store_true', default=False,
                        help='Tahminleri JSON olarak kaydet (istatistik testleri icin)')
    parser.add_argument('--preds_output', type=str, default='predictions.json',
                        help='Tahmin JSON kayit yolu')
    args = parser.parse_args()
    
    test_model(args)

if __name__ == '__main__':
    main()
