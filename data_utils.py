"""
Data Utilities for SLAKE Medical VQA Dataset
=============================================
Enhanced with:
- Strong data augmentation (RandAugment + Medical-specific)
- Test-Time Augmentation (TTA) support
- Answer normalization for improved open-ended accuracy
"""

import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
from pathlib import Path
import re
from typing import Dict, List, Any, Tuple
import numpy as np

# Import answer normalization
try:
    from answer_normalization import normalize_answer
    NORMALIZATION_ENABLED = True
    print("✅ Answer normalization: ENABLED")
except ImportError:
    NORMALIZATION_ENABLED = False
    print("⚠️ Answer normalization: DISABLED (answer_normalization.py not found)")
    def normalize_answer(x, **kwargs): return x.lower().strip() if isinstance(x, str) else x

# =============================================================================
# VQA-RAD PATHS (Kaggle: kaidegast/vqarad)
# =============================================================================
import pandas as pd

_VQA_RAD_BASE    = Path("/kaggle/input/datasets/kaidegast/vqarad")
VQA_RAD_IMG_DIR  = _VQA_RAD_BASE / "VQA_RAD Image Folder"
VQA_RAD_TRAIN_CSV = _VQA_RAD_BASE / "vqa_rad_train.csv"
VQA_RAD_VAL_CSV   = _VQA_RAD_BASE / "vqa_rad_valid.csv"
VQA_RAD_TEST_CSV  = _VQA_RAD_BASE / "vqa_rad_test.csv"

# VQA-RAD JSON fallback (shashankshekhar1205 dataset)
_VQA_RAD_JSON_BASE  = Path("/kaggle/input/datasets/shashankshekhar1205/vqa-rad-visual-question-answering-radiology")
VQA_RAD_JSON_IMG_DIR = _VQA_RAD_JSON_BASE / "VQA_RAD Image Folder"
VQA_RAD_JSON_PATH    = _VQA_RAD_JSON_BASE / "VQA_RAD Dataset Public.json"


# =============================================================================
# TEST-TIME AUGMENTATION (TTA)
# =============================================================================
class TTAWrapper:
    """
    Test-Time Augmentation wrapper for improved inference accuracy.
    
    Applies multiple augmentations and averages predictions.
    Typical improvement: +1-2% accuracy.
    """
    
    def __init__(self, model, device, num_augments=5):
        self.model = model
        self.device = device
        self.num_augments = num_augments
        
        # TTA transforms (geometric + slight intensity)
        self.tta_transforms = [
            transforms.Compose([]),
            transforms.Compose([transforms.ColorJitter(brightness=0.05, contrast=0.05)]),
            transforms.Compose([transforms.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.98, 1.02))]),
            transforms.Compose([transforms.RandomRotation(degrees=(-2, 2))]),
            transforms.Compose([transforms.ColorJitter(brightness=0.05)]),
        ]
    
    def predict(self, image: torch.Tensor, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Predict with TTA - averages logits from multiple augmented views.
        
        Args:
            image: Input image [1, 3, H, W]
            input_ids: Text tokens [1, seq_len]
            attention_mask: Attention mask [1, seq_len]
            
        Returns:
            Averaged logits [1, num_classes]
        """
        self.model.eval()
        all_logits = []
        
        # Convert to PIL for augmentation
        img_pil = transforms.ToPILImage()(image.squeeze(0).cpu())
        
        with torch.no_grad():
            for i, aug in enumerate(self.tta_transforms[:self.num_augments]):
                # Apply augmentation
                aug_img = aug(img_pil)
                
                # Convert back to tensor
                aug_tensor = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
                ])(aug_img).unsqueeze(0).to(self.device)
                
                # Get prediction
                logits = self.model(aug_tensor, input_ids, attention_mask)
                all_logits.append(logits)
        
        # Average logits (soft voting)
        avg_logits = torch.stack(all_logits).mean(dim=0)
        return avg_logits


def apply_tta(model, image, input_ids, attention_mask, device, num_augments=5):
    """
    Convenience function for TTA prediction.
    
    Args:
        model: The MedVQA model
        image: Input image tensor [1, 3, H, W]
        input_ids: Text tokens [1, seq_len]
        attention_mask: Attention mask [1, seq_len]
        device: torch device
        num_augments: Number of augmentations (1-5)
        
    Returns:
        Predicted class index
    """
    tta = TTAWrapper(model, device, num_augments)
    logits = tta.predict(image, input_ids, attention_mask)
    return logits.argmax(dim=1).item()

# =============================================================================
# SLAKE DATA LOADING UTILITIES
# =============================================================================

def _is_english(text: str) -> bool:
    """Check if text is English (no Chinese characters)."""
    if not isinstance(text, str): return False
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    return not bool(chinese_pattern.search(text))

def load_slake_data(data_dir: Path, split: str, use_normalization: bool = True) -> List[Dict]:
    """
    Load SLAKE dataset with optional answer normalization.
    
    Args:
        data_dir: Path to SLAKE dataset
        split: 'train', 'val', or 'test'
        use_normalization: Whether to apply answer normalization
        
    Returns:
        List of data items with normalized answers
    """
    data = []
    file_names = {'train': 'train.json', 'val': 'validate.json', 'test': 'test.json'}
    json_path = data_dir / file_names.get(split, 'train.json')
    
    if not json_path.exists():
        print(f"⚠️ SLAKE file not found: {json_path}")
        return []
        
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    img_dir = data_dir / "imgs"
    
    for item in raw_data:
        if not _is_english(item['question']) or not _is_english(item['answer']):
            continue
        
        # Apply answer normalization for consistency
        raw_answer = str(item['answer']).lower().strip()
        if use_normalization and NORMALIZATION_ENABLED:
            normalized_answer = normalize_answer(raw_answer)
        else:
            normalized_answer = raw_answer
            
        data.append({
            'img_path': str(img_dir / item['img_name']),
            'question': item['question'],
            'answer': normalized_answer,
            'answer_raw': raw_answer,  # Keep original for debugging
            'source': 'slake'
        })
    
    print(f"✅ SLAKE ({split}) loaded: {len(data)} samples")
    return data


def _load_vqarad_json_all(use_normalization: bool = True) -> List[Dict]:
    """Load all VQA-RAD samples from JSON format (shashankshekhar1205 dataset)."""
    if not VQA_RAD_JSON_PATH.exists():
        return []
    try:
        with open(VQA_RAD_JSON_PATH, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        data = []
        for item in raw:
            img_name = item.get('image_name', '')
            if not img_name:
                continue
            if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_name += '.jpg'
            question    = item.get('question', '').strip()
            raw_answer  = str(item.get('answer', '')).lower().strip()
            answer_type = item.get('answer_type', 'OPEN').upper()
            if not question or not raw_answer:
                continue
            answer = normalize_answer(raw_answer) if (use_normalization and NORMALIZATION_ENABLED) else raw_answer
            data.append({
                'img_path':    str(VQA_RAD_JSON_IMG_DIR / img_name),
                'question':    question,
                'answer':      answer,
                'answer_raw':  raw_answer,
                'answer_type': answer_type,
                'source':      'vqa-rad',
            })
        print(f"✅ VQA-RAD JSON loaded: {len(data)} total samples")
        return data
    except Exception as e:
        print(f"⚠️ VQA-RAD JSON loading failed: {e}")
        return []


def load_vqarad_split(split: str, use_normalization: bool = True) -> List[Dict]:
    """Load one split of VQA-RAD. Tries CSV first, falls back to JSON with 70/15/15 split."""
    csv_map = {
        'train': VQA_RAD_TRAIN_CSV,
        'val':   VQA_RAD_VAL_CSV,
        'test':  VQA_RAD_TEST_CSV,
    }
    csv_path = csv_map.get(split)
    if csv_path is not None and csv_path.exists():
        df = pd.read_csv(csv_path)
        data = []
        for _, row in df.iterrows():
            img_name    = str(row['img_id']).strip()
            question    = str(row['question']).strip()
            raw_answer  = str(row['answer']).lower().strip()
            answer_type = str(row.get('answer_type', 'OPEN')).upper()
            if not question or not raw_answer:
                continue
            answer = normalize_answer(raw_answer) if (use_normalization and NORMALIZATION_ENABLED) else raw_answer
            data.append({
                'img_path':    str(VQA_RAD_IMG_DIR / img_name),
                'question':    question,
                'answer':      answer,
                'answer_raw':  raw_answer,
                'answer_type': answer_type,
                'source':      'vqa-rad',
            })
        print(f"✅ VQA-RAD CSV ({split}) loaded: {len(data)} samples")
        return data

    # Fallback: JSON with deterministic 70/15/15 split
    print(f"⚠️ VQA-RAD CSV not found at {csv_path}, trying JSON fallback...")
    all_data = _load_vqarad_json_all(use_normalization)
    if not all_data:
        print(f"⚠️ VQA-RAD JSON also not found. Returning empty list for split='{split}'.")
        return []
    import random as _rnd
    _rng = _rnd.Random(42)
    shuffled = list(all_data)
    _rng.shuffle(shuffled)
    n       = len(shuffled)
    n_test  = max(1, int(n * 0.15))
    n_val   = max(1, int(n * 0.15))
    splits  = {
        'test':  shuffled[:n_test],
        'val':   shuffled[n_test:n_test + n_val],
        'train': shuffled[n_test + n_val:],
    }
    result = splits.get(split, [])
    print(f"✅ VQA-RAD JSON ({split}) split: {len(result)} samples")
    return result


def build_vqarad_vocab(
    use_normalization: bool = True,
    min_answer_freq: int = 1,
    unknown_token: str = None
) -> Dict[str, int]:
    """Build answer vocabulary from all VQA-RAD splits."""
    answer_counts: Dict[str, int] = {}
    for split in ['train', 'val', 'test']:
        for item in load_vqarad_split(split, use_normalization):
            ans = item['answer']
            answer_counts[ans] = answer_counts.get(ans, 0) + 1
    answers = [a for a, cnt in answer_counts.items() if cnt >= min_answer_freq]
    sorted_answers = sorted(answers)
    if 'yes' in sorted_answers:
        sorted_answers.remove('yes')
        sorted_answers.insert(0, 'yes')
    if 'no' in sorted_answers:
        sorted_answers.remove('no')
        sorted_answers.insert(1, 'no')
    if unknown_token and unknown_token not in sorted_answers:
        sorted_answers.append(unknown_token)
    vocab = {ans: idx for idx, ans in enumerate(sorted_answers)}
    print(f"📊 VQA-RAD vocabulary: {len(vocab)} unique answers")
    return vocab


def build_slake_vocab(
    data_dir: Path, 
    use_normalization: bool = True, 
    vocab_splits: List[str] = None,
    min_answer_freq: int = 1,
    unknown_token: str = None
) -> Dict[str, int]:
    """
    Build answer vocabulary from all splits.
    
    Uses normalized answers to reduce vocabulary fragmentation.
    Ensures 'yes' and 'no' are at indices 0 and 1 for consistency.
    """
    if vocab_splits is None:
        vocab_splits = ['train', 'val', 'test']
    if min_answer_freq > 1 and unknown_token is None:
        unknown_token = 'unknown'
    answer_counts = {}
    for split in vocab_splits:
        items = load_slake_data(data_dir, split, use_normalization=use_normalization)
        for item in items:
            ans = item['answer']
            answer_counts[ans] = answer_counts.get(ans, 0) + 1
    answers = [ans for ans, cnt in answer_counts.items() if cnt >= min_answer_freq]
    sorted_answers = sorted(answers)
    
    # Ensure Yes/No are at indices 0 and 1 for closed-ended detection
    if 'yes' in sorted_answers: 
        sorted_answers.remove('yes')
        sorted_answers.insert(0, 'yes')
    if 'no' in sorted_answers: 
        sorted_answers.remove('no')
        sorted_answers.insert(1, 'no')
    
    if unknown_token is not None and unknown_token not in sorted_answers:
        sorted_answers.append(unknown_token)
    vocab = {ans: idx for idx, ans in enumerate(sorted_answers)}
    
    print(f"📊 Vocabulary built: {len(vocab)} unique answers")
    print(f"   Index 0: '{sorted_answers[0] if sorted_answers else 'N/A'}'")
    print(f"   Index 1: '{sorted_answers[1] if len(sorted_answers) > 1 else 'N/A'}'")
    
    return vocab

# =============================================================================
# SLAKE DATASET CLASS
# =============================================================================

class SlakeDataset(Dataset):
    """
    SLAKE Medical VQA Dataset.
    
    Features:
    - Enhanced data augmentation for training (SOTA-level)
    - Answer normalization for reduced vocabulary fragmentation
    - BERT tokenization for questions
    """
    
    def __init__(self, data_list, answer2idx, image_size=448, is_train=False,
                 unknown_token=None, lighter_aug=False):
        if len(data_list) == 0:
            raise RuntimeError(
                "Dataset is empty — check that the dataset path is correct and "
                "the dataset is attached to this Kaggle notebook."
            )
        self.data_list = data_list
        self.answer2idx = answer2idx
        self.image_size = image_size
        self.idx2answer = {v: k for k, v in answer2idx.items()}
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                'bert-base-uncased', local_files_only=True
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.is_train = is_train
        self.unknown_token = unknown_token

        if is_train:
            if lighter_aug:
                # VQA-RAD icin hafif augmentation (klinik goruntulerde anatomik hassasiyet)
                self.transform = transforms.Compose([
                    transforms.Resize((image_size + 16, image_size + 16)),
                    transforms.RandomCrop((image_size, image_size)),
                    transforms.RandAugment(num_ops=2, magnitude=7),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            else:
                # SLAKE icin guclu augmentation
                # NOT: RandomHorizontalFlip kaldirildi — tibbi goruntulerde sol/sag
                # yon bilgisi klinik anlam tasir (akciger, kalp konumu vb.).
                self.transform = transforms.Compose([
                    transforms.Resize((image_size + 32, image_size + 32)),
                    transforms.RandomCrop((image_size, image_size)),
                    transforms.RandomRotation(degrees=5),
                    transforms.RandAugment(num_ops=2, magnitude=10),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
                ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self): 
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # Load and transform image
        try:
            image = Image.open(item['img_path']).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"⚠️ Error loading image {item['img_path']}: {e}")
            image = torch.zeros((3, self.image_size, self.image_size))
        
        # Tokenize question
        tokens = self.tokenizer(
            item['question'], 
            padding='max_length', 
            truncation=True, 
            max_length=32, 
            return_tensors='pt'
        )
        
        # Get answer index
        unknown_idx = None if self.unknown_token is None else self.answer2idx.get(self.unknown_token)
        if unknown_idx is None:
            ans_idx = self.answer2idx.get(item['answer'], 0)
        else:
            ans_idx = self.answer2idx.get(item['answer'], unknown_idx)
        
        return {
            'image': image,
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'answer': torch.tensor(ans_idx, dtype=torch.long),
            'answer_text': item['answer'],
            'question_text': item['question'],
            'answer_type': item.get('answer_type', 'UNKNOWN'),
        }

# =============================================================================
# DATA LOADER FACTORY
# =============================================================================

def get_data_loaders(
    data_dir: str, 
    batch_size=16,
    num_workers=2,
    image_size=448,
    use_normalization=True,
    use_vqarad=False,
    vocab_splits: List[str] = None,
    min_answer_freq: int = 1,
    unknown_token: str = None
):
    """
    Create data loaders for SLAKE dataset (+ optional VQA-RAD for training).

    Args:
        data_dir: Path to SLAKE dataset
        batch_size: Batch size for training/validation
        num_workers: Number of data loading workers
        image_size: Image size for model
        use_normalization: Whether to apply answer normalization
        use_vqarad: If True, appends VQA-RAD samples to training data only

    Returns:
        train_loader, val_loader, test_loader
    """
    slake_dir = Path(data_dir)

    print("\n" + "="*50)
    print("📂 Loading SLAKE Dataset" + (" + VQA-RAD" if use_vqarad else ""))
    print("="*50)

    # Build vocabulary
    answer2idx = build_slake_vocab(
        slake_dir,
        use_normalization=use_normalization,
        vocab_splits=vocab_splits,
        min_answer_freq=min_answer_freq,
        unknown_token=unknown_token
    )

    # Load data splits
    train_data = load_slake_data(slake_dir, 'train', use_normalization=use_normalization)
    val_data = load_slake_data(slake_dir, 'val', use_normalization=use_normalization)
    test_data = load_slake_data(slake_dir, 'test', use_normalization=use_normalization)

    # Optionally extend training data with VQA-RAD
    if use_vqarad:
        vqarad_data = load_vqarad_data(use_normalization=use_normalization)
        train_data = train_data + vqarad_data
        print(f"✅ VQA-RAD appended — total training samples: {len(train_data)}")

    # Create datasets
    train_dataset = SlakeDataset(train_data, answer2idx, image_size, is_train=True, unknown_token=unknown_token)
    val_dataset = SlakeDataset(val_data, answer2idx, image_size, is_train=False, unknown_token=unknown_token)
    test_dataset = SlakeDataset(test_data, answer2idx, image_size, is_train=False, unknown_token=unknown_token)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1,  # Batch size 1 for test to ensure accurate metrics
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n📊 Data splits:")
    print(f"   Train: {len(train_data)} samples")
    print(f"   Val:   {len(val_data)} samples")
    print(f"   Test:  {len(test_data)} samples")
    print(f"   Vocab: {len(answer2idx)} classes")
    print("="*50 + "\n")
    
    return train_loader, val_loader, test_loader


def get_combined_data_loaders(
    data_dir: str, 
    batch_size=16, 
    num_workers=2,
    image_size=448,
    use_normalization=True,
    use_vqarad=False,
    val_split=0.1,
    vocab_splits: List[str] = None,
    min_answer_freq: int = 1,
    unknown_token: str = None
):
    """
    Create data loaders with train+val COMBINED for training.

    This approach is common in Med-VQA papers to maximize training data.
    A small portion of training data is held out for validation (early stopping).
    Test set is used ONLY for final evaluation.

    Args:
        data_dir: Path to SLAKE dataset
        batch_size: Batch size
        num_workers: Number of workers
        image_size: Image size
        use_normalization: Answer normalization
        use_vqarad: If True, appends VQA-RAD samples to combined training data
        val_split: Fraction of combined train+val to use for validation

    Returns:
        train_loader, val_loader, test_loader
    """
    slake_dir = Path(data_dir)

    print("\n" + "="*50)
    print("📂 Loading SLAKE Dataset (COMBINED MODE)" + (" + VQA-RAD" if use_vqarad else ""))
    print("="*50)

    # Build vocabulary from all splits
    answer2idx = build_slake_vocab(
        slake_dir,
        use_normalization=use_normalization,
        vocab_splits=vocab_splits,
        min_answer_freq=min_answer_freq,
        unknown_token=unknown_token
    )

    # Load and COMBINE train + val
    train_data = load_slake_data(slake_dir, 'train', use_normalization=use_normalization)
    val_data = load_slake_data(slake_dir, 'val', use_normalization=use_normalization)
    test_data = load_slake_data(slake_dir, 'test', use_normalization=use_normalization)

    # Optionally extend with VQA-RAD before combining
    if use_vqarad:
        vqarad_data = load_vqarad_data(use_normalization=use_normalization)
        train_data = train_data + vqarad_data
        print(f"✅ VQA-RAD appended to train: {len(vqarad_data)} samples")

    # Combine train + val
    combined_data = train_data + val_data

    # Shuffle and split for internal validation
    import random
    random.shuffle(combined_data)

    val_size = int(len(combined_data) * val_split)
    internal_val_data = combined_data[:val_size]
    internal_train_data = combined_data[val_size:]

    print(f"\n🔀 COMBINED MODE:")
    print(f"   Original Train: {len(train_data)}")
    print(f"   Original Val:   {len(val_data)}")
    print(f"   Combined Total: {len(combined_data)}")
    print(f"   → New Train: {len(internal_train_data)} ({100-val_split*100:.0f}%)")
    print(f"   → New Val:   {len(internal_val_data)} ({val_split*100:.0f}%)")
    print(f"   → Test:      {len(test_data)} (unchanged)")
    
    # Create datasets
    train_dataset = SlakeDataset(internal_train_data, answer2idx, image_size, is_train=True, unknown_token=unknown_token)
    val_dataset = SlakeDataset(internal_val_data, answer2idx, image_size, is_train=False, unknown_token=unknown_token)
    test_dataset = SlakeDataset(test_data, answer2idx, image_size, is_train=False, unknown_token=unknown_token)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    
    print(f"   Vocab: {len(answer2idx)} classes")
    print("="*50 + "\n")
    
    return train_loader, val_loader, test_loader


# =============================================================================
# VQA-RAD DATA LOADER FACTORY
# =============================================================================

def get_vqarad_data_loaders(
    batch_size: int = 16,
    num_workers: int = 2,
    image_size: int = 448,
    use_normalization: bool = True,
    min_answer_freq: int = 1,
    unknown_token: str = None,
) -> tuple:
    """
    Create train/val/test data loaders for VQA-RAD dataset.

    Trains and evaluates entirely on VQA-RAD — independent of SLAKE.
    Returns separate metrics comparable to SLAKE results in the paper.
    """
    print("\n" + "="*50)
    print("📂 Loading VQA-RAD Dataset")
    print("="*50)

    answer2idx = build_vqarad_vocab(
        use_normalization=use_normalization,
        min_answer_freq=min_answer_freq,
        unknown_token=unknown_token,
    )

    train_data = load_vqarad_split('train', use_normalization)
    val_data   = load_vqarad_split('val',   use_normalization)
    test_data  = load_vqarad_split('test',  use_normalization)

    train_dataset = SlakeDataset(train_data, answer2idx, image_size, is_train=True,
                                 unknown_token=unknown_token, lighter_aug=True)
    val_dataset   = SlakeDataset(val_data,   answer2idx, image_size, is_train=False,
                                 unknown_token=unknown_token)
    test_dataset  = SlakeDataset(test_data,  answer2idx, image_size, is_train=False,
                                 unknown_token=unknown_token)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=1,          shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"\n📊 VQA-RAD splits:")
    print(f"   Train: {len(train_data)} samples")
    print(f"   Val:   {len(val_data)} samples")
    print(f"   Test:  {len(test_data)} samples")
    print(f"   Vocab: {len(answer2idx)} classes")
    print("="*50 + "\n")

    return train_loader, val_loader, test_loader


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_answer_distribution(data_loader):
    """Get distribution of answers in a data loader."""
    from collections import Counter
    
    answer_counts = Counter()
    for item in data_loader.dataset.data_list:
        answer_counts[item['answer']] += 1
    
    return answer_counts.most_common()


if __name__ == '__main__':
    # Test the data loading
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/datasets/amoghdumbre/slakemedvqa/Slake1.0')
    args = parser.parse_args()
    
    train_loader, val_loader, test_loader = get_data_loaders(args.data_dir)
    
    print("\nSample batch:")
    batch = next(iter(train_loader))
    print(f"  Image shape: {batch['image'].shape}")
    print(f"  Input IDs shape: {batch['input_ids'].shape}")
    print(f"  Answer: {batch['answer_text'][0]}")
    print(f"  Question: {batch['question_text'][0]}")
