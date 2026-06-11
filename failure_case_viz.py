# Failure Case Visualization — Figure 7
# ========================================
# Finds incorrectly predicted samples on the SLAKE test set
# and visualizes them with RE-LIG saliency maps.
# includes brain scan and diverse failure examples.

import argparse, torch, numpy as np, gc, json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib import cm
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import torch.nn.functional as F
from transformers import AutoTokenizer
from captum.attr import LayerIntegratedGradients, NoiseTunnel

from med_vqa_model import MedVQAModel
from data_utils import SlakeDataset, build_slake_vocab, load_slake_data
from relig_config import RELIG_CONFIG

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--model_path',  type=str, default=None)
_parser.add_argument('--data_dir',    type=str, default=None)
_parser.add_argument('--num_cases',   type=int, default=6)
_parser.add_argument('--output_dir',  type=str, default='/kaggle/working/fig7_failure')
_known, _ = _parser.parse_known_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATH = _known.model_path or \
    "/kaggle/input/datasets/anonymous/slakemodelyeni/model_best.pth"
DATA_DIR   = Path(_known.data_dir) if _known.data_dir else \
    Path("/kaggle/input/datasets/amoghdumbre/slakemedvqa/Slake1.0")
OUT_DIR    = Path(_known.output_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)
NUM_CASES  = _known.num_cases

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- Veri ---
answer2idx = build_slake_vocab(DATA_DIR)
idx2answer = {v: k for k, v in answer2idx.items()}
test_data  = load_slake_data(DATA_DIR, 'test')
dataset    = SlakeDataset(test_data, answer2idx, image_size=448, is_train=False)
dataset.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# --- Model ---
ckpt = torch.load(MODEL_PATH, map_location=device)
sd   = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
if 'num_classes' in ckpt:
    num_classes = ckpt['num_classes']
else:
    cls_keys    = [k for k in sd if 'classifier' in k and 'weight' in k]
    num_classes = sd[cls_keys[-1]].shape[0]
model = MedVQAModel(num_classes=num_classes, image_size=448).to(device)
model.load_state_dict(sd, strict=False)
model.eval()
print(f"Model loaded — {num_classes} classes")

if hasattr(model.vit, 'embeddings'):
    target_layer = model.vit.embeddings
elif hasattr(model.vit, 'vision_model'):
    target_layer = model.vit.vision_model.embeddings
else:
    target_layer = model.vit


def relig_attr_map(model, img, txt, msk, pred):
    """Returns 14×14 RE-LIG attribution map (numpy)."""
    def fwd(image):
        bs = image.shape[0]
        return model(image, txt.expand(bs, -1), msk.expand(bs, -1))
    lig  = LayerIntegratedGradients(fwd, target_layer)
    nt   = NoiseTunnel(lig)
    attr = nt.attribute(
        img, baselines=torch.zeros_like(img), target=pred,
        n_steps=RELIG_CONFIG['n_steps'],
        nt_type=RELIG_CONFIG['nt_type'],
        nt_samples=RELIG_CONFIG['nt_samples'],
        stdevs=RELIG_CONFIG['stdev'],
        attribute_to_layer_input=False,
        internal_batch_size=1,
    )
    attr = attr.sum(dim=-1).squeeze()
    if attr.shape[0] == 197: attr = attr[1:]
    attr = attr.reshape(14, 14)
    # 99th-percentile normalization
    vmax = np.percentile(attr.abs().detach().cpu().numpy(), 99)
    attr = attr.clamp(-vmax, vmax)
    attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
    return attr.detach().cpu().numpy()


def load_original_image(sample_data, data_dir):
    """Loads original image from SLAKE data."""
    try:
        img_name = sample_data.get('img_name', sample_data.get('image_name', ''))
        img_path = data_dir / 'imgs' / img_name
        if not img_path.exists():
            # Recursive search in subdirectories
            matches = list(data_dir.rglob(img_name))
            if matches:
                img_path = matches[0]
            else:
                return None
        img = Image.open(img_path).convert('RGB')
        return img
    except Exception:
        return None


def overlay_heatmap(pil_img, attr_map, alpha=0.5):
    """Overlays saliency map on the original image."""
    img_arr = np.array(pil_img.resize((448, 448))) / 255.0
    # 14×14 → 448×448 upscale
    heatmap = F.interpolate(
        torch.tensor(attr_map).float().unsqueeze(0).unsqueeze(0),
        size=(448, 448), mode='bilinear', align_corners=False
    ).squeeze().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    colormap = cm.get_cmap('jet')
    colored  = colormap(heatmap)[:, :, :3]   # (448, 448, 3)
    blended  = (1 - alpha) * img_arr + alpha * colored
    blended  = np.clip(blended, 0, 1)
    return blended


# ---------------------------------------------------------------
# Collect failure cases — prioritize brain scan examples
# ---------------------------------------------------------------
print("\nSearching for failure cases...")

brain_failures   = []
other_failures   = []
brain_keywords   = ['brain', 'head', 'mri', 'mr', 'ct', 'cranial', 'cerebr', 'skull']
seen_combos      = set()   # deduplication key: (question, gt, pred)

for idx in tqdm(range(len(dataset))):
    try:
        item = dataset[idx]
        img  = item['image'].to(device).unsqueeze(0).float()
        txt  = item['input_ids'].to(device).unsqueeze(0)
        msk  = item['attention_mask'].to(device).unsqueeze(0)
        tgt  = item['answer'].item() if hasattr(item['answer'], 'item') else item['answer']

        with torch.no_grad():
            out  = model(img, txt, msk)
            pred = out.argmax(1).item()
            conf = torch.softmax(out, dim=1)[0, pred].item()

        if pred == tgt:
            del img, txt, msk
            continue

        raw       = test_data[idx]
        question  = raw.get('question', raw.get('Question', ''))
        gt_ans    = raw.get('answer', raw.get('Answer', idx2answer.get(tgt, str(tgt))))
        pred_ans  = idx2answer.get(pred, str(pred))
        q_type    = raw.get('answer_type',
                   raw.get('content_type',
                   'CLOSED' if str(raw.get('answer', raw.get('Answer', ''))).strip().lower()
                               in ('yes', 'no') else 'OPEN'))
        img_name  = raw.get('img_name', raw.get('image_name', ''))

        # Skip duplicate (question, gt, pred) combos
        combo = (question.strip().lower(), str(gt_ans).lower(), str(pred_ans).lower())
        if combo in seen_combos:
            del img, txt, msk
            continue
        seen_combos.add(combo)

        entry = {
            'dataset_idx': idx,
            'question':    question,
            'gt_answer':   gt_ans,
            'pred_answer': pred_ans,
            'confidence':  conf,
            'img_name':    img_name,
            'q_type':      q_type,
            'raw':         raw,
        }

        is_brain = any(kw in img_name.lower() or kw in question.lower()
                       for kw in brain_keywords)
        if is_brain:
            brain_failures.append(entry)
        else:
            other_failures.append(entry)

        del img, txt, msk

        # Search entire dataset for brain; stop early only when other bucket is full
        if len(other_failures) >= 20 and len(brain_failures) >= 3:
            break

    except Exception as e:
        continue

print(f"Brain scan failures: {len(brain_failures)}")
print(f"Other failures: {len(other_failures)}")

# Guarantee at least 1 brain scan example
selected = []
if brain_failures:
    selected.append(brain_failures[0])
    if len(brain_failures) >= 2:
        selected.append(brain_failures[1])

# Fill remaining slots from other_failures (diverse question types)
q_types_seen = set()
for entry in other_failures:
    if len(selected) >= NUM_CASES:
        break
    if entry['q_type'] not in q_types_seen or len(selected) < NUM_CASES:
        selected.append(entry)
        q_types_seen.add(entry['q_type'])

selected = selected[:NUM_CASES]
print(f"\n{len(selected)} cases selected for visualization.")

# ---------------------------------------------------------------
# Compute RE-LIG maps and visualize
# ---------------------------------------------------------------
n_cols = 3
n_rows = (len(selected) + n_cols - 1) // n_cols
fig = plt.figure(figsize=(n_cols * 5, n_rows * 7))

for i, entry in enumerate(selected):
    clear_gpu()
    idx  = entry['dataset_idx']
    item = dataset[idx]
    img  = item['image'].to(device).unsqueeze(0).float()
    txt  = item['input_ids'].to(device).unsqueeze(0)
    msk  = item['attention_mask'].to(device).unsqueeze(0)

    with torch.no_grad():
        pred = model(img, txt, msk).argmax(1).item()

    try:
        attr_map = relig_attr_map(model, img.requires_grad_(True), txt, msk, pred)
    except Exception as e:
        print(f"Attribution error [{idx}]: {e}")
        attr_map = np.zeros((14, 14))

    pil_img = load_original_image(entry['raw'], DATA_DIR)
    if pil_img is None:
        # Fallback: reconstruct from tensor
        img_np = item['image'].permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        pil_img = Image.fromarray((img_np * 255).astype(np.uint8))

    overlay = overlay_heatmap(pil_img, attr_map, alpha=0.45)

    ax = fig.add_subplot(n_rows, n_cols, i + 1)
    ax.imshow(overlay)
    ax.axis('off')

    # Truncate long questions
    q_text = entry['question']
    if len(q_text) > 60:
        q_text = q_text[:57] + '...'

    title = (f"Q: {q_text}\n"
             f"GT: {entry['gt_answer']}  |  Pred: {entry['pred_answer']}\n"
             f"[{entry['q_type']}]  conf={entry['confidence']:.2f}")
    ax.set_title(title, fontsize=8.5, loc='left',
                 color='darkred', fontweight='bold',
                 pad=4, wrap=True)

    del img, txt, msk
    clear_gpu()

plt.suptitle(
    'Figure 7: Qualitative Failure Analysis — RE-LIG Saliency Maps\n'
    'Red title = incorrect prediction  |  Heatmap shows attributed regions',
    fontsize=11, fontweight='bold', y=1.01)
plt.tight_layout(pad=1.5, h_pad=4.0, w_pad=1.5)
out_png = OUT_DIR / 'figure7_failure_cases.png'
fig.savefig(out_png, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out_png}")

# Save metadata
with open(OUT_DIR / 'failure_cases_meta.json', 'w') as f:
    json.dump([{k: v for k, v in e.items() if k != 'raw'}
               for e in selected], f, indent=2, ensure_ascii=False)
print(f"Saved: {OUT_DIR}/failure_cases_meta.json")
print("\nDone.")
