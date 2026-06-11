# Deletion & Insertion AUC Curves — Figure 5
# =============================================
# RE-LIG attribution ile Deletion ve Insertion eğrileri
# Protokol: sıfır maskeleme, en önemli patch'ler önce
# SLAKE test seti, N=50 doğru sınıflandırılmış örnek

import argparse, torch, numpy as np, gc, json
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer
from captum.attr import LayerIntegratedGradients, NoiseTunnel

from med_vqa_model import MedVQAModel
from data_utils import SlakeDataset, build_slake_vocab, load_slake_data
from relig_config import RELIG_CONFIG

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--model_path',  type=str, default=None)
_parser.add_argument('--data_dir',    type=str, default=None)
_parser.add_argument('--num_samples', type=int, default=50)
_parser.add_argument('--output_dir',  type=str, default='/kaggle/working/fig5_del_ins')
_known, _ = _parser.parse_known_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATH = _known.model_path or \
    "/kaggle/input/datasets/anonymous/slakemodelyeni/model_best.pth"
DATA_DIR   = Path(_known.data_dir) if _known.data_dir else \
    Path("/kaggle/input/datasets/amoghdumbre/slakemedvqa/Slake1.0")
OUT_DIR    = Path(_known.output_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

STEPS          = 14          # 14 adım → her adım 14 patch (%7.1)
N_PATCHES      = 196
PER_STEP       = N_PATCHES // STEPS

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- Veri ---
answer2idx = build_slake_vocab(DATA_DIR)
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
print(f"Model yüklendi — {num_classes} sınıf")

if hasattr(model.vit, 'embeddings'):
    target_layer = model.vit.embeddings
elif hasattr(model.vit, 'vision_model'):
    target_layer = model.vit.vision_model.embeddings
else:
    target_layer = model.vit


def relig_attr(model, img, txt, msk, pred):
    """RE-LIG attribution — 196 patch skoru döndürür."""
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
    return attr.detach().cpu().flatten()   # (196,)


# --- Ana döngü ---
# Tam tekrarlanabilirlik: np (ornek secimi) + torch/cuda (NoiseTunnel gurultusu)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
candidate_idx = np.random.choice(len(dataset), min(300, len(dataset)), replace=False)

del_curves, ins_curves = [], []
target_n = _known.num_samples

print(f"\n{target_n} doğru örnek toplanıyor...")
for idx in tqdm(candidate_idx):
    if len(del_curves) >= target_n:
        break
    try:
        clear_gpu()
        item = dataset[int(idx)]
        img  = item['image'].to(device).unsqueeze(0).float()
        txt  = item['input_ids'].to(device).unsqueeze(0)
        msk  = item['attention_mask'].to(device).unsqueeze(0)
        tgt  = item['answer'].item() if hasattr(item['answer'], 'item') else item['answer']

        with torch.no_grad():
            out  = model(img, txt, msk)
            pred = out.argmax(1).item()
            conf = torch.softmax(out, dim=1)[0, pred].item()
        if pred != tgt:
            continue

        attr       = relig_attr(model, img.requires_grad_(True), txt, msk, pred)
        ranked     = torch.argsort(attr, descending=True)   # en önemli patch önce

        # ---- DELETION ----
        del_probs = [conf]
        img_d = img.clone()
        for step in range(STEPS):
            for pi in ranked[step*PER_STEP:(step+1)*PER_STEP]:
                r = pi.item() // 14; c = pi.item() % 14
                img_d[:, :, r*32:(r+1)*32, c*32:(c+1)*32] = 0
            with torch.no_grad():
                p = torch.softmax(model(img_d, txt, msk), dim=1)[0, pred].item()
            del_probs.append(p)

        # ---- INSERTION ----
        ins_probs = [0.0]
        img_i = torch.zeros_like(img)
        for step in range(STEPS):
            for pi in ranked[step*PER_STEP:(step+1)*PER_STEP]:
                r = pi.item() // 14; c = pi.item() % 14
                img_i[:, :, r*32:(r+1)*32, c*32:(c+1)*32] = \
                    img[:, :, r*32:(r+1)*32, c*32:(c+1)*32]
            with torch.no_grad():
                p = torch.softmax(model(img_i, txt, msk), dim=1)[0, pred].item()
            ins_probs.append(p)

        del_curves.append(del_probs)
        ins_curves.append(ins_probs)
        del img, txt, msk, attr, img_d, img_i
        clear_gpu()

    except Exception as e:
        print(f"Hata [{idx}]: {e}")
        clear_gpu()

n = len(del_curves)
print(f"Toplam örnek: {n}")

del_arr = np.array(del_curves)   # (N, STEPS+1)
ins_arr = np.array(ins_curves)

del_mean = del_arr.mean(0);  del_std = del_arr.std(0)
ins_mean = ins_arr.mean(0);  ins_std = ins_arr.std(0)
x        = np.linspace(0, 100, STEPS+1)

del_auc = float(np.trapz(del_mean, x) / 100)
ins_auc = float(np.trapz(ins_mean, x) / 100)

print(f"\nDeletion  AUC (düşük = iyi): {del_auc:.4f}")
print(f"Insertion AUC (yüksek = iyi): {ins_auc:.4f}")

# --- Grafik ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.plot(x, del_mean, color='#1565C0', linewidth=2.5,
        label=f'RE-LIG  (AUC = {del_auc:.3f})')
ax.fill_between(x, del_mean - del_std, del_mean + del_std,
                alpha=0.18, color='#1565C0')
ax.set_xlabel('Fraction of Patches Removed (%, Most Important First)', fontsize=11)
ax.set_ylabel('Model Confidence Score', fontsize=11)
ax.set_title('Deletion AUC  (Lower = More Faithful)', fontsize=11, fontweight='bold')
ax.legend(fontsize=10); ax.set_xlim(0, 100); ax.set_ylim(bottom=0); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(x, ins_mean, color='#C62828', linewidth=2.5,
        label=f'RE-LIG  (AUC = {ins_auc:.3f})')
ax.fill_between(x, ins_mean - ins_std, ins_mean + ins_std,
                alpha=0.18, color='#C62828')
ax.set_xlabel('Fraction of Patches Inserted (%, Most Important First)', fontsize=11)
ax.set_ylabel('Model Confidence Score', fontsize=11)
ax.set_title('Insertion AUC  (Higher = More Faithful)', fontsize=11, fontweight='bold')
ax.legend(fontsize=10); ax.set_xlim(0, 100); ax.set_ylim(bottom=0); ax.grid(alpha=0.3)

plt.suptitle(
    f'Deletion & Insertion AUC Curves — RE-LIG  (N={n}, SLAKE Test Set)\n'
    'Shaded area: ±1 std',
    fontsize=12, fontweight='bold')
plt.tight_layout()
out_png = OUT_DIR / 'figure5_deletion_insertion_auc.png'
fig.savefig(out_png, dpi=300, bbox_inches='tight')
plt.close()
print(f"Kaydedildi: {out_png}")

# JSON kaydet
with open(OUT_DIR / 'del_ins_results.json', 'w') as f:
    json.dump({
        'deletion_auc':          del_auc,
        'insertion_auc':         ins_auc,
        'n_samples':             n,
        'deletion_mean_curve':   del_mean.tolist(),
        'insertion_mean_curve':  ins_mean.tolist(),
        'deletion_std_curve':    del_std.tolist(),
        'insertion_std_curve':   ins_std.tolist(),
        'x_axis_pct':            x.tolist(),
    }, f, indent=2)
print(f"Kaydedildi: {OUT_DIR}/del_ins_results.json")
