# Grad-CAM Analysis — Deneysel Grad-CAM karsilastirmasi
# =====================================================
# "Bolum 3.3'te teorik Grad-CAM karsilastirmasi var;
# DENEYSEL olarak da Grad-CAM uygulayip sonuc bolumune ekleyin."
#
# Bu script Grad-CAM'i, diger 4 yontemle (Vanilla IG, Pixel IG+NT, Layer-IG,
# RE-LIG) BIREBIR AYNI AOPC protokolunde olcer:
#   - Ayni seed (42) ve ayni num_samples  -> ayni shared_indices (ayni goruntular)
#   - Ayni sifir-maskeleme protokolu (img x 0, 10 adim, 14x14 yama)
#   - Paired Wilcoxon signed-rank test (ornek indeksine gore hizalanmis)
# Boylece Grad-CAM dogrudan AOPC karsilastirma tablosuna 5. satir olarak girer.
#
# Grad-CAM tanimi: ViT'in SON transformer blogunun cikti aktivasyonlari (A) ve
# hedef-logit'e gore gradyanlari (G) uzerinden hesaplanir (jacobgil/pytorch-grad-cam
# transformer reshape yaklasimi):
#     w = mean_token(G)               # kanal onem agirliklari [dim]
#     cam = ReLU( sum_dim(w * A) )     # [196] -> 14x14 (CLS atilir)
# Single forward+backward; IG/NT'den cok daha ucuz (~30x daha hizli).
# Memory-optimized for Kaggle T4 GPU.

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gc
import json
import os
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from med_vqa_model import MedVQAModel
from data_utils import (SlakeDataset, build_slake_vocab, load_slake_data,
                        build_vqarad_vocab, load_vqarad_split)
from captum.attr import LayerIntegratedGradients, NoiseTunnel
from relig_config import RELIG_CONFIG

# ------------------------------------------------------------------ #
# CLI argumanlari                                                     #
# ------------------------------------------------------------------ #
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--dataset',     type=str, default=None, choices=['slake', 'vqarad'])
_parser.add_argument('--model_path',  type=str, default=None)
_parser.add_argument('--data_dir',    type=str, default=None)
_parser.add_argument('--num_samples', type=int, default=105,
                     help='AOPC icin cekilecek goruntu sayisi. AOPC scriptiyle AYNI '
                          'deger verilmeli ki ayni shared_indices (paired) olussun.')
_parser.add_argument('--qualitative', type=int, default=4,
                     help='Grad-CAM vs RE-LIG niteliksel gorsel icin panel sayisi '
                          '(0 = kapat). Her panel icin RE-LIG yeniden hesaplanir (~40s).')
_known, _ = _parser.parse_known_args()

DATASET = _known.dataset or 'slake'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model / veri yollari (CLI > Kaggle varsayilanlari) — stability_analysis.py ile ayni
if DATASET == 'vqarad':
    if _known.model_path:
        model_path = _known.model_path
    elif os.path.exists("/kaggle/input/datasets/anonymous/vqaraddataset/model_best_vqa_rad.pth"):
        model_path = "/kaggle/input/datasets/anonymous/vqaraddataset/model_best_vqa_rad.pth"
    else:
        model_path = "outputs/model_best.pth"
    data_dir = None
else:
    if _known.model_path:
        model_path = _known.model_path
    elif os.path.exists("/kaggle/input/datasets/anonymous/slakemodelyeni/model_best.pth"):
        model_path = "/kaggle/input/datasets/anonymous/slakemodelyeni/model_best.pth"
    else:
        model_path = "outputs/model_best.pth"
    data_dir = Path(_known.data_dir) if _known.data_dir else Path("/kaggle/input/datasets/amoghdumbre/slakemedvqa/Slake1.0")


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def bootstrap_ci(scores, n_bootstrap=1000, ci=0.95):
    scores = [s for s in scores if not np.isnan(s)]
    if len(scores) < 2:
        return float('nan'), float('nan')
    boot = [np.mean(np.random.choice(scores, size=len(scores), replace=True))
            for _ in range(n_bootstrap)]
    alpha = (1 - ci) / 2
    return float(np.percentile(boot, alpha * 100)), float(np.percentile(boot, (1 - alpha) * 100))


def aligned_scores(res_a, res_b):
    """Iki yontemin skorlarini ORNEK INDEKSINE gore hizalar (paired test icin).
    aopc_analizi_yeni.py'deki ile ayni mantik."""
    map_a = dict(zip(res_a.get('valid_indices', []), res_a.get('per_sample_scores', [])))
    map_b = dict(zip(res_b.get('valid_indices', []), res_b.get('per_sample_scores', [])))
    common = sorted(set(map_a) & set(map_b))
    a = np.array([map_a[i] for i in common])
    b = np.array([map_b[i] for i in common])
    return a, b


# ------------------------------------------------------------------
# Veri ve model
# ------------------------------------------------------------------
print(f"Loading data and model... (dataset={DATASET})")
if DATASET == 'vqarad':
    answer2idx = build_vqarad_vocab()
    test_data  = load_vqarad_split('test')
else:
    answer2idx = build_slake_vocab(data_dir)
    test_data  = load_slake_data(data_dir, 'test')
dataset = SlakeDataset(test_data, answer2idx, image_size=448, is_train=False)
dataset.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

ckpt = torch.load(model_path, map_location=device)
sd   = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
if 'num_classes' in ckpt:
    num_classes = ckpt['num_classes']
else:
    classifier_keys = [k for k in sd if 'classifier' in k and 'weight' in k]
    num_classes = sd[classifier_keys[-1]].shape[0]
print(f"✅ num_classes: {num_classes}")
model = MedVQAModel(num_classes=num_classes, image_size=448).to(device)
model.load_state_dict(sd, strict=False)
model.eval()

# Layer-IG / RE-LIG ile ayni hedef katman (niteliksel RE-LIG haritasi icin)
if hasattr(model.vit, 'embeddings'):
    ig_target_layer = model.vit.embeddings
elif hasattr(model.vit, 'vision_model'):
    ig_target_layer = model.vit.vision_model.embeddings
else:
    ig_target_layer = model.vit

# Grad-CAM hedefi: ViT'in SON transformer blogu (cikti aktivasyonlari)
def _find_last_block():
    enc = None
    if hasattr(model.vit, 'encoder'):
        enc = model.vit.encoder
    elif hasattr(model.vit, 'vision_model') and hasattr(model.vit.vision_model, 'encoder'):
        enc = model.vit.vision_model.encoder
    if enc is not None and hasattr(enc, 'layers') and len(enc.layers) > 0:
        return enc.layers[-1]
    raise RuntimeError("ViT encoder.layers bulunamadi; Grad-CAM hedef katmani secilemedi.")

gradcam_layer = _find_last_block()
print(f"✅ Grad-CAM hedef katman: {type(gradcam_layer).__name__} (son blok)")


# ------------------------------------------------------------------
# Grad-CAM — forward/backward hook
# ------------------------------------------------------------------
class GradCAM:
    """Son transformer blogunun cikti aktivasyonlari + gradyanlari uzerinden
    14x14 sinif-ayirici saliency haritasi uretir."""
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self._fh = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inp, out):
        act = out[0] if isinstance(out, (tuple, list)) else out
        self.activations = act
        # Bu aktivasyona gelen gradyani yakala (backward sirasinda dolar)
        if act.requires_grad:
            act.register_hook(self._save_grad)

    def _save_grad(self, grad):
        self.gradients = grad

    def remove(self):
        self._fh.remove()

    def __call__(self, img, txt, msk, pred):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(img, txt.expand(img.shape[0], -1), msk.expand(img.shape[0], -1))
        score = logits[0, pred]
        score.backward()

        A = self.activations[0].detach()   # [197, dim]
        G = self.gradients[0].detach()      # [197, dim]
        if A.shape[0] == 197:               # CLS token'i at
            A = A[1:]; G = G[1:]
        weights = G.mean(dim=0)             # [dim] — kanal onem agirliklari
        cam = F.relu((weights.unsqueeze(0) * A).sum(dim=1))  # [196]
        if cam.numel() == 196:
            cam = cam.reshape(14, 14)
        return cam.detach().cpu()


gradcam = GradCAM(model, gradcam_layer)


def _attr_gradcam(model, img, txt, msk, pred, device):
    """calculate_aopc'nin bekledigi imza ile uyumlu Grad-CAM sarmalayicisi."""
    return gradcam(img, txt, msk, pred)


def _relig_map(img, txt, msk, pred):
    """Niteliksel gorsel icin RE-LIG (Layer-IG + NT) 14x14 haritasi.
    aopc_analizi_yeni.py'deki _attr_relig ile ayni."""
    def fwd(image):
        bs = image.shape[0]
        return model(image, txt.expand(bs, -1), msk.expand(bs, -1))
    lig = LayerIntegratedGradients(fwd, ig_target_layer)
    nt  = NoiseTunnel(lig)
    attr = nt.attribute(
        img, baselines=torch.zeros_like(img), target=pred,
        n_steps=RELIG_CONFIG['n_steps'], nt_type=RELIG_CONFIG['nt_type'],
        nt_samples=RELIG_CONFIG['nt_samples'], stdevs=RELIG_CONFIG['stdev'],
        attribute_to_layer_input=False, internal_batch_size=1,
    )
    attr = attr.sum(dim=-1).squeeze()
    if attr.shape[0] == 197: attr = attr[1:]
    if attr.numel() == 196: attr = attr.reshape(14, 14)
    return attr.detach().cpu().numpy()


# ------------------------------------------------------------------
# AOPC hesabi — aopc_analizi_yeni.py'deki calculate_aopc ile BIREBIR ayni protokol
# ------------------------------------------------------------------
def calculate_aopc(model, dataset, attr_fn, name, indices, steps=10):
    model.eval()
    aopc_scores = []
    valid_indices = []
    print(f"\n--- {name} | samples={len(indices)}, steps={steps} ---")

    for idx in tqdm(indices, desc=name):
        try:
            clear_gpu()
            item = dataset[idx]
            img  = item['image'].to(device).unsqueeze(0).float()
            txt  = item['input_ids'].to(device).unsqueeze(0)
            msk  = item['attention_mask'].to(device).unsqueeze(0)
            tgt  = item['answer'].item() if hasattr(item['answer'], 'item') else item['answer']

            with torch.no_grad():
                out  = model(img, txt, msk)
                pred = out.argmax(1).item()
                orig = torch.softmax(out, dim=1)[0, pred].item()
            if pred != tgt:
                continue

            try:
                attr = attr_fn(model, img.requires_grad_(True), txt, msk, pred, device)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    clear_gpu(); continue
                raise e

            attr       = attr.flatten()
            sorted_idx = torch.argsort(attr, descending=True)
            per_step   = attr.shape[0] // steps

            probs = []
            for i in range(1, steps + 1):
                top_k  = sorted_idx[:i * per_step]
                mask_t = torch.ones_like(img)
                for pi in top_k:
                    r = pi.item() // 14
                    c = pi.item() % 14
                    mask_t[:, :, r*32:(r+1)*32, c*32:(c+1)*32] = 0
                perturbed = img * mask_t.to(device)
                with torch.no_grad():
                    p = torch.softmax(model(perturbed, txt, msk), dim=1)[0, pred].item()
                probs.append(p)

            aopc_scores.append(float(np.mean(orig - np.array(probs))))
            valid_indices.append(int(idx))
            del img, txt, msk, attr
            clear_gpu()

        except Exception as e:
            print(f"Error at {idx}: {e}")
            clear_gpu()

    mean_v = np.mean(aopc_scores) if aopc_scores else 0.0
    std_v  = np.std(aopc_scores)  if aopc_scores else 0.0
    lo, hi = bootstrap_ci(aopc_scores)
    print(f"AOPC: {mean_v:.4f} ± {std_v:.4f}  95%CI [{lo:.4f}, {hi:.4f}]  n={len(aopc_scores)}")
    return {'mean': mean_v, 'std': std_v, 'ci_lo': lo, 'ci_hi': hi,
            'n': len(aopc_scores), 'per_sample_scores': aopc_scores,
            'valid_indices': valid_indices}


# ------------------------------------------------------------------
# Niteliksel gorsel: orijinal | Grad-CAM | RE-LIG
# ------------------------------------------------------------------
def _to_display(img_t):
    """[3,H,W] normalize tensoru gosterilebilir [H,W,3] (0-1) hale getirir."""
    x = img_t.detach().cpu().numpy()
    x = np.transpose(x, (1, 2, 0))
    rng = x.max() - x.min()
    if rng > 0:
        x = (x - x.min()) / rng
    return x


def _upsample(cam, p_lo=20, p_hi=99):
    """14x14 haritayi 448x448'e buyutur (her yama 32x32) ve percentile-clip ile
    0-1 normalize eder. Tek bir uc deger tum haritayi ezmesin diye min-max yerine
    [p_lo, p_hi] persentil araligi kullanilir -> daha temiz, kontrastli harita."""
    cam = np.asarray(cam, dtype=np.float64)
    lo, hi = np.percentile(cam, p_lo), np.percentile(cam, p_hi)
    if hi > lo:
        cam = np.clip((cam - lo) / (hi - lo), 0, 1)
    else:
        cam = np.zeros_like(cam)
    return np.kron(cam, np.ones((32, 32)))


def visualize_qualitative(samples, path):
    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(11, 3.6 * n))
    if n == 1:
        axes = axes.reshape(1, 3)
    col_titles = ['Input', 'Grad-CAM', 'RE-LIG']
    for row, s in enumerate(samples):
        disp = s['image']
        for col, (title, cam) in enumerate(
                zip(col_titles, [None, s['gradcam'], s['relig']])):
            ax = axes[row, col]
            ax.imshow(disp, cmap='gray')
            if cam is not None:
                heat = _upsample(cam)
                # Dusuk-onem bolgelerini seffaf birak -> sadece onemli bolge renklensin
                heat_masked = np.ma.masked_where(heat < 0.15, heat)
                ax.imshow(heat_masked, cmap='jet', alpha=0.55)
            ax.axis('off')
            if row == 0:
                ax.set_title(title, fontsize=12, fontweight='bold')
        axes[row, 0].set_ylabel(s['question'][:40], fontsize=8)
        # soru metnini sol panele yaz (axis kapali oldugu icin text ile)
        axes[row, 0].text(0.5, -0.06, f"Q: {s['question'][:55]}", fontsize=8,
                          ha='center', va='top', transform=axes[row, 0].transAxes)
    plt.suptitle('Qualitative Comparison: Grad-CAM vs. RE-LIG',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# Uzamsal-cevapli soru filtresi: nitel figur icin SADECE belirli bir bolgeye
# isaret eden sorular secilir (modalite/normal-mi gibi 'global' sorular degil).
# Bu, atif kalitesine gore degil SORU TIPINE gore secim -> cherry-pick degil.
SPATIAL_KW = ['where', 'which', 'locat', 'organ', 'left', 'right', 'upper', 'lower',
              'lobe', 'lesion', 'mass', 'abnormal', 'contain', 'position', 'quadrant',
              'side', 'region', 'cardiomegaly']


def _is_spatial(q):
    ql = q.lower()
    return any(k in ql for k in SPATIAL_KW)


def run_qualitative(valid_indices, n_panels):
    samples = []
    print(f"\nNiteliksel gorsel: uzamsal-cevapli sorulardan en fazla {n_panels} ornek")
    for idx in valid_indices:
        if len(samples) >= n_panels:
            break
        try:
            clear_gpu()
            item = dataset[idx]
            try:
                q = dataset.tokenizer.decode(item['input_ids'].tolist(),
                                             skip_special_tokens=True)
            except Exception:
                q = ""
            if not _is_spatial(q):
                continue   # global soru -> atla
            img  = item['image'].to(device).unsqueeze(0).float()
            txt  = item['input_ids'].to(device).unsqueeze(0)
            msk  = item['attention_mask'].to(device).unsqueeze(0)
            with torch.no_grad():
                pred = model(img, txt, msk).argmax(1).item()
            cam_gc = gradcam(img.requires_grad_(True), txt, msk, pred).numpy()
            cam_rl = _relig_map(img.clone().detach().requires_grad_(True), txt, msk, pred)
            samples.append({'image': _to_display(item['image']),
                            'gradcam': cam_gc, 'relig': cam_rl, 'question': q})
            print(f"  + secildi (idx={idx}): {q[:60]}")
            clear_gpu()
        except Exception as e:
            print(f"Qualitative error at {idx}: {e}")
            clear_gpu()
    if len(samples) < n_panels:
        print(f"⚠ Sadece {len(samples)} uzamsal-cevapli ornek bulundu (istenen {n_panels}).")
    return samples


# ------------------------------------------------------------------
# Ana islem
# ------------------------------------------------------------------
if __name__ == "__main__":
    # AOPC scriptiyle AYNI seed + AYNI num_samples -> AYNI shared_indices (paired)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    shared_indices = np.random.choice(len(dataset),
                                      min(_known.num_samples, len(dataset)),
                                      replace=False)

    gc_res = calculate_aopc(model, dataset, _attr_gradcam, 'Grad-CAM',
                            shared_indices, steps=10)

    out_json = f'gradcam_results_{DATASET}.json'
    with open(out_json, 'w') as f:
        json.dump({'Grad-CAM': gc_res}, f, indent=2)
    print(f"\nSaved: {out_json}")

    # --- Mevcut AOPC sonuclariyla birlestir (varsa) ve paired Wilcoxon ---
    aopc_json = f'aopc_results_{DATASET}.json'
    if os.path.exists(aopc_json):
        with open(aopc_json) as f:
            aopc = json.load(f)
        aopc['Grad-CAM'] = gc_res

        order = ['Vanilla IG', 'Pixel IG + NT', 'Grad-CAM', 'Layer-IG', 'RE-LIG']
        print(f"\n{'='*65}")
        print(f"{'Method':<15} {'AOPC':>8}  {'95% CI':>22}  {'n':>5}")
        print("-" * 65)
        for name in order:
            if name in aopc:
                r = aopc[name]
                print(f"{name:<15} {r['mean']:>8.4f}  "
                      f"[{r['ci_lo']:>7.4f}, {r['ci_hi']:>7.4f}]  {r['n']:>5}")
        print("=" * 65)

        # Goreceli kazanim: RE-LIG vs Grad-CAM
        if 'RE-LIG' in aopc:
            base = gc_res['mean']
            gain = (aopc['RE-LIG']['mean'] - base) / abs(base) * 100 if base else 0
            print(f"\nRE-LIG vs Grad-CAM      : {gain:+.1f}%")

        # Paired Wilcoxon: RE-LIG ve Layer-IG, Grad-CAM'e karsi daha mi yuksek?
        print("\n--- Istatistiksel Anlamlilik (Wilcoxon signed-rank, paired) ---")
        for ref_name in ['RE-LIG', 'Layer-IG']:
            if ref_name not in aopc:
                continue
            ref_a, gc_a = aligned_scores(aopc[ref_name], gc_res)
            if len(ref_a) < 2:
                print(f"  {ref_name} vs Grad-CAM: yetersiz ortak ornek")
                continue
            try:
                _, p = wilcoxon(ref_a, gc_a, alternative='greater')
                sig = "(*p<0.05)" if p < 0.05 else "(n.s.)"
                print(f"  {ref_name:<8} > Grad-CAM : p={p:.4f} {sig}  [n_pair={len(ref_a)}]")
            except Exception as e:
                print(f"  Test hatasi ({ref_name}): {e}")

        with open(f'aopc_with_gradcam_{DATASET}.json', 'w') as f:
            json.dump(aopc, f, indent=2)
        print(f"\nSaved: aopc_with_gradcam_{DATASET}.json")
    else:
        print(f"\nNot: {aopc_json} bulunamadi — once aopc_analizi_yeni.py'yi ayni "
              f"--num_samples ile calistir, sonra birlesik tablo olusur.")

    # --- Niteliksel gorsel ---
    if _known.qualitative > 0 and gc_res['valid_indices']:
        samples = run_qualitative(gc_res['valid_indices'], _known.qualitative)
        if samples:
            visualize_qualitative(samples, f'gradcam_vs_relig_{DATASET}.png')

    gradcam.remove()
