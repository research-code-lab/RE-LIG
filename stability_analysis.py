# Stability / Robustness Analysis — Noise Tunneling (NT) gerekcelendirmesi
# =======================================================================
# "NT'nin olculebilir bir faydasi yok; p=0.98, 30x
# hesap maliyeti, stabilite iddiasinin sayisal kaniti yok."
#
# Bu script NT'nin GERCEK faydasini (SmoothGrad'in asil iddiasini) olcer.
# AOPC degil — AOPC seviyesinde NT katki yapmiyor (zaten biliniyor). Burada
# saliency HARITASININ kalitesini iki eksende olcuyoruz:
#
#   1. Input-perturbation STABILITY: ayni goruntunun K kucuk-bozulmus
#      versiyonundan uretilen haritalarin tutarliligi (ortalama ikili
#      Spearman rank korelasyonu — YUKSEK = daha kararli). SmoothGrad'in
#      temel iddiasi: gurultu uzerinden ortalama almak haritayi girdi
#      gurultusune karsi dayanikli kilar.
#
#   2. Map SMOOTHNESS (Total Variation): haritadaki yuksek-frekans gurultu
#      (DUSUK TV = daha pürüzsüz harita). NT'nin "shattered gradient" gurultu-
#      sunu bastirdigi iddiasi.
#
# Karsilastirma: Layer-IG (NT yok) vs RE-LIG (NT var), Wilcoxon paired test.
# Test TARAFSIZ: sonuc NT'yi destekler ya da NT'nin demote edilmesi gerektigini
# gosterir. Memory-optimized for Kaggle T4 GPU.

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gc
import json
import os
from itertools import combinations
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon

from med_vqa_model import MedVQAModel
from data_utils import (SlakeDataset, build_slake_vocab, load_slake_data,
                        build_vqarad_vocab, load_vqarad_split)
from captum.attr import LayerIntegratedGradients, NoiseTunnel
from relig_config import RELIG_CONFIG

# ------------------------------------------------------------------ #
# CLI argümanları                                                     #
# ------------------------------------------------------------------ #
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--dataset',     type=str, default=None, choices=['slake', 'vqarad'])
_parser.add_argument('--model_path',  type=str, default=None)
_parser.add_argument('--data_dir',    type=str, default=None)
_parser.add_argument('--num_samples', type=int, default=20,
                     help='Kac goruntu uzerinde olculecek (varsayilan 20)')
_parser.add_argument('--n_perturb',   type=int, default=5,
                     help='Her goruntu icin kac bozulmus versiyon (varsayilan 5)')
_parser.add_argument('--eval_sigma',  type=float, default=0.10,
                     help='Girdi bozma gurultusunun std sapmasi (varsayilan 0.10). '
                          'NT ic gurultusunden (0.05) farkli secilir ki test adil olsun.')
_known, _ = _parser.parse_known_args()

DATASET = _known.dataset or 'slake'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model / veri yollari (CLI > Kaggle varsayilanlari)
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

if hasattr(model.vit, 'embeddings'):
    target_layer = model.vit.embeddings
elif hasattr(model.vit, 'vision_model'):
    target_layer = model.vit.vision_model.embeddings
else:
    target_layer = model.vit


# ------------------------------------------------------------------
# Attribution fonksiyonlari (aopc_analizi_yeni.py ile ayni) — 14x14 harita
# ------------------------------------------------------------------
def _attr_layer_ig(img, txt, msk, pred):
    def fwd(image):
        bs = image.shape[0]
        return model(image, txt.expand(bs, -1), msk.expand(bs, -1))
    lig  = LayerIntegratedGradients(fwd, target_layer)
    attr = lig.attribute(img, baselines=torch.zeros_like(img),
                         target=pred, n_steps=RELIG_CONFIG['n_steps'],
                         attribute_to_layer_input=False)
    attr = attr.sum(dim=-1).squeeze()
    if attr.shape[0] == 197: attr = attr[1:]
    if attr.numel() == 196: attr = attr.reshape(14, 14)
    return attr.detach().cpu().numpy()


def _attr_relig(img, txt, msk, pred):
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
    if attr.numel() == 196: attr = attr.reshape(14, 14)
    return attr.detach().cpu().numpy()


METHODS = {'Layer-IG': _attr_layer_ig, 'RE-LIG': _attr_relig}


# ------------------------------------------------------------------
# Metrikler
# ------------------------------------------------------------------
def pairwise_stability(maps):
    """Haritalar arasi ortalama ikili Spearman rank korelasyonu (yuksek=kararli)."""
    flats = [m.flatten() for m in maps]
    vals = []
    for a, b in combinations(flats, 2):
        # Sabit (varyanssiz) harita -> korelasyon tanimsiz; atla
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        rho, _ = spearmanr(a, b)
        if not np.isnan(rho):
            vals.append(rho)
    return float(np.mean(vals)) if vals else float('nan')


def total_variation(m):
    """14x14 haritanin normalize edilmis toplam degisimi (dusuk=pürüzsüz)."""
    m = m.reshape(14, 14).astype(np.float64)
    rng = m.max() - m.min()
    if rng > 0:
        m = (m - m.min()) / rng
    dh = np.abs(np.diff(m, axis=0)).mean()
    dw = np.abs(np.diff(m, axis=1)).mean()
    return float((dh + dw) / 2.0)


# ------------------------------------------------------------------
# Ana islem
# ------------------------------------------------------------------
def run_stability(num_samples, n_perturb, eval_sigma):
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    res = {m: {'stability': [], 'tv': [], 'idx': []} for m in METHODS}

    print(f"\nProtokol: {num_samples} goruntu x {n_perturb} bozulma "
          f"(sigma={eval_sigma}); yalnizca dogru tahmin edilenler.\n")

    for idx in tqdm(indices, desc="Stability"):
        try:
            clear_gpu()
            item = dataset[idx]
            img  = item['image'].to(device).unsqueeze(0).float()
            txt  = item['input_ids'].to(device).unsqueeze(0)
            msk  = item['attention_mask'].to(device).unsqueeze(0)
            tgt  = item['answer'].item() if hasattr(item['answer'], 'item') else item['answer']

            with torch.no_grad():
                pred = model(img, txt, msk).argmax(1).item()
            if pred != tgt:
                continue

            lo, hi = img.min().item(), img.max().item()

            # Tum yontemler AYNI bozulmus girdileri gorsun -> adil + paired
            torch.manual_seed(1000 + int(idx))
            perturbed_inputs = []
            for _ in range(n_perturb):
                noise = torch.randn_like(img) * eval_sigma
                perturbed_inputs.append((img + noise).clamp(lo, hi))

            ok = True
            method_maps = {}
            for name, fn in METHODS.items():
                maps = []
                for xp in perturbed_inputs:
                    try:
                        m = fn(xp.clone().detach().requires_grad_(True), txt, msk, pred)
                    except RuntimeError as e:
                        if 'out of memory' in str(e).lower():
                            clear_gpu(); ok = False; break
                        raise e
                    maps.append(m)
                if not ok or len(maps) < 2:
                    ok = False; break
                method_maps[name] = maps

            if not ok:
                clear_gpu(); continue

            for name, maps in method_maps.items():
                res[name]['stability'].append(pairwise_stability(maps))
                res[name]['tv'].append(float(np.mean([total_variation(m) for m in maps])))
                res[name]['idx'].append(int(idx))

            del img, txt, msk
            clear_gpu()

        except Exception as e:
            print(f"Error at {idx}: {e}")
            clear_gpu()

    return res


def summarize(res):
    print(f"\n{'='*70}")
    print(f"{'Metric':<22}{'Layer-IG':>14}{'RE-LIG':>14}{'Wilcoxon p':>16}")
    print("-" * 70)

    out = {'config': {'eval_sigma': _known.eval_sigma,
                      'n_perturb': _known.n_perturb,
                      'nt_samples': RELIG_CONFIG['nt_samples'],
                      'nt_stdev': RELIG_CONFIG['stdev']}}

    # --- STABILITY: RE-LIG > Layer-IG bekleriz (yuksek = kararli) ---
    s_l = np.array(res['Layer-IG']['stability'])
    s_r = np.array(res['RE-LIG']['stability'])
    mask = ~(np.isnan(s_l) | np.isnan(s_r))
    s_l, s_r = s_l[mask], s_r[mask]
    try:
        _, p_stab = wilcoxon(s_r, s_l, alternative='greater')
    except Exception:
        p_stab = float('nan')
    lo_l, hi_l = bootstrap_ci(list(s_l)); lo_r, hi_r = bootstrap_ci(list(s_r))
    print(f"{'Stability (Spearman)':<22}{np.mean(s_l):>14.4f}{np.mean(s_r):>14.4f}{p_stab:>16.4g}")

    # --- TOTAL VARIATION: RE-LIG < Layer-IG bekleriz (dusuk = pürüzsüz) ---
    t_l = np.array(res['Layer-IG']['tv'])
    t_r = np.array(res['RE-LIG']['tv'])
    mask = ~(np.isnan(t_l) | np.isnan(t_r))
    t_l, t_r = t_l[mask], t_r[mask]
    try:
        _, p_tv = wilcoxon(t_l, t_r, alternative='greater')  # Layer-IG TV daha mi yuksek?
    except Exception:
        p_tv = float('nan')
    print(f"{'Total Variation':<22}{np.mean(t_l):>14.4f}{np.mean(t_r):>14.4f}{p_tv:>16.4g}")
    print("=" * 70)
    print(f"n (paired) = {len(s_l)}")
    print("\nYorum:")
    print("  - Stability'de RE-LIG anlamli sekilde YUKSEK ise (p<0.05) -> NT girdi")
    print("    gurultusune dayaniklilik saglar; NT savunulur.")
    print("  - TV'de RE-LIG anlamli sekilde DUSUK ise (p<0.05) -> NT daha pürüzsüz")
    print("    harita uretir; NT savunulur.")
    print("  - Ikisi de anlamsiz (n.s.) ise -> NT'yi dürüstçe opsiyonel modüle indir.")

    out['stability'] = {
        'layer_ig_mean': float(np.mean(s_l)), 'layer_ig_ci': [lo_l, hi_l],
        'relig_mean': float(np.mean(s_r)), 'relig_ci': [lo_r, hi_r],
        'wilcoxon_p_relig_greater': float(p_stab), 'n_pairs': int(len(s_l)),
        'layer_ig_per_sample': list(map(float, s_l)),
        'relig_per_sample': list(map(float, s_r)),
    }
    out['total_variation'] = {
        'layer_ig_mean': float(np.mean(t_l)), 'relig_mean': float(np.mean(t_r)),
        'wilcoxon_p_layerig_greater': float(p_tv), 'n_pairs': int(len(t_l)),
        'layer_ig_per_sample': list(map(float, t_l)),
        'relig_per_sample': list(map(float, t_r)),
    }
    return out


def visualize(out, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Stability
    s = out['stability']
    axes[0].boxplot([s['layer_ig_per_sample'], s['relig_per_sample']],
                    patch_artist=True, labels=['Layer-IG', 'RE-LIG'],
                    medianprops=dict(color='black', linewidth=2))
    for patch, c in zip(axes[0].artists, ['#FF9800', '#F44336']):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    p = s['wilcoxon_p_relig_greater']
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
    axes[0].set_title(f'Saliency Stability under Input Perturbation\n'
                      f'(higher = more stable)  RE-LIG>Layer-IG: p={p:.3g} {sig}',
                      fontsize=10, fontweight='bold')
    axes[0].set_ylabel('Mean pairwise Spearman correlation', fontsize=10)
    axes[0].grid(axis='y', alpha=0.3)

    # Panel 2: Total Variation
    t = out['total_variation']
    axes[1].boxplot([t['layer_ig_per_sample'], t['relig_per_sample']],
                    patch_artist=True, labels=['Layer-IG', 'RE-LIG'],
                    medianprops=dict(color='black', linewidth=2))
    for patch, c in zip(axes[1].artists, ['#FF9800', '#F44336']):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    p = t['wilcoxon_p_layerig_greater']
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
    axes[1].set_title(f'Saliency Map Smoothness\n'
                      f'(lower = smoother)  Layer-IG>RE-LIG: p={p:.3g} {sig}',
                      fontsize=10, fontweight='bold')
    axes[1].set_ylabel('Total Variation (normalized)', fontsize=10)
    axes[1].grid(axis='y', alpha=0.3)

    plt.suptitle('Noise Tunneling Effect on Saliency Map Quality',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    res = run_stability(_known.num_samples, _known.n_perturb, _known.eval_sigma)
    out = summarize(res)

    out_json = f'stability_results_{DATASET}.json'
    out_png  = f'stability_{DATASET}.png'
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_json}")
    visualize(out, out_png)
