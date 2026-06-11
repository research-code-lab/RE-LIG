# AOPC Analysis — 4-Configuration Ablation + Bootstrap CI
# ========================================================
# Bootstrap %95 CI eklendi.
# Ablation tablosu 4 konfigürasyona genişletildi.
# Memory-optimized for Kaggle T4 GPU (15GB VRAM).
# Maskeleme protokolü: sıfır maskeleme (img × 0) — tüm yöntemler aynı protokol.

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
from scipy.stats import mannwhitneyu, wilcoxon

from med_vqa_model import MedVQAModel
from data_utils import (SlakeDataset, build_slake_vocab, load_slake_data,
                        build_vqarad_vocab, load_vqarad_split)
from captum.attr import IntegratedGradients, LayerIntegratedGradients, NoiseTunnel
from relig_config import RELIG_CONFIG

# ------------------------------------------------------------------ #
# CLI argümanları — sed ile dosya düzenlemek gerekmez                 #
# Örnek: python aopc_analizi.py --dataset slake --model_path ...      #
# ------------------------------------------------------------------ #
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--dataset',    type=str, default=None, choices=['slake', 'vqarad'])
_parser.add_argument('--model_path', type=str, default=None)
_parser.add_argument('--data_dir',   type=str, default=None)
_parser.add_argument('--num_samples',type=int, default=50)
_known, _ = _parser.parse_known_args()

# Varsayılan dataset (argüman verilmezse bu değer kullanılır)
DATASET = _known.dataset or 'vqarad'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if DATASET == 'vqarad':
    # VQA-RAD model yolları (CLI > checkpoint dizini > varsayılan)
    if _known.model_path:
        model_path = _known.model_path
    elif os.path.exists("/kaggle/input/datasets/anonymous/vqaraddataset/model_best_vqa_rad.pth"):
        model_path = "/kaggle/input/datasets/anonymous/vqaraddataset/model_best_vqa_rad.pth"
    elif os.path.exists("/kaggle/working/outputs_vqarad/model_best.pth"):
        model_path = "/kaggle/working/outputs_vqarad/model_best.pth"
    else:
        model_path = "outputs/model_best.pth"
    data_dir = None  # VQA-RAD için kullanılmaz
else:
    # SLAKE model yolları
    if _known.model_path:
        model_path = _known.model_path
    elif os.path.exists("/kaggle/input/datasets/anonymous/slakemodel/model_best_slake.pth"):
        model_path = "/kaggle/input/datasets/anonymous/slakemodel/model_best_slake.pth"
    elif os.path.exists("/kaggle/working/outputs_seed42/model_best.pth"):
        model_path = "/kaggle/working/outputs_seed42/model_best.pth"
    else:
        model_path = "outputs/model_best.pth"
    data_dir = Path(_known.data_dir) if _known.data_dir else Path("/kaggle/input/datasets/amoghdumbre/slakemedvqa/Slake1.0")


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def bootstrap_ci(scores, n_bootstrap=1000, ci=0.95):
    # Bootstrap %95 guven araligi (ortalama icin)
    if len(scores) < 2:
        return float('nan'), float('nan')
    boot = [np.mean(np.random.choice(scores, size=len(scores), replace=True))
            for _ in range(n_bootstrap)]
    alpha = (1 - ci) / 2
    return float(np.percentile(boot, alpha * 100)), float(np.percentile(boot, (1 - alpha) * 100))


def aligned_scores(res_a, res_b):
    """Iki konfigurasyonun skorlarini ORNEK INDEKSINE gore hizalar.

    Wilcoxon signed-rank PAIRED test, ayni ornegin iki yontemdeki skorunu
    eslestirmeyi gerektirir. Bir yontemde (ornegin OOM nedeniyle) atlanan
    bir ornek digerinde atlanmamis olabilir; bu durumda listeleri sirayla
    eslestirmek YANLIS olur. Bu fonksiyon yalnizca her iki yontemde de
    gecerli olan ortak orneklerin skorlarini, indeks sirasiyla dondurur.
    """
    map_a = dict(zip(res_a.get('valid_indices', []), res_a.get('per_sample_scores', [])))
    map_b = dict(zip(res_b.get('valid_indices', []), res_b.get('per_sample_scores', [])))
    common = sorted(set(map_a) & set(map_b))
    a = np.array([map_a[i] for i in common])
    b = np.array([map_b[i] for i in common])
    return a, b


print(f"Loading data and model... (dataset={DATASET})")
if DATASET == 'vqarad':
    answer2idx = build_vqarad_vocab()
    test_data  = load_vqarad_split('test')
else:
    answer2idx = build_slake_vocab(data_dir)
    test_data  = load_slake_data(data_dir, 'test')
dataset    = SlakeDataset(test_data, answer2idx, image_size=448, is_train=False)
dataset.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

try:
    ckpt = torch.load(model_path, map_location=device)
    sd   = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    # KRITIK: num_classes checkpoint'ten al — vocab ile eslesme garantisi.
    # Eski kod len(answer2idx) kullaniyordu; egitimde farkli vocab kullanilmissa
    # sinif sayisi uyusmaz ve model yanlis cikti boyutunda olusturulurdu.
    if 'num_classes' in ckpt:
        num_classes = ckpt['num_classes']
    else:
        classifier_keys = [k for k in sd if 'classifier' in k and 'weight' in k]
        num_classes = sd[classifier_keys[-1]].shape[0]
    print(f"✅ Checkpoint'ten num_classes: {num_classes}")
    model = MedVQAModel(num_classes=num_classes, image_size=448).to(device)
    model.load_state_dict(sd, strict=False)
    print("Model loaded.")
except Exception as e:
    print(f"Model loading error: {e}")
    num_classes = len(answer2idx)
    model = MedVQAModel(num_classes=num_classes, image_size=448).to(device)
model.eval()

if hasattr(model.vit, 'embeddings'):
    target_layer = model.vit.embeddings
elif hasattr(model.vit, 'vision_model'):
    target_layer = model.vit.vision_model.embeddings
else:
    target_layer = model.vit


# ------------------------------------------------------------------
# Attribution fonksiyonları — her biri 14x14 yama haritası döndürür
# ------------------------------------------------------------------
class _PixelWrapper(nn.Module):
    def __init__(self, model, input_ids, attention_mask):
        super().__init__()
        self.model, self.input_ids, self.attention_mask = model, input_ids, attention_mask

    def forward(self, image):
        bs = image.shape[0]
        return self.model(image,
                          self.input_ids.expand(bs, -1),
                          self.attention_mask.expand(bs, -1))


def _attr_vanilla_ig(model, img, txt, msk, pred, device):
    wrapper = _PixelWrapper(model, txt, msk)
    ig   = IntegratedGradients(wrapper)
    attr = ig.attribute(img, baselines=torch.zeros_like(img),
                        target=pred, n_steps=20)
    attr = attr.sum(dim=1).squeeze().abs()
    attr = F.adaptive_avg_pool2d(attr.unsqueeze(0).unsqueeze(0), (14, 14)).squeeze()
    return attr.detach().cpu()


def _attr_pixel_ig_nt(model, img, txt, msk, pred, device):
    wrapper = _PixelWrapper(model, txt, msk)
    ig  = IntegratedGradients(wrapper)
    nt  = NoiseTunnel(ig)
    attr = nt.attribute(
        img, baselines=torch.zeros_like(img), target=pred,
        n_steps=RELIG_CONFIG['n_steps'],
        nt_type=RELIG_CONFIG['nt_type'],
        nt_samples=RELIG_CONFIG['nt_samples'],
        stdevs=RELIG_CONFIG['stdev'],
        internal_batch_size=1,
    )
    attr = attr.sum(dim=1).squeeze().abs()
    attr = F.adaptive_avg_pool2d(attr.unsqueeze(0).unsqueeze(0), (14, 14)).squeeze()
    return attr.detach().cpu()


def _attr_layer_ig(model, img, txt, msk, pred, device):
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
    return attr.detach().cpu()


def _attr_relig(model, img, txt, msk, pred, device):
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
    return attr.detach().cpu()


CONFIGS = {
    'Vanilla IG':    _attr_vanilla_ig,
    'Pixel IG + NT': _attr_pixel_ig_nt,
    'Layer-IG':      _attr_layer_ig,
    'RE-LIG':        _attr_relig,
}


# ------------------------------------------------------------------
# Tek bir konfigürasyon için AOPC hesapla (sıfır maskeleme)
# ------------------------------------------------------------------
def calculate_aopc(model, dataset, attr_fn, name, indices, steps=10):
    model.eval()
    aopc_scores = []
    valid_indices = []   # her skorun ait oldugu ornek indeksi (paired test icin)
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
    return mean_v, std_v, aopc_scores, lo, hi, valid_indices


# ------------------------------------------------------------------
# Görselleştirme: 4 yöntem histogram + CI band
# ------------------------------------------------------------------
def visualize_4configs(all_results, path):
    names  = list(all_results.keys())
    colors = {'Vanilla IG': '#2196F3', 'Pixel IG + NT': '#9C27B0',
              'Layer-IG': '#FF9800', 'RE-LIG': '#F44336'}
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 5))
    if len(names) == 1: axes = [axes]

    for ax, name in zip(axes, names):
        r      = all_results[name]
        scores = r['per_sample_scores']
        if not scores:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            continue
        mean_v = r['mean']
        lo, hi = r['ci_lo'], r['ci_hi']
        ax.hist(scores, bins=12, color=colors.get(name, '#607D8B'),
                edgecolor='black', alpha=0.75)
        ax.axvline(mean_v, color='black', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_v:.4f}')
        ax.axvspan(lo, hi, alpha=0.18, color='gray',
                   label=f'95% CI: [{lo:.4f}, {hi:.4f}]')
        ax.set_title(f'{name}\nn={len(scores)}, AOPC={mean_v:.4f}',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('AOPC Score', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('AOPC Distribution — 4-Configuration Ablation\n(Zero-masking protocol)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def visualize_boxplot(all_results, path):
    """Box plot + Wilcoxon anlamlılık bracketları — makale ana figürü."""
    order  = ['Vanilla IG', 'Pixel IG + NT', 'Layer-IG', 'RE-LIG']
    colors = {'Vanilla IG': '#2196F3', 'Pixel IG + NT': '#9C27B0',
              'Layer-IG': '#FF9800', 'RE-LIG': '#F44336'}
    names  = [n for n in order if n in all_results]
    data   = [all_results[n]['per_sample_scores'] for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color='black', linewidth=2),
                    flierprops=dict(marker='o', markersize=3, alpha=0.4))
    for patch, name in zip(bp['boxes'], names):
        patch.set_facecolor(colors.get(name, '#607D8B'))
        patch.set_alpha(0.72)

    for i, (d, name) in enumerate(zip(data, names), 1):
        jx = np.random.normal(i, 0.06, size=len(d))
        ax.scatter(jx, d, alpha=0.22, s=9, color=colors.get(name, '#607D8B'), zorder=3)

    relig_pos = names.index('RE-LIG') + 1 if 'RE-LIG' in names else None
    if relig_pos:
        all_vals = [v for d in data for v in d]
        y_top    = max(all_vals)
        step     = (max(all_vals) - min(all_vals)) * 0.09

        for offset, cmp_name in enumerate(['Vanilla IG', 'Pixel IG + NT', 'Layer-IG']):
            if cmp_name not in names:
                continue
            cmp_pos = names.index(cmp_name) + 1
            p = all_results.get(cmp_name, {}).get('wilcoxon_p', None)
            if p is None:
                continue
            label = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
            y = y_top + step * (offset + 0.8)
            ax.plot([cmp_pos, cmp_pos, relig_pos, relig_pos],
                    [y, y + step * 0.2, y + step * 0.2, y], 'k-', lw=1.2)
            ax.text((cmp_pos + relig_pos) / 2, y + step * 0.25, label,
                    ha='center', va='bottom', fontsize=13, fontweight='bold')

    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel('AOPC Score', fontsize=12)
    ax.set_title(
        f'AOPC — 4-Configuration Ablation (n={len(data[0])})\n'
        'Wilcoxon signed-rank paired test  |  *** p<0.001  n.s. p≥0.05',
        fontsize=11, fontweight='bold')
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def visualize_paired_differences(all_results, path):
    """Her örnek için (RE-LIG − diğer) fark grafiği — paired test gerekçesi."""
    relig_res = all_results.get('RE-LIG', {})
    cmps = ['Vanilla IG', 'Pixel IG + NT', 'Layer-IG']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, cmp_name in zip(axes, cmps):
        # Skorlari ornek indeksine gore hizala (Wilcoxon ile ayni ciftler)
        ref, cmp = aligned_scores(relig_res, all_results.get(cmp_name, {}))
        if len(ref) == 0 or len(cmp) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            continue

        diff        = ref - cmp
        sorted_diff = np.sort(diff)
        bar_colors  = ['#4CAF50' if d >= 0 else '#E53935' for d in sorted_diff]
        pct_pos     = 100 * np.mean(diff > 0)
        mean_diff   = np.mean(diff)

        ax.bar(range(len(sorted_diff)), sorted_diff, color=bar_colors,
               alpha=0.75, width=1.0, edgecolor='none')
        ax.axhline(0, color='black', linewidth=1.2)
        ax.axhline(mean_diff, color='navy', linewidth=2, linestyle='--',
                   label=f'Mean diff: {mean_diff:+.4f}')

        p = all_results.get(cmp_name, {}).get('wilcoxon_p', None)
        p_str = f'  (p={p:.4f})' if p is not None else ''
        ax.set_title(
            f'RE-LIG − {cmp_name}\n{pct_pos:.0f}% samples: RE-LIG better{p_str}',
            fontsize=10, fontweight='bold')
        ax.set_xlabel('Samples (sorted by difference)', fontsize=9)
        ax.set_ylabel('AOPC Difference', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle(
        'Paired AOPC Difference: RE-LIG vs. Baselines\n'
        'Green = RE-LIG better  |  Red = Baseline better',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    # Tam tekrarlanabilirlik (reproducibility):
    # Hem numpy (ornek secimi) hem torch (NoiseTunnel gurultu ornekleri)
    # tohumlanir; boylece RE-LIG/Pixel-IG+NT sonuclari her calistirmada ayni cikar.
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    _n = _known.num_samples
    shared_indices = np.random.choice(len(dataset), min(_n, len(dataset)), replace=False)

    all_results = {}
    for name, fn in CONFIGS.items():
        clear_gpu()
        mean_v, std_v, scores, lo, hi, valid_idx = calculate_aopc(
            model, dataset, fn, name, shared_indices, steps=10)
        all_results[name] = {
            'mean': mean_v, 'std': std_v,
            'ci_lo': lo, 'ci_hi': hi,
            'n': len(scores),
            'per_sample_scores': scores,
            'valid_indices': valid_idx,
        }

    print(f"\n{'='*65}")
    print(f"{'Method':<15} {'AOPC':>8}  {'95% CI':>22}  {'n':>5}")
    print("-" * 55)
    for name, r in all_results.items():
        print(f"{name:<15} {r['mean']:>8.4f}  [{r['ci_lo']:>7.4f}, {r['ci_hi']:>7.4f}]  {r['n']:>5}")
    print("=" * 65)

    # Göreceli kazanımlar
    if 'Vanilla IG' in all_results and 'RE-LIG' in all_results:
        base = all_results['Vanilla IG']['mean']
        gain = (all_results['RE-LIG']['mean'] - base) / abs(base) * 100 if base else 0
        print(f"\nRE-LIG vs Vanilla IG    : {gain:+.1f}%")
    if 'Pixel IG + NT' in all_results and 'RE-LIG' in all_results:
        base = all_results['Pixel IG + NT']['mean']
        gain = (all_results['RE-LIG']['mean'] - base) / abs(base) * 100 if base else 0
        print(f"RE-LIG vs Pixel IG + NT : {gain:+.1f}%")
    if 'Layer-IG' in all_results and 'RE-LIG' in all_results:
        base = all_results['Layer-IG']['mean']
        gain = (all_results['RE-LIG']['mean'] - base) / abs(base) * 100 if base else 0
        print(f"RE-LIG vs Layer-IG      : {gain:+.1f}%")

    # İstatistiksel anlamlılık testleri
    # Tüm yöntemler aynı shared_indices üzerinde çalışır → paired (eşleştirilmiş) veri.
    # Paired veri için Wilcoxon signed-rank test, Mann-Whitney U'dan daha güçlüdür.
    # Uzunluklar eşleşmezse (nadir OOM durumu) Mann-Whitney U'ya düşülür.
    print("\n--- İstatistiksel Anlamlılık (Wilcoxon signed-rank, paired, RE-LIG > diğer) ---")
    relig_res = all_results.get('RE-LIG', {})
    for cmp_name in ['Vanilla IG', 'Pixel IG + NT', 'Layer-IG']:
        cmp_res = all_results.get(cmp_name, {})
        if not relig_res.get('per_sample_scores') or not cmp_res.get('per_sample_scores'):
            continue
        try:
            # ORTAK orneklerde indekse gore hizalanmis skorlar -> gercek paired test
            ref_a, cmp_a = aligned_scores(relig_res, cmp_res)
            if len(ref_a) < 2:
                print(f"  RE-LIG vs {cmp_name:<15}: yetersiz ortak ornek")
                continue
            stat, pval = wilcoxon(ref_a, cmp_a, alternative='greater')
            sig = "(*p<0.05)" if pval < 0.05 else "(n.s.)"
            print(f"  RE-LIG vs {cmp_name:<15}: p={pval:.4f} {sig}  [Wilcoxon, n_pair={len(ref_a)}]")
            all_results[cmp_name]['wilcoxon_p'] = float(pval)
            all_results[cmp_name]['wilcoxon_n_pairs'] = int(len(ref_a))
        except Exception as e:
            print(f"  Test hatası ({cmp_name}): {e}")

    out_json     = f'aopc_results_{DATASET}.json'
    out_png_hist = f'aopc_distribution_comparison_{DATASET}.png'
    out_png_box  = f'aopc_boxplot_{DATASET}.png'
    out_png_diff = f'aopc_paired_diff_{DATASET}.png'

    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_json}")

    visualize_4configs(all_results, out_png_hist)
    visualize_boxplot(all_results, out_png_box)
    visualize_paired_differences(all_results, out_png_diff)
