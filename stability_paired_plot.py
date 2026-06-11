# Stability paired-difference plot — RE-RUN YOK
# =============================================
# Mevcut stability_results_{dataset}.json'u okur ve her ornek icin
# (RE-LIG - Layer-IG) stabilite farkini ciddi bir paired grafikle gosterir.
# Boxplot marjinal dagilimi gosterir ve eslestirmeyi gizler; bu grafik ise
# "orneklerin %X'inde RE-LIG daha kararli" diyerek Wilcoxon p-degerinin
# gosterdigi tutarli ustunlugu GORUNUR kilar ().

import argparse, json
import numpy as np
import matplotlib.pyplot as plt

_p = argparse.ArgumentParser(add_help=False)
_p.add_argument('--dataset', default='slake')
_p.add_argument('--json', default=None, help='varsayilan: stability_results_{dataset}.json')
_a, _ = _p.parse_known_args()
path = _a.json or f'stability_results_{_a.dataset}.json'

with open(path) as f:
    d = json.load(f)

s = d['stability']
r = np.array(s['relig_per_sample'], dtype=float)
l = np.array(s['layer_ig_per_sample'], dtype=float)
mask = ~(np.isnan(r) | np.isnan(l))
r, l = r[mask], l[mask]

diff = r - l                       # >0  -> RE-LIG daha kararli
order = np.argsort(diff)
diff_sorted = diff[order]
n = len(diff)
pct_pos = 100.0 * np.mean(diff > 0)
mean_diff = float(diff.mean())
pval = float(s.get('wilcoxon_p_relig_greater', float('nan')))
sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else 'n.s.'))

colors = ['#4CAF50' if x >= 0 else '#E53935' for x in diff_sorted]
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(range(n), diff_sorted, color=colors, width=1.0, edgecolor='none', alpha=0.85)
ax.axhline(0, color='black', linewidth=1.2)
ax.axhline(mean_diff, color='navy', linewidth=2, linestyle='--',
           label=f'Mean difference: {mean_diff:+.4f}')
ax.set_title('Per-sample Saliency Stability Difference (RE-LIG − Layer-IG)\n'
             f'{pct_pos:.0f}% of samples: RE-LIG more stable   |   '
             f'Wilcoxon p={pval:.2e} {sig}  (n={n})',
             fontsize=11, fontweight='bold')
ax.set_xlabel('Test samples (sorted by difference)', fontsize=10)
ax.set_ylabel('Δ Mean pairwise Spearman\n(RE-LIG − Layer-IG)', fontsize=10)
ax.legend(fontsize=9, loc='upper left')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()

out = f'stability_paired_diff_{_a.dataset}.png'
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")
print(f"  %pozitif (RE-LIG daha kararli) = {pct_pos:.1f}%  |  ortalama fark = {mean_diff:+.5f}  |  p = {pval:.3e}  |  n = {n}")
