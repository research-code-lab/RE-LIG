# RE-LIG: A Faithfulness-Driven Layer Integrated Gradients Framework for Explainable Medical VQA

Source code for reproducing the experiments in the manuscript *"RE-LIG: A Faithfulness-Driven
Layer Integrated Gradients Framework for Explainable Medical Visual Question Answering"*
(anonymous submission).

RE-LIG combines **Layer Integrated Gradients** with **Noise Tunneling** (stochastic smoothing)
to produce faithful, noise-robust, patch-level visual saliency maps and token-level linguistic
attributions for a multimodal Med-VQA model (PubMedCLIP + BioLinkBERT + Co-Attention).

## Repository contents

| File | Purpose |
|---|---|
| `med_vqa_model.py` | Model architecture (PubMedCLIP vision encoder, BioLinkBERT text encoder, Co-Attention fusion) |
| `data_utils.py`, `answer_normalization.py` | SLAKE / VQA-RAD data loading and answer normalization |
| `relig_config.py` | Global RE-LIG hyperparameters (n_steps=50, nt_samples=30, stdev=0.05) |
| `main.py` | Training (supports transfer learning via `--pretrained_path`) |
| `test.py` | Accuracy evaluation (overall / closed / open) |
| `aopc_analizi_yeni.py` | AOPC faithfulness — 4-configuration ablation + Bootstrap CI + paired Wilcoxon |
| `stability_analysis.py` | Saliency stability under input perturbation (Spearman) + Total Variation |
| `stability_paired_plot.py` | Per-sample paired stability-difference plot (from the stability JSON) |
| `gradcam_analizi.py` | Experimental Grad-CAM comparison under the identical AOPC protocol |
| `deletion_insertion_curves.py` | Deletion / Insertion AUC curves |
| `interpret_final.py` | Qualitative multimodal saliency + token-attribution visualizations (`--failure_mode` for failure cases) |
| `failure_case_viz.py` | Failure-case mining and visualization |

## Setup

```bash
pip install -r requirements.txt
```

Datasets: **SLAKE** (English subset) and **VQA-RAD** are publicly available
(https://github.com/med-vqa/SLAKE and the VQA-RAD release).

## Checkpoint

The trained SLAKE checkpoint (`model_best.pth`, 221 answer classes) reproduces the reported
test accuracy **80.77 / 87.61 / 77.34** (overall / closed / open). It is available at:

> **[anonymized checkpoint download link — to be added]**

All scripts accept `--model_path /path/to/model_best.pth`.

## Reproducibility

All experiments are **fully deterministic** (fixed seed = 42 for NumPy + PyTorch + CUDA).
Re-running any script reproduces the reported numbers exactly.

```bash
# Accuracy (Table 4)
python test.py --dataset slake --model_path model_best.pth

# AOPC 4-configuration ablation (Table 5, Figure 4)
python aopc_analizi_yeni.py --dataset slake --model_path model_best.pth --num_samples 105

# Noise-tunneling stability (Table 6, Figure 6)
python stability_analysis.py --dataset slake --model_path model_best.pth
python stability_paired_plot.py --dataset slake --json stability_results_slake.json

# Grad-CAM comparison (Table 7)
python gradcam_analizi.py --dataset slake --model_path model_best.pth --num_samples 105

# Deletion / Insertion AUC (Figure 5)
python deletion_insertion_curves.py --dataset slake --model_path model_best.pth --num_samples 50

# Qualitative visualizations (Figure 8) and failure cases (Figure 9)
python interpret_final.py --dataset slake --model_path model_best.pth
python interpret_final.py --dataset slake --model_path model_best.pth --failure_mode
```

Raw output artifacts (`aopc_results_slake.json`, `stability_results_slake.json`,
`gradcam_results_slake.json`) are produced by the corresponding scripts so that every reported
table value can be independently verified.
