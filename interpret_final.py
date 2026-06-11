"""
Q1 SCI Publication Quality - Multimodal Attribution Visualization
==================================================================
RE-LIG (Robust Layer Integrated Gradients) with Stochastic Smoothing
via Noise Tunneling for publication-ready visualizations.

Produces professional visualizations with:
- Focused attention heatmaps using NoiseTunnel (not over-smoothed)
- Numerical text attribution scores below each token
- Publication-ready layout (300 DPI, subplot labels)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import argparse
from tqdm import tqdm
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from transformers import AutoTokenizer
from med_vqa_model import MedVQAModel
from data_utils import SlakeDataset, build_slake_vocab, load_slake_data, build_vqarad_vocab, load_vqarad_split
from matplotlib.patches import Rectangle, FancyBboxPatch
from pathlib import Path
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 14

# --- CAPTUM with NoiseTunnel ---
from captum.attr import LayerIntegratedGradients, NoiseTunnel
from relig_config import RELIG_CONFIG

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='slake', choices=['slake', 'vqarad'])
parser.add_argument('--model_path', type=str, default="/kaggle/working/outputs_slake/model_best.pth")
parser.add_argument('--data_dir', type=str, default="/kaggle/input/datasets/amoghdumbre/slakemedvqa/Slake1.0")
parser.add_argument('--output_dir', type=str, default="final_visuals_q1")
parser.add_argument('--num_samples', type=int, default=20)
parser.add_argument('--vocab_from_train', action='store_true', default=False)
parser.add_argument('--min_answer_freq', type=int, default=1)
parser.add_argument('--unknown_token', type=str, default=None)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--failure_mode', action='store_true', default=False,
                    help='Visualize misclassified examples for failure analysis')
args, _ = parser.parse_known_args()

model_path = args.model_path if os.path.exists(args.model_path) else "outputs/model_best.pth"
data_dir = Path(args.data_dir)
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# RELIG_CONFIG is imported from relig_config.py (above):
# n_steps=50, nt_samples=30, stdev=0.05 — consistent across all XAI scripts.

# =============================================================================
# 2. LOAD MODEL & DATA
# =============================================================================
print(f"📚 Loading Data & Model ({args.dataset.upper()})...")
if args.dataset == 'vqarad':
    answer2idx = build_vqarad_vocab(
        use_normalization=True,
        min_answer_freq=args.min_answer_freq,
        unknown_token=args.unknown_token,
    )
    test_data = load_vqarad_split('test')
    if not args.output_dir or args.output_dir == 'final_visuals_q1':
        output_dir = 'final_visuals_vqarad'
        os.makedirs(output_dir, exist_ok=True)
else:
    vocab_splits = ['train', 'val'] if args.vocab_from_train else None
    answer2idx = build_slake_vocab(
        data_dir,
        vocab_splits=vocab_splits,
        min_answer_freq=args.min_answer_freq,
        unknown_token=args.unknown_token,
    )
    test_data = load_slake_data(data_dir, 'test')

idx2answer = {v: k for k, v in answer2idx.items()}
dataset = SlakeDataset(test_data, answer2idx, image_size=448, is_train=False, unknown_token=args.unknown_token)
try:
    dataset.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
except Exception:
    dataset.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

try:
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    if 'num_classes' in checkpoint:
        num_classes = checkpoint['num_classes']
    else:
        keys = list(state_dict.keys())
        last_key = [k for k in keys if 'classifier' in k and 'weight' in k][-1]
        num_classes = state_dict[last_key].shape[0]
    model = MedVQAModel(num_classes=num_classes, image_size=448).to(device)
    model.load_state_dict(state_dict, strict=False)
    if num_classes != len(answer2idx):
        print(f"⚠️ Vocab size mismatch: vocab={len(answer2idx)} vs checkpoint={num_classes}")
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"Model loading error: {e}")

model.eval()

# =============================================================================
# 3. RE-LIG ATTRIBUTION TOOLS (with Noise Tunneling)
# =============================================================================

# --- A) Determine Target Layer for Visual Attribution ---
if hasattr(model.vit, 'embeddings'):
    vis_layer = model.vit.embeddings
elif hasattr(model.vit, 'vision_model'):
    vis_layer = model.vit.vision_model.embeddings
else:
    vis_layer = model.vit

print(f"🎯 Target Visual Layer: {type(vis_layer).__name__}")

# Store text inputs globally for wrapper
_current_input_ids = None
_current_attention_mask = None

def image_forward_for_lig(inputs):
    """
    Forward wrapper for LayerIntegratedGradients.
    Text inputs are fixed; only image varies.
    """
    global _current_input_ids, _current_attention_mask
    
    batch_size = inputs.shape[0]
    
    # Expand text to match batch size created by Captum
    if _current_input_ids is not None:
        if _current_input_ids.shape[0] != batch_size:
            input_ids = _current_input_ids.expand(batch_size, -1)
            attention_mask = _current_attention_mask.expand(batch_size, -1)
        else:
            input_ids = _current_input_ids
            attention_mask = _current_attention_mask
    else:
        # Fallback
        input_ids = torch.zeros((batch_size, 32), dtype=torch.long, device=device)
        attention_mask = torch.ones((batch_size, 32), dtype=torch.long, device=device)
    
    return model(inputs, input_ids, attention_mask)


def compute_relig_attribution(image, input_ids, attention_mask, target_class, 
                               use_noise_tunnel=True):
    """
    Compute RE-LIG (Robust Layer Integrated Gradients) attribution.
    
    This function implements Stochastic Smoothing via Noise Tunneling
    for more stable and interpretable saliency maps.
    
    Args:
        image: Input image tensor [1, 3, 448, 448]
        input_ids: Text token IDs [1, seq_len]
        attention_mask: Text attention mask [1, seq_len]
        target_class: Target class index for attribution
        use_noise_tunnel: Whether to apply Stochastic Smoothing
        
    Returns:
        Attribution tensor at patch level
    """
    global _current_input_ids, _current_attention_mask
    
    model.eval()
    
    # Store text inputs for wrapper
    _current_input_ids = input_ids.to(device)
    _current_attention_mask = attention_mask.to(device)
    
    # Prepare image
    image = image.to(device).float()
    if not image.requires_grad:
        image = image.detach().requires_grad_(True)
    
    # Create baseline (black image)
    baseline = torch.zeros_like(image).to(device)
    
    # Initialize Layer Integrated Gradients
    lig = LayerIntegratedGradients(image_forward_for_lig, vis_layer)
    
    try:
        if use_noise_tunnel and RELIG_CONFIG['nt_samples'] > 1:
            # ================================================================
            # RE-LIG: STOCHASTIC SMOOTHING VIA NOISE TUNNELING
            # ================================================================
            # This is the core of RE-LIG - wrapping LayerIG with NoiseTunnel
            # to reduce noise and produce more stable attributions
            nt = NoiseTunnel(lig)
            
            print(f"   🔬 Using Noise Tunnel (samples={RELIG_CONFIG['nt_samples']}, stdev={RELIG_CONFIG['stdev']})")
            
            attributions = nt.attribute(
                inputs=image,
                baselines=baseline,
                target=target_class,
                n_steps=RELIG_CONFIG['n_steps'],
                nt_type=RELIG_CONFIG['nt_type'],
                nt_samples=RELIG_CONFIG['nt_samples'],
                stdevs=RELIG_CONFIG['stdev'],
                attribute_to_layer_input=False,
                internal_batch_size=1  # Memory efficient
            )
        else:
            # Standard Layer IG (without Noise Tunneling)
            print("   📊 Using Standard Layer IG (no Noise Tunnel)")
            attributions = lig.attribute(
                inputs=image,
                baselines=baseline,
                target=target_class,
                n_steps=RELIG_CONFIG['n_steps'],
                attribute_to_layer_input=False
            )
        
        return attributions.detach()
        
    except Exception as e:
        print(f"   ❌ Attribution error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # Cleanup globals
        _current_input_ids = None
        _current_attention_mask = None


# --- B) TEXT GRADIENT ATTRIBUTION ---
def get_text_attribution(model, img, txt, mask, target_idx):
    model.zero_grad()
    embedding_layer = model.bert.embeddings.word_embeddings
    vocab_size = embedding_layer.num_embeddings
    txt_clamped = torch.clamp(txt, min=0, max=vocab_size - 1)
    pad_id = dataset.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0
    baseline = torch.full_like(txt_clamped, fill_value=pad_id)

    def text_forward(input_ids, image, attention_mask):
        return model(image, input_ids, attention_mask)

    lig = LayerIntegratedGradients(text_forward, embedding_layer)
    attributions = lig.attribute(
        inputs=txt_clamped,
        baselines=baseline,
        additional_forward_args=(img, mask),
        target=target_idx,
        n_steps=RELIG_CONFIG['n_steps']
    )
    attr = attributions.sum(dim=-1).abs()
    return attr.squeeze().detach().cpu().numpy()


def merge_subword_tokens(tokens, scores):
    """Merge BERT WordPiece subword tokens (## prefix) into whole words."""
    merged_tokens, merged_scores = [], []
    for token, score in zip(tokens, scores):
        if token.startswith('##') and merged_tokens:
            merged_tokens[-1] += token[2:]
            merged_scores[-1] = max(merged_scores[-1], score)
        else:
            merged_tokens.append(token)
            merged_scores.append(score)
    return merged_tokens, np.array(merged_scores)


def process_heatmap_focused(attr, image_size=448):
    """
    Process attribution to create FOCUSED heatmap.
    
    Optimized for RE-LIG output:
    - Lower Gaussian sigma for sharper edges
    - Gamma correction to concentrate on peaks
    - Percentile-based normalization
    """
    # Convert to numpy if tensor
    if isinstance(attr, torch.Tensor):
        attr = attr.detach().cpu().numpy()
    
    # Handle CLS token if present (ViT outputs 197 patches = 1 CLS + 196 patches)
    if attr.shape[0] == 197:
        attr = attr[1:]
    
    side = int(np.sqrt(attr.shape[0]))
    attr_2d = attr.reshape(side, side)
    
    # Take absolute values
    attr_abs = np.abs(attr_2d)
    
    # Resize to image size with bicubic interpolation
    heatmap = resize(attr_abs, (image_size, image_size), order=3, mode='constant')
    
    # Mild Gaussian smoothing (sigma=4 for RE-LIG, keeps important details)
    heatmap = gaussian_filter(heatmap, sigma=4)
    
    # Percentile normalization to enhance contrast
    v_min, v_max = np.percentile(heatmap, (5, 98))
    heatmap = np.clip((heatmap - v_min) / (v_max - v_min + 1e-8), 0, 1)
    
    # Gamma correction to concentrate attention on peaks
    # gamma > 1 makes low values darker, high values stay bright
    heatmap = np.power(heatmap, 1.8)
    
    # Re-normalize after gamma
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    return heatmap


# =============================================================================
# 4. Q1 PUBLICATION VISUALIZATION
# =============================================================================
def create_q1_visualization(image, heatmap, tokens, token_scores, 
                            q_text, pred, truth, conf, save_path):
    """
    Create Q1 SCI journal quality visualization with:
    - Subplot labels (a), (b), (c), (d)
    - Numerical attribution scores below text tokens
    - Professional colorbar
    - High resolution output (300 DPI)
    - RE-LIG method indicator
    """
    
    # --- FIGURE SETUP ---
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.2, 0.5], 
                           wspace=0.15, hspace=0.35,
                           left=0.05, right=0.95, top=0.88, bottom=0.08)
    
    # Prepare image for display
    img_np = image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    
    # --- (a) Original Image ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_np)
    ax1.axis('off')
    ax1.set_title("(a) Original Image", fontsize=13, fontweight='bold', pad=10)
    
    # --- (b) Saliency Map (RE-LIG) ---
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(heatmap, cmap='inferno', vmin=0, vmax=1)
    ax2.axis('off')
    ax2.set_title("(b) RE-LIG Saliency Map", fontsize=13, fontweight='bold', pad=10)
    
    # Professional colorbar
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, aspect=25)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    # --- (c) Overlay ---
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(img_np)
    ax3.imshow(heatmap, cmap='inferno', alpha=0.5)
    ax3.axis('off')
    ax3.set_title("(c) Overlay", fontsize=13, fontweight='bold', pad=10)
    
    # Add prediction info as text box
    textstr = f"Pred: {pred}\nConf: {conf:.2f}"
    props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray')
    ax3.text(0.98, 0.02, textstr, transform=ax3.transAxes, fontsize=10,
             verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    # --- (d) Question Token Importance ---
    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title("(d) Question Token Importance", fontsize=15, fontweight='bold', pad=15)
    
    # Filter and normalize token scores
    clean_tokens = []
    clean_scores = []
    raw_scores = []

    if len(token_scores) == len(tokens):
        # First merge subword tokens, then filter special tokens
        merged_tokens, merged_scores = merge_subword_tokens(tokens, token_scores)
        max_score = np.max(merged_scores) + 1e-9

        for t, s in zip(merged_tokens, merged_scores):
            if t not in ['[CLS]', '[SEP]', '[PAD]']:
                clean_tokens.append(t)
                clean_scores.append(s / max_score)
                raw_scores.append(s)
    
    # Normalize raw scores 0-1 for display
    if raw_scores:
        raw_min, raw_max = min(raw_scores), max(raw_scores)
        display_scores = [(s - raw_min) / (raw_max - raw_min + 1e-9) for s in raw_scores]
    else:
        display_scores = []
    
    # Draw token boxes with scores below (FIXED: proper row spacing)
    x_start = 0.03
    y_base = 0.60  # Base Y for token boxes
    box_height = 0.22
    score_offset = 0.28  # Distance from box bottom to score
    row_height = 0.42  # Total height for one row (box + score + gap)
    
    for idx, (token, norm_score, disp_score) in enumerate(zip(clean_tokens, clean_scores, display_scores)):
        box_w = max(0.07, len(token) * 0.016 + 0.02)
        
        # Check if we need to wrap to next line
        if x_start + box_w > 0.97:
            x_start = 0.03
            y_base -= row_height
            
            # Stop if we've run out of vertical space
            if y_base - score_offset < 0.02:
                break
        
        y_box = y_base
        y_score = y_base - score_offset
        
        intensity = 0.2 + (norm_score * 0.8)
        color = plt.cm.Greens(intensity)
        
        fancy_box = FancyBboxPatch(
            (x_start, y_box), box_w, box_height,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor=color, edgecolor='#2d5016', linewidth=1.5
        )
        ax4.add_patch(fancy_box)
        
        text_color = 'white' if intensity > 0.55 else 'black'
        ax4.text(x_start + box_w/2, y_box + box_height/2, token,
                 ha='center', va='center', fontsize=17, fontweight='bold',
                 color=text_color, family='monospace')

        ax4.text(x_start + box_w/2, y_score, f"{disp_score:.2f}",
                 ha='center', va='center', fontsize=15, fontweight='normal',
                 color='#333333', family='sans-serif')
        
        x_start += box_w + 0.015
    
    # --- MAIN TITLE ---
    if len(q_text) > 60:
        q_display = q_text[:57] + "..."
    else:
        q_display = q_text
    
    fig.suptitle(f"Q: {q_display}  |  Truth: {truth}", 
                 fontsize=14, fontweight='bold', y=0.96)
    
    # --- METHOD ANNOTATION ---
    fig.text(0.99, 0.01, f"Method: RE-LIG (nt_samples={RELIG_CONFIG['nt_samples']}, stdev={RELIG_CONFIG['stdev']})", 
             fontsize=8, ha='right', va='bottom', alpha=0.6, style='italic')
    
    # --- SAVE ---
    plt.savefig(save_path, bbox_inches='tight', dpi=300, 
                facecolor='white', edgecolor='none')
    plt.close()


# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    num_samples = args.num_samples
    use_noise_tunnel = True
    
    print(f"\n{'='*60}")
    print(f"🚀 RE-LIG Visualization Generator")
    print(f"{'='*60}")
    print(f"  Samples: {num_samples}")
    print(f"  Noise Tunnel: {'ENABLED' if use_noise_tunnel else 'DISABLED'}")
    print(f"  Config: {RELIG_CONFIG}")
    print(f"{'='*60}\n")
    
    random.seed(args.seed)

    # YENI: --failure_mode ile yanlis siniflandirilan ornekler secilir
    if args.failure_mode:
        print("\n--- FAILURE MODE: misclassified samples ---")
        failure_indices = []
        for scan_idx in range(len(dataset)):
            if len(failure_indices) >= num_samples:
                break
            try:
                it = dataset[scan_idx]
                ig = it['image'].to(device).unsqueeze(0)
                it2 = it['input_ids'].to(device).unsqueeze(0)
                im2 = it['attention_mask'].to(device).unsqueeze(0)
                with torch.no_grad():
                    op  = model(ig, it2, im2)
                    pid = op.argmax(1).item()
                if pid != it['answer'].item():
                    failure_indices.append(scan_idx)
            except Exception:
                continue
        indices = failure_indices
        output_dir = output_dir + "_failures"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Found {len(indices)} misclassified samples")
    else:
        indices = random.sample(range(len(dataset)), min(len(dataset), num_samples))
    
    for i, idx in enumerate(tqdm(indices, desc="Generating Visualizations")):
        try:
            item = dataset[idx]
            img = item['image'].to(device).unsqueeze(0).requires_grad_()
            txt = item['input_ids'].to(device).unsqueeze(0)
            mask = item['attention_mask'].to(device).unsqueeze(0)
            
            # Get prediction
            with torch.no_grad():
                out = model(img, txt, mask)
                probs = torch.nn.functional.softmax(out, dim=1)
                pred_id = out.argmax(1).item()
                conf = probs[0, pred_id].item()
            
            pred_ans = idx2answer.get(pred_id, "Unknown")
            
            print(f"\n[{i+1}/{num_samples}] Q: {item['question_text'][:50]}...")
            print(f"   Pred: {pred_ans} | Truth: {item['answer_text']} | Conf: {conf:.3f}")
            
            # ================================================================
            # RE-LIG IMAGE ATTRIBUTION (with Noise Tunneling)
            # ================================================================
            attrs = compute_relig_attribution(
                img, txt, mask, pred_id, 
                use_noise_tunnel=use_noise_tunnel
            )
            
            if attrs is None:
                print(f"   ⚠️ Skipping due to attribution error")
                continue
            
            # Process attribution to heatmap
            attr_summed = attrs.sum(dim=-1).squeeze()  # [197]
            heatmap = process_heatmap_focused(attr_summed)
            
            # Text attribution
            text_scores = get_text_attribution(model, img, txt, mask, pred_id)
            tokens = dataset.tokenizer.convert_ids_to_tokens(txt.squeeze().tolist())
            
            # Generate visualization
            save_path = os.path.join(output_dir, f"q1_relig_{i+1:02d}.png")
            create_q1_visualization(
                img, heatmap, tokens, text_scores,
                item['question_text'], pred_ans, item['answer_text'], conf, save_path
            )
            print(f"   ✅ Saved: {save_path}")
            
            # Cleanup
            model.zero_grad()
            
        except Exception as e:
            print(f"   ❌ Error sample {i}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"🎉 Done! RE-LIG visualizations saved to '{output_dir}/'")
    print(f"{'='*60}")
