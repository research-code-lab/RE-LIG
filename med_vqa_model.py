"""
MedVQAModel - Medical Visual Question Answering Model
======================================================
Architecture:
- Vision Encoder: PubMedCLIP (ViT-Base-Patch32)
- Text Encoder: BioLinkBERT (Medical + Link Prediction Pre-trained)
- Fusion: Question-Guided Co-Attention (Novel - Adaptive attention based on question type)
- Classifier: MLP Head

Novel Contribution:
- Question-Guided Modulation: Dynamically adjusts co-attention weights based on
  whether the question is closed-ended (yes/no) or open-ended (requires localization).

Supports:
- inputs_embeds for Layer Integrated Gradients attribution
- Dynamic positional embedding interpolation for high-resolution images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer
import math
from typing import Optional


class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, context, mask=None):
        b, n, _, h = *x.shape, self.heads
        q = self.to_q(x).reshape(b, n, h, -1).permute(0, 2, 1, 3)
        k = self.to_k(context).reshape(b, context.shape[1], h, -1).permute(0, 2, 1, 3)
        v = self.to_v(context).reshape(b, context.shape[1], h, -1).permute(0, 2, 1, 3)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            if mask.dim() == 2: mask = mask.unsqueeze(1).unsqueeze(1)
            dots = dots.masked_fill(mask == 0, -1e4)

        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(b, n, -1)
        return self.to_out(out)


class CoAttentionLayer(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        self.v2t = CrossAttention(dim, heads, dim_head, dropout)
        self.t2v = CrossAttention(dim, heads, dim_head, dropout)
        self.norm_v = nn.LayerNorm(dim)
        self.norm_t = nn.LayerNorm(dim)

    def forward(self, v, t, t_mask=None):
        # V -> T
        v = self.norm_v(v + self.v2t(v, t, t_mask))
        # T -> V
        t = self.norm_t(t + self.t2v(t, v))
        return v, t


class CoAttention(nn.Module):
    def __init__(self, dim, num_layers=2, heads=12, dim_head=64, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([CoAttentionLayer(dim, heads, dim_head, dropout) for _ in range(num_layers)])
        
    def forward(self, v, t, t_mask=None):
        for layer in self.layers:
            v, t = layer(v, t, t_mask)
        return v, t


# =============================================================================
# QUESTION-GUIDED CO-ATTENTION (Novel Architecture)
# =============================================================================
class QuestionTypeClassifier(nn.Module):
    """
    Lightweight classifier to predict question type (closed vs open-ended).
    Used to modulate co-attention behavior.
    """
    def __init__(self, dim=768, hidden_dim=256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2)  # 2 classes: closed, open
        )
    
    def forward(self, text_cls: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_cls: [CLS] token representation [B, dim]
        Returns:
            Question type logits [B, 2]
        """
        return self.classifier(text_cls)


class QuestionGuidedCoAttention(nn.Module):
    """
    Question-Guided Co-Attention for Medical VQA.
    
    Novel Contribution:
    - Dynamically modulates attention based on inferred question type
    - Closed-ended questions (yes/no): Emphasizes global visual features
    - Open-ended questions: Emphasizes local region-based attention
    
    Architecture:
        Text [CLS] → Question Type Predictor → Modulation Weights
                                                    ↓
        Vision ←→ Text (Bidirectional Co-Attention) → Weighted Fusion
    
    Reference: Novel architecture for Med-VQA (2024)
    """
    
    def __init__(self, dim=768, num_layers=2, heads=12, dropout=0.1):
        super().__init__()
        
        # Question type classifier for modulation
        self.question_classifier = QuestionTypeClassifier(dim)
        
        # Standard co-attention layers
        self.co_attention = CoAttention(dim, num_layers, heads, dropout=dropout)
        
        # Modulation networks
        # For closed-ended: emphasize global pooling
        self.global_pool_weight = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        
        # For open-ended: emphasize local attention
        self.local_attn_weight = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        
        # Learnable temperature for attention sharpness
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Layer norms
        self.norm_v = nn.LayerNorm(dim)
        self.norm_t = nn.LayerNorm(dim)
        
    def forward(self, v: torch.Tensor, t: torch.Tensor, t_mask: torch.Tensor = None):
        """
        Forward pass through question-guided co-attention.
        
        Args:
            v: Vision features [B, N_patches, dim]
            t: Text features [B, seq_len, dim]
            t_mask: Text attention mask [B, seq_len]
            
        Returns:
            Fused vision and text features
        """
        batch_size = v.shape[0]
        
        # === Step 1: Predict question type from [CLS] token ===
        text_cls = t[:, 0, :]  # [B, dim]
        qtype_logits = self.question_classifier(text_cls)  # [B, 2]
        qtype_probs = F.softmax(qtype_logits / self.temperature, dim=-1)  # [B, 2]
        
        # closed_prob: probability of closed-ended (global-focused)
        # open_prob: probability of open-ended (local-focused)
        closed_prob = qtype_probs[:, 0].unsqueeze(-1)  # [B, 1]
        open_prob = qtype_probs[:, 1].unsqueeze(-1)  # [B, 1]
        
        # === Step 2: Standard Co-Attention ===
        v_attn, t_attn = self.co_attention(v, t, t_mask)
        
        # === Step 3: Compute global and local visual representations ===
        # Global: average pooled, then broadcast back
        v_global = v_attn.mean(dim=1, keepdim=True).expand_as(v_attn)  # [B, N, dim]
        
        # Local: the attention-weighted features as-is
        v_local = v_attn  # [B, N, dim]
        
        # === Step 4: Question-Guided Modulation ===
        # Compute modulation weights
        global_weight = self.global_pool_weight(text_cls).unsqueeze(1)  # [B, 1, dim]
        local_weight = self.local_attn_weight(text_cls).unsqueeze(1)  # [B, 1, dim]
        
        # Blend based on question type
        # closed-ended → more global, open-ended → more local
        v_modulated = (
            closed_prob.unsqueeze(-1) * global_weight * v_global +
            open_prob.unsqueeze(-1) * local_weight * v_local
        )
        
        # === Step 5: Residual and normalization ===
        v_out = self.norm_v(v_modulated + v)
        t_out = self.norm_t(t_attn + t)
        
        return v_out, t_out


# =============================================================================
# MED-VQA MODEL (Updated with Question-Guided Co-Attention)
# =============================================================================
class MedVQAModel(nn.Module):
    """
    Medical Visual Question Answering Model
    
    This model supports the `inputs_embeds` parameter in forward() to enable
    Layer Integrated Gradients (Layer IG) for text attribution with Captum.
    
    Novel Features:
    - Question-Guided Co-Attention: Adapts fusion based on question type
    - PubMedCLIP: Medical domain-specific visual features
    - BioLinkBERT: Medical + scientific text understanding
    
    Args:
        num_classes: Number of answer classes
        bert_model: HuggingFace model name for text encoder
        vit_model: HuggingFace model name for vision encoder
        image_size: Input image resolution (default 448 for high-res medical imaging)
    """
    
    def __init__(self, num_classes=235, bert_model='michiyasunaga/BioLinkBERT-base', 
                 vit_model='flaviagiammarino/pubmed-clip-vit-base-patch32', image_size=448):  
        super().__init__()
        
        # --- SOTA IMPROVEMENT 1: PubMedCLIP for Medical Domain-Specific Features ---
        print(f"Loading Medical Vision Encoder (PubMedCLIP): {vit_model}...")
        try:
            self.vit = AutoModel.from_pretrained(vit_model)
            # Check if it's a CLIP model and extract vision component
            if hasattr(self.vit, 'vision_model'):
                print("ℹ️ Detected CLIP model, extracting vision encoder...")
                self.vit = self.vit.vision_model
                
            # Dynamically determine dimension from config
            if hasattr(self.vit, 'config') and hasattr(self.vit.config, 'hidden_size'):
                self.vit_dim = self.vit.config.hidden_size
            else:
                self.vit_dim = 768 # Default fallback for ViT-Base
                
            print(f"✓ Vision Encoder loaded (dim={self.vit_dim})")
        except Exception as e:
            print(f"⚠ PubMedCLIP loading failed: {e}")
            print("Falling back to standard ViT-Base-224...")
            self.vit = AutoModel.from_pretrained('google/vit-base-patch16-224')
            self.vit_dim = 768
            
        # Text Encoder (BioLinkBERT - Better for Medical QA)
        print(f"Loading BioLinkBERT: {bert_model}...")
        try:
            self.bert = AutoModel.from_pretrained(bert_model)
            print("✓ BioLinkBERT loaded successfully")
        except:
            print("⚠ BioLinkBERT loading failed. Using bert-base-uncased...")
            self.bert = AutoModel.from_pretrained('bert-base-uncased')
            
        self.bert_output_dim = 768
        
        # --- DIMENSION PROJECTION: Align PubMedCLIP (512 or 768) with BioBERT (768) ---
        self.visual_projection = nn.Linear(self.vit_dim, 768)
        self.text_projection = nn.Linear(768, 768)
        
        # --- QUESTION-GUIDED CO-ATTENTION (Novel Architecture) ---
        print("Initializing Question-Guided Co-Attention...")
        self.fusion = QuestionGuidedCoAttention(
            dim=768, num_layers=2, heads=12, dropout=0.1
        )
        print("✓ Question-Guided Co-Attention initialized")
        
        # Classifier - Enhanced with residual connection
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
        self.num_classes = num_classes
        self.image_size = image_size
        
        # Ensure embeddings are resized to match image_size immediately
        self.interpolate_pos_encoding((1, 3, image_size, image_size), next(self.parameters()).device)

    def interpolate_pos_encoding(self, input_shape, device):
        """
        Dynamic interpolation of positional embeddings for high-resolution images.
        Handles both ViT and CLIPVisionModel architectures.
        """
        # Get current positional embeddings
        if hasattr(self.vit, 'embeddings'):
            pos_embed_layer = self.vit.embeddings.position_embedding
            # CLIP uses 'position_ids' buffer, ViT might not
            has_pos_ids = hasattr(self.vit.embeddings, 'position_ids')
        else:
            return # Should not happen for standard HF models

        # Current parameters
        old_pos_embed = pos_embed_layer.weight.data
        old_num_tokens, hidden_dim = old_pos_embed.shape
        
        # Robust CLS token detection
        grid_size_cls = int(math.sqrt(old_num_tokens - 1))
        grid_size_no_cls = int(math.sqrt(old_num_tokens))
        
        if (grid_size_cls * grid_size_cls) == (old_num_tokens - 1):
            has_cls_token = True
            old_grid_size = grid_size_cls
        elif (grid_size_no_cls * grid_size_no_cls) == old_num_tokens:
            has_cls_token = False
            old_grid_size = grid_size_no_cls
        else:
            has_cls_token = True
            old_grid_size = int(math.sqrt(old_num_tokens - 1))
            
        # Determine patch size from config or infer
        patch_size = 32 # Default
        if hasattr(self.vit.config, 'patch_size'):
            patch_size = self.vit.config.patch_size
            
        new_h, new_w = input_shape[2], input_shape[3]
        new_grid_h = new_h // patch_size
        new_grid_w = new_w // patch_size
        new_num_patches = new_grid_h * new_grid_w
        
        # Check if interpolation is actually needed
        if (has_cls_token and (new_num_patches + 1) == old_num_tokens) or \
           (not has_cls_token and new_num_patches == old_num_tokens):
            return 

        print(f"🔄 Interpolating Positional Embeddings: {old_num_tokens} -> {new_num_patches + (1 if has_cls_token else 0)}")
        
        if has_cls_token:
            cls_pos_embed = old_pos_embed[0:1, :]
            patch_pos_embed = old_pos_embed[1:, :]
        else:
            cls_pos_embed = None
            patch_pos_embed = old_pos_embed

        # Reshape to square grid for interpolation
        patch_pos_embed = patch_pos_embed.reshape(1, old_grid_size, old_grid_size, hidden_dim).permute(0, 3, 1, 2)
        
        new_patch_pos_embed = F.interpolate(
            patch_pos_embed, 
            size=(new_grid_h, new_grid_w), 
            mode='bicubic', 
            align_corners=False
        )
        
        new_patch_pos_embed = new_patch_pos_embed.permute(0, 2, 3, 1).reshape(-1, hidden_dim)
        
        if has_cls_token:
            new_pos_embed = torch.cat([cls_pos_embed, new_patch_pos_embed], dim=0)
        else:
            new_pos_embed = new_patch_pos_embed
            
        # Update model embeddings
        self.vit.embeddings.position_embedding = nn.Embedding(
            new_pos_embed.shape[0], 
            hidden_dim, 
            _weight=new_pos_embed
        ).to(device)
        
        # Update position_ids if they exist
        if has_pos_ids:
            self.vit.embeddings.position_ids = torch.arange(
                new_pos_embed.shape[0]
            ).expand((1, -1)).to(device)
            
        # Update config to prevent errors in future calls
        self.vit.config.image_size = self.image_size
        
        # CRITICAL FIX: Update CLIPVisionEmbeddings attributes directly
        if hasattr(self.vit, 'embeddings'):
            self.vit.embeddings.image_size = self.image_size
            self.vit.embeddings.num_patches = new_num_patches
            self.vit.embeddings.num_positions = new_num_patches + 1

    def forward(
        self, 
        image: torch.Tensor, 
        input_ids: Optional[torch.Tensor] = None, 
        attention_mask: Optional[torch.Tensor] = None, 
        inputs_embeds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for the MedVQA model.
        
        Args:
            image: Input image tensor [B, 3, H, W]
            input_ids: Token IDs tensor [B, seq_len] (optional if inputs_embeds provided)
            attention_mask: Attention mask tensor [B, seq_len]
            inputs_embeds: Pre-computed embeddings [B, seq_len, hidden_dim]
                          Used by Captum LayerIntegratedGradients for text attribution.
        
        Returns:
            Logits tensor [B, num_classes]
        """
        # --- Visual Encoding (PubMedCLIP) ---
        
        # Dynamic Positional Interpolation
        self.interpolate_pos_encoding(image.shape, image.device)
            
        vit_output = self.vit(pixel_values=image)
        vision_embeds = vit_output.last_hidden_state  # [B, N_patches, 512 or 768]
        
        # Project to 768-dim to match BioBERT
        vision_embeds = self.visual_projection(vision_embeds)  # [B, N_patches, 768]
        
        # --- Text Encoding (BioBERT) ---
        # Support for inputs_embeds enables Layer Integrated Gradients
        if inputs_embeds is not None:
            # Use pre-computed embeddings (for Captum attribution)
            text_out = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        else:
            # Standard path: compute embeddings from input_ids
            input_ids = torch.clamp(input_ids, max=self.bert.config.vocab_size-1)
            text_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            
        text_embeds = self.text_projection(text_out.last_hidden_state)
        
        if attention_mask is not None: 
            attention_mask = attention_mask.to(dtype=vision_embeds.dtype)
            
        # --- Fusion (Question-Guided Co-Attention) ---
        vision_fused, text_fused = self.fusion(vision_embeds, text_embeds, attention_mask)
        
        # --- Pooling ---
        vision_pooled = vision_fused.mean(dim=1)  # Global Average Pooling
        text_pooled = text_fused[:, 0, :]  # [CLS] token
        
        combined = torch.cat([vision_pooled, text_pooled], dim=1)
        
        # --- Classification ---
        return self.classifier(combined)


class InterpretableMedVQA(nn.Module):
    """
    Wrapper class for interpretability utilities.
    Provides convenient methods for attribution analysis.
    """
    
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        self.tokenizer = AutoTokenizer.from_pretrained('michiyasunaga/BioLinkBERT-base')
        if hasattr(self.model, 'bert'): 
            self.model.config = self.model.bert.config

    def forward(self, image, text_or_tokens, device):
        if isinstance(text_or_tokens, str):
            tokens = self.tokenizer(text_or_tokens, padding='max_length', truncation=True, 
                                   max_length=32, return_tensors="pt")
            input_ids = tokens['input_ids'].to(device)
            mask = tokens['attention_mask'].to(device)
        elif isinstance(text_or_tokens, dict):
            input_ids = text_or_tokens['input_ids'].to(device)
            mask = text_or_tokens['attention_mask'].to(device)
        else:
            input_ids = text_or_tokens.to(device)
            mask = torch.ones_like(input_ids).to(device)
        return self.model(image, input_ids=input_ids, attention_mask=mask), None
    
    def get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get text embeddings for Layer Integrated Gradients.
        
        Args:
            input_ids: Token IDs tensor [B, seq_len]
            
        Returns:
            Embeddings tensor [B, seq_len, hidden_dim]
        """
        vocab_size = self.model.bert.embeddings.word_embeddings.num_embeddings
        input_ids_clamped = torch.clamp(input_ids, min=0, max=vocab_size - 1)
        return self.model.bert.embeddings.word_embeddings(input_ids_clamped)
