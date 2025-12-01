import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Helper: extract 2D patches from image
# ============================================================

def extract_patches(img, patch_size):
    """
    img: [B, C, H, W]
    returns: [B, N, patch_dim] where patch_dim = C * patch_size * patch_size
    """
    B, C, H, W = img.shape
    assert H % patch_size == 0 and W % patch_size == 0
    patches = img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # [B, C, H//P, W//P, P, P]
    patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
    # [B, C, N, P, P]
    patches = patches.permute(0, 2, 1, 3, 4)  # [B, N, C, P, P]
    patches = patches.reshape(B, -1, C * patch_size * patch_size)  # [B, N, patch_dim]
    return patches


# ============================================================
# Multi-Head Self-Attention (with optional mask)
# ============================================================

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        """
        x: [B, N, D]
        attn_mask: [1, 1, N, N] or broadcastable, with 0 for allowed, -inf for disallowed
        """
        B, N, D = x.shape

        qkv = self.qkv(x)  # [B, N, 3D]
        q, k, v = qkv.chunk(3, dim=-1)  # each [B, N, D]

        def reshape_heads(t):
            return t.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q = reshape_heads(q)  # [B, H, N, Hd]
        k = reshape_heads(k)
        v = reshape_heads(v)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # attn_scores: [B, H, N, N]

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask  # broadcast add

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # [B, H, N, Hd]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, N, D)
        out = self.out_proj(attn_output)  # [B, N, D]
        return out


# ============================================================
# Transformer Encoder Block (ViT / CLIP style)
# ============================================================

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)

        hidden_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None):
        # x: [B, N, D]
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================
# ViT-like Image Encoder (CLS token)
# ============================================================

class ViT(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        d_model=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()

        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.image_size = image_size
        self.patch_size = patch_size
        self.d_model = d_model

        num_patches = (image_size // patch_size) ** 2
        self.num_patches = num_patches

        patch_dim = in_channels * patch_size * patch_size  # e.g. 3*16*16=768

        # Patch embedding: [patch_dim -> d_model]
        self.patch_embedding = nn.Linear(patch_dim, d_model)

        # CLS token (global token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Positional embedding: one for each patch + CLS
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))

        self.pos_drop = nn.Dropout(dropout)

        # Stack of Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(d_model)

        # Simple init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, img):
        """
        img: [B, 3, H, W], here H=W=image_size
        returns: CLS embedding [B, d_model]
        """
        B, C, H, W = img.shape
        assert H == self.image_size and W == self.image_size, "Resize image first"

        # 1) Extract patches: [B, N, patch_dim]
        x = extract_patches(img, self.patch_size)

        # 2) Patch embedding: [B, N, d_model]
        x = self.patch_embedding(x)

        # 3) Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        x = torch.cat((cls_tokens, x), dim=1)          # [B, N+1, d_model]

        # 4) Add positional embedding
        x = x + self.pos_embed                        # [B, N+1, d_model]
        x = self.pos_drop(x)

        # 5) Transformer blocks (no mask for image)
        for blk in self.blocks:
            x = blk(x)                                # [B, N+1, d_model]

        # 6) Final norm
        x = self.norm(x)                              # [B, N+1, d_model]

        # 7) Take CLS token as global embedding
        cls_embedding = x[:, 0]                       # [B, d_model]
        return cls_embedding


# ============================================================
# Text Encoder (CLIP-style, EOS token as global)
# ============================================================

class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size=49408,
        max_len=77,
        d_model=512,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding   = nn.Embedding(max_len, d_model)   # 1D positions

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.ln_final = nn.LayerNorm(d_model)

        # Projection into joint space (512x512)
        self.text_projection = nn.Parameter(torch.empty(d_model, d_model))
        nn.init.normal_(self.text_projection, std=d_model ** -0.5)

    def build_causal_mask(self, seq_len, device):
        # [1, 1, N, N], upper triangle masked with -inf
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)  # positions j > i get -inf
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]

    def forward(self, token_ids):
        """
        token_ids: [B, N] of int64, padded/truncated to <= max_len
        Returns: [B, d_model] text embeddings (CLIP-style)
        NOTE: for simplicity, assumes EOS is at position N-1.
        """
        B, N = token_ids.shape
        assert N <= self.max_len

        device = token_ids.device
        positions = torch.arange(N, device=device).unsqueeze(0)  # [1, N]

        # 1) Embed tokens + positions
        x = self.token_embedding(token_ids) + self.pos_embedding(positions)  # [B, N, D]

        # 2) Causal mask
        attn_mask = self.build_causal_mask(N, device)  # [1, 1, N, N]

        # 3) Transformer blocks
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)  # [B, N, D]

        # 4) Final layer norm
        x = self.ln_final(x)  # [B, N, D]

        # 5) Take last token as EOS embedding (simplified)
        eos_embedding = x[:, -1, :]  # [B, D]

        # 6) Project to joint space
        text_emb = eos_embedding @ self.text_projection  # [B, D]

        # 7) Normalize
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        return text_emb


# ============================================================
# CLIP-like wrapper
# ============================================================

class CLIPModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Vision: 768-dim CLS -> project to 512
        self.visual = ViT(
            image_size=224,
            patch_size=16,
            in_channels=3,
            d_model=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            dropout=0.0,
        )
        self.visual_projection = nn.Linear(768, 512)

        # Text: already 512-dim; projection inside TextEncoder is 512x512
        self.text_encoder = TextEncoder(
            vocab_size=49408,
            max_len=77,
            d_model=512,
            depth=12,
            num_heads=8,
            mlp_ratio=4.0,
            dropout=0.0,
        )

    @torch.no_grad()
    def encode_image(self, image):
        """
        image: [B, 3, 224, 224]
        returns: [B, 512] normalized image embeddings
        """
        cls_emb = self.visual(image)                    # [B, 768]
        img_emb = self.visual_projection(cls_emb)       # [B, 512]
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        return img_emb

    @torch.no_grad()
    def encode_text(self, token_ids):
        """
        token_ids: [B, N]
        returns: [B, 512] normalized text embeddings
        """
        text_emb = self.text_encoder(token_ids)         # [B, 512], already normalized there
        return text_emb


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(0)

    model = CLIPModel()

    # Dummy image (like CLIP's input after preprocessing)
    image = torch.randn(1, 3, 224, 224)

    # Dummy token sequence (e.g. "a cat on a mat" tokenized to length 10, padded to 16)
    B = 1
    seq_len = 16
    token_ids = torch.randint(low=0, high=49408, size=(B, seq_len), dtype=torch.long)

    img_emb = model.encode_image(image)   # [1, 512]
    txt_emb = model.encode_text(token_ids)  # [1, 512]

    print("Image embedding shape:", img_emb.shape)
    print("Text embedding shape:", txt_emb.shape)

    # Cosine similarity (because both are L2-normalized)
    sim = (img_emb @ txt_emb.T).item()
    print("Cosine similarity:", sim)
