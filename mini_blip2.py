import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
#  Tiny Vision Encoder (ViT-ish)
# =========================

class PatchEmbed(nn.Module):
    """
    Simple image -> patch embeddings using a Conv2d.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.proj(x)           # (B, D, H/ps, W/ps)
        x = x.flatten(2)           # (B, D, N)
        x = x.transpose(1, 2)      # (B, N, D)
        return x


class TinyViTEncoder(nn.Module):
    """
    Minimal ViT-style encoder: patch embedding + a few TransformerEncoder layers.
    """
    def __init__(self, img_size=224, patch_size=16, embed_dim=256, depth=2, nhead=4):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        """
        x: (B, 3, H, W) -> vision_tokens: (B, N_patches, D)
        """
        x = self.patch_embed(x)      # (B, N, D)
        x = x + self.pos_embed       # add positional encodings
        x = self.encoder(x)          # (B, N, D)
        return x


# =========================
#  Q-Former Lite
# =========================

class QFormerLite(nn.Module):
    """
    Mini Q-Former:
      - Learnable query tokens
      - Cross-attention over frozen vision tokens
      - Stacked a few times
    """
    def __init__(self, vision_dim=256, q_dim=256, num_queries=16, num_layers=2, nhead=4):
        super().__init__()
        self.num_queries = num_queries
        self.q_embed = nn.Parameter(torch.randn(1, num_queries, q_dim) * 0.02)

        # Project vision tokens into q_dim if needed
        self.vision_proj = nn.Linear(vision_dim, q_dim) if vision_dim != q_dim else nn.Identity()

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": nn.MultiheadAttention(q_dim, nhead, batch_first=True),
                "ln1": nn.LayerNorm(q_dim),
                "mlp": nn.Sequential(
                    nn.Linear(q_dim, q_dim * 4),
                    nn.GELU(),
                    nn.Linear(q_dim * 4, q_dim),
                ),
                "ln2": nn.LayerNorm(q_dim),
            })
            for _ in range(num_layers)
        ])

    def forward(self, vision_tokens):
        """
        vision_tokens: (B, N_v, D_v)
        returns: q_tokens: (B, num_queries, q_dim)
        """
        B = vision_tokens.size(0)
        v = self.vision_proj(vision_tokens)  # (B, N_v, q_dim)

        # Expand learnable queries for the batch
        q = self.q_embed.expand(B, self.num_queries, -1)  # (B, num_queries, q_dim)

        for layer in self.layers:
            # Cross-attention: queries attend to vision tokens
            q2, _ = layer["attn"](
                query=q,       # (B, num_queries, q_dim)
                key=v,         # (B, N_v, q_dim)
                value=v
            )
            q = q + q2
            q = layer["ln1"](q)

            # MLP
            q2 = layer["mlp"](q)
            q = q + q2
            q = layer["ln2"](q)

        return q  # (B, num_queries, q_dim)


# =========================
#  Tiny Text Head (VQA-style classifier)
# =========================

class TinyTextEncoder(nn.Module):
    """
    Encode a question as token embeddings + a small Transformer encoder.
    """
    def __init__(self, vocab_size=1000, d_model=256, max_len=32, depth=2, nhead=4):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.max_len = max_len

    def forward(self, input_ids):
        """
        input_ids: (B, L) integer tokens
        """
        B, L = input_ids.shape
        x = self.token_embed(input_ids)  # (B, L, D)
        pos = self.pos_embed[:, :L, :]   # (1, L, D)
        x = x + pos
        x = self.encoder(x)              # (B, L, D)
        return x


class MiniBLIP2VQA(nn.Module):
    """
    Miniature BLIP2-style VQA model:

      image -> TinyViTEncoder -> QFormerLite -> q_tokens
      question_ids -> TinyTextEncoder -> q_emb

      fuse [q_tokens, question_tokens] and classify an answer.
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        vision_dim=256,
        q_dim=256,
        num_q=16,
        text_vocab=1000,
        max_q_len=32,
        num_answers=50,
    ):
        super().__init__()

        # Vision + Q-Former
        self.vision_encoder = TinyViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=vision_dim,
            depth=2,
            nhead=4
        )
        self.q_former = QFormerLite(
            vision_dim=vision_dim,
            q_dim=q_dim,
            num_queries=num_q,
            num_layers=2,
            nhead=4
        )

        # Question encoder
        self.text_encoder = TinyTextEncoder(
            vocab_size=text_vocab,
            d_model=q_dim,
            max_len=max_q_len,
            depth=1,
            nhead=4
        )

        # Simple fusion + classifier
        self.fusion_ln = nn.LayerNorm(q_dim)
        self.classifier = nn.Linear(q_dim, num_answers)

    def forward(self, images, question_ids):
        """
        images: (B, 3, H, W)
        question_ids: (B, L)
        returns: answer logits (B, num_answers)
        """
        # 1) Vision -> tokens
        vision_tokens = self.vision_encoder(images)    # (B, N_v, D_v)

        # 2) Q-Former -> compressed visual tokens
        q_tokens = self.q_former(vision_tokens)        # (B, N_q, q_dim)

        # 3) Text encoder
        q_emb = self.text_encoder(question_ids)        # (B, L, q_dim)

        # 4) Fuse: mean-pool q_tokens + mean-pool text tokens
        v_feat = q_tokens.mean(dim=1)                  # (B, q_dim)
        t_feat = q_emb.mean(dim=1)                     # (B, q_dim)

        fused = self.fusion_ln(v_feat + t_feat)        # (B, q_dim)

        # 5) Predict answer (classification over a small answer vocab)
        logits = self.classifier(fused)                # (B, num_answers)
        return logits


# =========================
#  Demo
# =========================

if __name__ == "__main__":
    torch.manual_seed(0)

    B = 2
    H = W = 224
    img = torch.randn(B, 3, H, W)          # dummy images
    question_ids = torch.randint(0, 1000, (B, 12))  # dummy tokenized questions

    model = MiniBLIP2VQA(
        img_size=224,
        patch_size=16,
        vision_dim=256,
        q_dim=256,
        num_q=16,
        text_vocab=1000,
        max_q_len=32,
        num_answers=20,
    )

    logits = model(img, question_ids)
    print("logits shape:", logits.shape)  # (B, 20)
    print("example probs:", F.softmax(logits[0], dim=-1))
