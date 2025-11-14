import torch
import torch.nn as nn


class QFormerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.2, mlp_dropout=0.3):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)

        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(mlp_dropout),
        )

    def forward(self, queries, visual_features, visual_mask=None):
        """
        Args:
            queries: [B, num_queries, dim]
            visual_features: [B, num_patches, dim]
            visual_mask: [B, num_patches] optional attention mask
        """
        attn_out, _ = self.self_attn(queries, queries, queries)
        queries = queries + self.dropout1(attn_out)
        queries = self.norm1(queries)

        attn_out, _ = self.cross_attn(
            query=queries,
            key=visual_features,
            value=visual_features,
            key_padding_mask=visual_mask,
        )
        queries = queries + self.dropout2(attn_out)
        queries = self.norm2(queries)

        # Feed-forward
        queries = queries + self.mlp(self.norm3(queries))

        return queries


class QFormer(nn.Module):
    def __init__(
        self,
        num_queries=64,
        dim=768,
        num_heads=8,
        depth=6,
        visual_dim=1024,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()

        self.num_queries = num_queries
        self.dim = dim

        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

        self.visual_proj = nn.Linear(visual_dim, dim)
        self.blocks = nn.ModuleList(
            [QFormerBlock(dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, visual_features, visual_mask=None):
        """
        Args:
            visual_features: [B, num_patches, visual_dim] e.g., [1, 8192, 1024]
            visual_mask: [B, num_patches] boolean mask (True = masked position)
        """
        batch_size = visual_features.shape[0]

        # Project visual features
        visual_features = self.visual_proj(visual_features)  # [B, 8192, 768]

        # Expand query tokens for batch
        queries = self.query_tokens.expand(batch_size, -1, -1)  # [B, 64, 768]

        # Pass through Q-Former blocks
        for block in self.blocks:
            queries = block(queries, visual_features, visual_mask)

        # Final norm
        queries = self.norm(queries)

        return queries
