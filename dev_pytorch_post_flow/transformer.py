import torch
import torch.nn as nn


class ValueEmbed(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.linear = nn.Linear(1, embed_dim)

    def forward(self, x):  # x: (B, M, 1)
        return self.linear(x)

class MarginalEmbed(nn.Module):
    def __init__(self, num_nodes, dim_local, dim_global):
        """Goal: each token's representation carries both
        its own masking bit (local) and global context about which other tokens were dropped
        num_nodes: total number of tokens M
        dim_local: size of the per-token "local" mask embedding
        dim_global: size of the global summary embedding
        """
        super().__init__()
        # local embedding token
        self.local_token = nn.Parameter(torch.randn(dim_local))
        # global summary MLP over the boolean mask vector
        self.global_mlp = nn.Sequential(
            nn.Linear(num_nodes, dim_global),
            nn.ReLU(),
            nn.Linear(dim_global, dim_global),
        )

    def forward(self, mask):
        """
        mask: (B, M) boolean tensor, True if absent, False if present
        returns: (B, M, dim_local + dim_global)
        """
        B, M = mask.shape
        # local: only present tokens get non-zero
        present = (~mask).unsqueeze(-1).float()               # (B, M, 1)
        local = self.local_token.unsqueeze(0).unsqueeze(0)    # (1,1,dim_local)
        local = local.expand(B, M, -1)                        # (B, M, dim_local)
        local_emb = local * present                           # zero out absent
        # global: summary of which tokens are absent
        global_summary = self.global_mlp(mask.float())       # (B, dim_global)
        global_emb = global_summary.unsqueeze(1).expand(B, M, -1)  # (B, M, dim_global)
        return torch.cat([local_emb, global_emb], dim=-1)     # (B, M, dim_local+dim_global)


class Tokenizer(nn.Module):
    def __init__(self, num_nodes, dim_value, dim_id, dim_local, dim_global, embed_dim):
        super().__init__()
        self.value_embed = ValueEmbed(dim_value)
        self.id_embed = nn.Embedding(num_nodes, dim_id)
        self.marginal_embed = MarginalEmbed(num_nodes, dim_local, dim_global)
        self.fc = nn.Linear(dim_value + dim_id + dim_local + dim_global, embed_dim)

    def forward(self, x, node_ids, mask):
        # x: (B, M), node_ids: (B, M), mask: (B, M)
        v = x.unsqueeze(-1)  # (B, M, 1)
        e_val = self.value_embed(v)  # (B, M, dv)
        e_id = self.id_embed(node_ids)  # (B, M, di)
        e_m = self.marginal_embed(mask)  # (B, M, dm)
        e = torch.cat([e_val, e_id, e_m], dim=-1)
        return self.fc(e)  # (B, M, embed_dim)


class MaskedMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)

    def forward(self, x, attn_mask):  # attn_mask shape: (B, N, N)
        B, N, _ = x.shape
        if attn_mask is not None:
            # Ensure float mask with -inf where attention is blocked
            # attn_mask --> ~attn_mask, so False becomes True (1) = blocked, True -> False (0) = allowed
            float_mask = (~attn_mask).float() * -1e30  # Now: True = -1e30, False = 0
            # ensure it’s on the same device as x
            float_mask = float_mask.to(x.device)
            # Expand to all attention heads, shape must be: (B * num_heads, N, N)
            float_mask = float_mask.repeat_interleave(self.num_heads, dim=0)
            x_out, _ = self.attn(x, x, x, attn_mask=float_mask)
        else:
            x_out, _ = self.attn(x, x, x)
        return x_out


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, widening_factor=4, dropout_rate=0.1):
        super().__init__()
        self.attn = MaskedMultiheadAttention(embed_dim, num_heads, dropout=dropout_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout_attn = nn.Dropout(dropout_rate)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * widening_factor),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * widening_factor, embed_dim),
            nn.Dropout(dropout_rate),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, attn_mask, context=None):
        # Skip connection + attention
        x = x + self.dropout_attn(self.attn(self.norm1(x), attn_mask))
        x_context = self.norm2(x)
        if context is not None:
            while context.ndim < x_context.ndim:
                context = context.unsqueeze(1)
            x_context = x_context + context
        # Skip connection + feedforward
        x = x + self.ff(x_context)
        return x


