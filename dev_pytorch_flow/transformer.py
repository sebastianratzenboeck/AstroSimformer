import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueEmbed(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.linear = nn.Linear(1, embed_dim)

    def forward(self, x):
        return self.linear(x)


class ErrorEmbed(nn.Module):
    """
    Embed measurement errors using Fourier features for robust scale handling.
    """

    def __init__(self, embed_dim, fourier_dim=128, scale=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        # Random Fourier features (fixed, not learned)
        self.register_buffer('B', torch.randn(fourier_dim // 2) * scale * 2 * torch.pi)
        # Project Fourier features to desired embedding dimension
        self.projection = nn.Linear(fourier_dim, embed_dim)

    def forward(self, errors):
        # errors: (B, N, 1)
        # Squeeze to (B, N) for Fourier projection
        e = errors.squeeze(-1)  # (B, N)
        # Fourier features
        e_proj = e.unsqueeze(-1) * self.B  # (B, N, fourier_dim//2)
        e_fourier = torch.cat([torch.sin(e_proj), torch.cos(e_proj)], dim=-1)  # (B, N, fourier_dim)
        # Project to embedding dimension
        e_embed = self.projection(e_fourier)  # (B, N, embed_dim)
        # Handle NaN errors (missing measurements)
        e_embed = torch.nan_to_num(e_embed, nan=0.0)
        return e_embed


class ConditionEmbed(nn.Module):
    """
    If this variable is conditioned on, use this global learned vector. Otherwise, use nothing (i.e. zero vector).
    """
    def __init__(self, dim_condition):
        super().__init__()
        self.condition_embedding = nn.Parameter(torch.randn(1, 1, dim_condition) * 0.5)

    def forward(self, condition_mask):
        # condition_mask: (B, N, 1)
        cond_emb = self.condition_embedding * condition_mask  # (1, 1, C) * (B, N, 1)
        return cond_emb.expand(condition_mask.size(0), condition_mask.size(1), -1)


class NodeIDEmbed(nn.Module):
    def __init__(self, num_nodes, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embed_dim)

    def forward(self, node_ids):
        return self.embedding(node_ids)


class TimeEmbed(nn.Module):
    def __init__(self, time_embed_dim, input_dim=1):
        super().__init__()
        assert time_embed_dim % 2 == 0, "time_embed_dim must be even"
        self.B = nn.Parameter(torch.randn(time_embed_dim // 2, input_dim) * 2 * torch.pi)

    def forward(self, t):
        # t shape: (B, 1, 1)
        t = t.squeeze(-1)  # shape: (B, 1)
        proj = 2 * torch.pi * t @ self.B.T  # (B, time_embed_dim//2 + 1)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (B, time_embed_dim)

class Tokenizer(nn.Module):
    def __init__(
            self,
            dim_value,
            dim_id,
            dim_condition,
            attn_embed_dim,
            num_nodes,
            dim_error=None,  # NEW: dimension for error embedding
            use_error_embedding=True  # NEW: whether to use error embeddings
    ):
        super().__init__()
        self.use_error_embedding = use_error_embedding

        self.value_embed = ValueEmbed(dim_value)
        self.id_embed = NodeIDEmbed(num_nodes, dim_id)
        self.cond_embed = ConditionEmbed(dim_condition)
        # NEW: Error embedding
        if use_error_embedding and (dim_error is not None):
            self.error_embed = ErrorEmbed(dim_error, fourier_dim=128, scale=1.0)
            total_dim = dim_value + dim_id + dim_condition + dim_error
        else:
            total_dim = dim_value + dim_id + dim_condition

        # Linear projection to the attention embedding dimension
        self.output_proj = nn.Linear(total_dim, attn_embed_dim)

    def forward(self, x, node_ids, condition_mask, errors=None):
        val_emb = self.value_embed(x)  # (B, N, dim_value)
        id_emb = self.id_embed(node_ids)  # (B, N, dim_id)
        cond_emb = self.cond_embed(condition_mask)  # (B, N, dim_condition)
        # Concatenate embeddings
        embeddings = [val_emb, id_emb, cond_emb]
        # NEW: Error embedding (if provided and enabled)
        if self.use_error_embedding and (errors is not None):
            # Ensure errors have shape (B, N, 1)
            if errors.dim() == 2:
                errors = errors.unsqueeze(-1)
            err_emb = self.error_embed(errors)  # (B, N, dim_error)
            embeddings.append(err_emb)

        # Concatenate all embeddings
        token = torch.cat(embeddings, dim=-1)  # (B, N, total_dim)
        # Project to attention embedding dimension
        return self.output_proj(token)  # (B, N, attn_embed_dim)


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


class Simformer(nn.Module):
    def __init__(self,
                 num_nodes,  # Number of nodes (i.e., variables/features) in x
                 # Tokenizer parameters
                 dim_value,      # Dimension of the value embedding
                 dim_id,         # Dimension of the node ID embedding
                 dim_condition,  # Dimension of the condition embedding
                 dim_error=None,  # NEW: Dimension of error embedding (optional)
                 use_error_embedding=True,  # NEW: Whether to use errors
                 # Attention embedding dimension
                 attn_embed_dim=64,  # Dimension of the attention embedding
                 num_heads=4,        # Number of attention heads
                 num_layers=3,       # Number of transformer layers
                 widening_factor=4,  # Widening factor for feedforward layers
                 time_embed_dim=32,
                 dropout=0.1):
        super().__init__()
        self.use_error_embedding = use_error_embedding

        self.tokenizer = Tokenizer(
            dim_value,
            dim_id,
            dim_condition,
            attn_embed_dim,
            num_nodes,
            dim_error=dim_error,
            use_error_embedding=use_error_embedding
        )
        self.time_embed = TimeEmbed(time_embed_dim)
        self.time_proj = nn.Linear(time_embed_dim, attn_embed_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(attn_embed_dim, num_heads, widening_factor=widening_factor, dropout_rate=dropout)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(attn_embed_dim, 1)

    def forward(self, t, x, node_ids, condition_mask, edge_mask, errors=None):
        tokens = self.tokenizer(x, node_ids, condition_mask, errors=errors)
        t_context = self.time_embed(t)  # shape: (B, time_embed_dim)
        t_context = self.time_proj(t_context)  # shape: (B, attn_embed_dim)

        for block in self.transformer_blocks:
            # reshape for MultiheadAttention: (M, B, embed_dim)
            tokens = block(tokens, edge_mask, context=t_context)
        # return self.output_layer(tokens)
        # Predict velocity
        v = self.output_layer(tokens)  # (B, M, 1)
        # Zero velocity on conditioned coords
        if condition_mask is not None:
            v = v * (1.0 - condition_mask)  # broadcast-safe
        return v



