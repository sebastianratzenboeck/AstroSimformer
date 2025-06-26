import jax
import jax.numpy as jnp
import haiku as hk
from jaxtyping import Array
from typing import Optional
from transformers import Transformer, GaussianFourierEmbedding
from train_interface.utils import marginalize


def build_transformer_model(
        nodes_max: int,
        dim_value: int,      # Dimension for value embeddings
        dim_id: int,         # Dimension for node ID embeddings
        dim_condition: int,  # Dimension for condition embeddings
        num_heads: int,
        num_layers: int,
        attn_size: int,
        widening_factor: int,
        embed_time_dim: int = 128
):
    """
    Returns a haiku-transformed model function suitable for flow matching.
    """
    def model(t: Array, x: Array, node_ids: Array, condition_mask: Array, edge_mask: Optional[Array] = None) -> Array:
        if x.ndim == 2:
            x = x[..., None]
        elif x.ndim != 3:
            raise ValueError(f"x must have shape (B, T) or (B, T, 1), got {x.shape}")
        # x shape: (B, T, 1)
        batch_size, seq_len, _ = x.shape
        # Handle condition mask shape
        condition_mask = condition_mask.astype(jnp.bool_).reshape(-1, seq_len, 1)

        # Generate edge mask from NaNs if not provided
        if edge_mask is None:
            x_2d = x.reshape(batch_size, seq_len)
            edge_mask = jax.vmap(marginalize)(x_2d)
        edge_mask = edge_mask.astype(jnp.bool_)

        # Prepare embeddings
        node_ids = node_ids.reshape(-1, seq_len)
        t = t.reshape(-1, 1, 1)

        time_embeddings = GaussianFourierEmbedding(embed_time_dim)(t)
        # Tokenizer: Each variable (parameter or data) is represented as a token containing
        # its identity, value, and whether it is conditioned or latent.
        # These tokens are processed by a transformer, with variable interactions controlled via an attention mask.
        embedding_net_value = lambda x: jnp.repeat(x, dim_value, axis=-1)
        value_embeddings = embedding_net_value(x)
        # ID embeddings for nodes
        embedding_net_id = hk.Embed(nodes_max, dim_id, w_init=hk.initializers.RandomNormal(stddev=3.0))
        id_embeddings = embedding_net_id(node_ids)
        # Condition embeddings
        condition_embedding = hk.get_parameter(
            "condition_embedding",
            shape=(1, 1, dim_condition),
            init=hk.initializers.RandomNormal(stddev=0.5)
        )
        condition_embedding = condition_embedding * condition_mask
        condition_embedding = jnp.broadcast_to(condition_embedding, (batch_size, seq_len, dim_condition))

        value_embeddings, id_embeddings = jnp.broadcast_arrays(value_embeddings, id_embeddings)

        x_encoded = jnp.concatenate([value_embeddings, id_embeddings, condition_embedding], axis=-1)

        transformer = Transformer(
            num_heads=num_heads,
            num_layers=num_layers,
            attn_size=attn_size,
            widening_factor=widening_factor,
            dropout_rate=None
        )

        h = transformer(x_encoded, context=time_embeddings, mask=edge_mask)
        return hk.Linear(1)(h)  # Output shape: (B, T, 1)
        # return 10.0 * jnp.tanh(hk.Linear(1)(h))  # safely bounded output

    init, model_fn = hk.without_apply_rng(hk.transform(model))
    return init, model_fn

