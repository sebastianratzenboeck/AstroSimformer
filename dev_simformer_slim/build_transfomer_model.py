import jax
import jax.numpy as jnp
from jaxtyping import Array
from typing import Optional
import haiku as hk
from sim_core.transformers import Transformer, GaussianFourierEmbedding
from utils import marginalize



class BuildTransformerModel:
    def __init__(
            self,
            sde, nodes_max, dim_value, dim_id, dim_condition, num_heads, num_layers, attn_size, widening_factor
    ):
        self.sde = sde
        self.nodes_max = nodes_max
        self.dim_value = dim_value
        self.dim_id = dim_id
        self.dim_condition = dim_condition
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.attn_size = attn_size
        self.widening_factor = widening_factor

    def __call__(self):
        return self.init_model_fn()

    def get_sde(self):
        return self.sde

    def output_scale_fn(self, t, x):
        scale = jnp.clip(self.sde.marginal_stddev(t, jnp.ones_like(x)), 1e-2, None)
        return (1/scale * x).reshape(x.shape)

    def model(self, t: Array, x: Array, node_ids: Array, condition_mask: Array, edge_mask: Optional[Array]=None):
        """Simplified Simformer model.

        Args:
            t (Array): Diffusion time
            x (Array): Value of the nodes
            node_ids (Array): Id of the nodes
            condition_mask (Array): Condition state of the nodes
            edge_mask (Array, optional): Edge mask for attention. If None, will be created using marginalize function
                based on NaN values in x. Defaults to None.

        Returns:
            Array: Score estimate of p(x_t)
        """
        # Get the batch size and sequence length from the shape of 'x'
        batch_size, seq_len, _ = x.shape
        # Prepare the condition_mask to ensure it is boolean and reshape it
        condition_mask = condition_mask.astype(jnp.bool_).reshape(-1, seq_len, 1)
        # If edge_mask is None, create it using the marginalize function
        if edge_mask is None:
            # Remove the last dimension and apply marginalize per sample in batch
            x_2d = x.reshape(batch_size, seq_len)
            # Vmap marginalize over the batch dimension
            edge_mask = jax.vmap(marginalize)(x_2d)
        # Ensure edge_mask has correct type
        edge_mask = edge_mask.astype(jnp.bool_)
        # Reshape node_ids for proper tokenization and embedding
        node_ids = node_ids.reshape(-1, seq_len)
        # Reshape diffusion time 't' to broadcast across the nodes
        t = t.reshape(-1, 1, 1)
        # Diffusion time embedding net (here we use a Gaussian Fourier embedding)
        embedding_time = GaussianFourierEmbedding(128)
        time_embeddings = embedding_time(t)
        # Tokenization part --------------------------------------------------------------------------------
        embedding_net_value = lambda x: jnp.repeat(x, self.dim_value, axis=-1)
        embedding_net_id = hk.Embed(self.nodes_max, self.dim_id, w_init=hk.initializers.RandomNormal(stddev=3.))
        condition_embedding = hk.get_parameter(
            "condition_embedding",
            shape=(1, 1, self.dim_condition),
            init=hk.initializers.RandomNormal(stddev=0.5)
        )
        condition_embedding = condition_embedding * condition_mask
        condition_embedding = jnp.broadcast_to(condition_embedding, (batch_size, seq_len, self.dim_condition))
        # Embed inputs and broadcast
        value_embeddings = embedding_net_value(x)
        id_embeddings = embedding_net_id(node_ids)
        value_embeddings, id_embeddings = jnp.broadcast_arrays(value_embeddings, id_embeddings)
        # Concatenate embeddings
        x_encoded = jnp.concatenate([value_embeddings, id_embeddings, condition_embedding], axis=-1)
        # Example usage:
        # jax.debug.print("x_enoded is {}",x_encoded)
        # Transformer part --------------------------------------------------------------------------------
        model = Transformer(
            num_heads=self.num_heads, num_layers=self.num_layers,
            attn_size=self.attn_size, widening_factor=self.widening_factor
        )
        # Pass the edge_mask to the transformer for masked attention
        h = model(x_encoded, context=time_embeddings, mask=edge_mask)
        # Decode
        out = hk.Linear(1)(h)
        out = self.output_scale_fn(t, out)
        return out

    def init_model_fn(self):
        # In Haiku, we need to initialize the model first, before we can use it.
        # Init function initializes the parameters of the model,
        # model_fn is the actual model function (which takes the parameters as first argument, hence is a "pure function")
        init, model_fn = hk.without_apply_rng(hk.transform(self.model))
        return init, model_fn