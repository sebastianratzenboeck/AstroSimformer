import jax
import jax.numpy as jnp
import haiku as hk
from typing import Dict, Tuple


def make_encoder(in_dim, out_dim, dropout_rate=0.0, use_batchnorm=False):
    def forward(x, is_training=True):
        layers = []
        layers.append(hk.Linear(out_dim))

        if use_batchnorm:
            layers.append(hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9))
            # note: this layer will need `is_training` at apply time
        layers.append(jax.nn.relu)

        if dropout_rate > 0:
            layers.append(lambda x: jnp.where(
                is_training,
                hk.dropout(hk.next_rng_key(), dropout_rate, x),
                x
            ))
        layers.append(hk.Linear(out_dim))

        def sequential(x):
            for layer in layers:
                if isinstance(layer, hk.BatchNorm):
                    x = layer(x, is_training)
                else:
                    x = layer(x)
            return x
        return sequential(x)

    return hk.transform_with_state(forward)


def make_adapter(in_dim, out_dim):
    def forward(x):
        return hk.Linear(out_dim)(x)
    return hk.without_apply_rng(hk.transform(forward))


def initialize_adapters_and_encoders(
    input_dims: Dict[str, int],
    latent_dims: Dict[str, int],
    key: jax.random.PRNGKey,
    dropout_rate: float = 0.0,
    use_batchnorm: bool = False,
    freeze_adapters: bool = False,
    freeze_encoders: bool = False
) -> Tuple[Dict[str, Tuple], Dict[str, Tuple]]:
    """
    Initializes Haiku adapters and encoders for each modality, with modality-specific latent dimensions.

    Returns:
        adapters: dict mapping modality to (fn, params, frozen)
        encoders: dict mapping modality to (fn, params, state, frozen)
    """
    adapters = {}
    encoders = {}
    subkeys = jax.random.split(key, len(input_dims) * 4)

    for i, (mod, in_dim) in enumerate(input_dims.items()):
        k1, k2, k3, k4 = subkeys[4 * i:4 * i + 4]
        latent_dim = latent_dims[mod]

        adapter_fn = make_adapter(in_dim, latent_dim)
        encoder_fn = make_encoder(latent_dim, latent_dim, dropout_rate, use_batchnorm)

        dummy_input = jnp.zeros((1, in_dim))
        adapter_params = adapter_fn.init(k1, dummy_input)

        dummy_latent = jnp.zeros((1, latent_dim))
        encoder_params, encoder_state = encoder_fn.init(k2, dummy_latent, is_training=True)

        adapters[mod] = (adapter_fn, adapter_params, freeze_adapters)
        encoders[mod] = (encoder_fn, encoder_params, encoder_state, freeze_encoders)

    return adapters, encoders
