import jax.numpy as jnp
from typing import Dict, List, Set, Tuple
from ott.solvers.linear import solve
from ott.geometry import pointcloud
import itertools
import jax


@jax.jit
def safe_cosine_loss(z1: jnp.ndarray, z2: jnp.ndarray) -> jnp.ndarray:
    nan_mask1 = jnp.isnan(z1)
    nan_mask2 = jnp.isnan(z2)
    valid_mask = ~(jnp.any(nan_mask1, axis=-1) | jnp.any(nan_mask2, axis=-1))

    z1_clean = jnp.nan_to_num(z1)
    z2_clean = jnp.nan_to_num(z2)

    norm1 = jnp.linalg.norm(z1_clean, axis=-1) + 1e-8
    norm2 = jnp.linalg.norm(z2_clean, axis=-1) + 1e-8
    dot = jnp.sum(z1_clean * z2_clean, axis=-1)
    sim = dot / (norm1 * norm2)

    sim = jnp.where(valid_mask, sim, 0.0)
    loss = 1.0 - jnp.sum(sim) / jnp.maximum(jnp.sum(valid_mask), 1.0)

    return loss


@jax.jit
def compute_sinkhorn_ott(x: jnp.ndarray, y: jnp.ndarray, epsilon=0.05, min_iterations=10, max_iterations=100) -> jnp.ndarray:
    a = jnp.ones(x.shape[0]) / x.shape[0]
    b = jnp.ones(y.shape[0]) / y.shape[0]

    # Compute intra-cluster distances
    geom_xx = pointcloud.PointCloud(x, x, epsilon=epsilon)
    geom_yy = pointcloud.PointCloud(y, y, epsilon=epsilon)
    scale = jnp.minimum(geom_xx.mean_cost_matrix, geom_yy.mean_cost_matrix)

    # Compute cross-domain OT loss
    geom_xy = pointcloud.PointCloud(x, y, epsilon=epsilon)
    loss = solve(geom_xy, a=a, b=b, min_iterations=min_iterations, max_iterations=max_iterations).reg_ot_cost

    return jnp.abs(loss / (scale + 1e-8))

# compute_sinkhorn_ott = jax.jit(jax.vmap(single_batch_sinkhorn_ott, in_axes=(0, 0)))

@jax.jit
def compute_mmd(x: jnp.ndarray, y: jnp.ndarray, kernel_mul=2.0, kernel_num=5) -> jnp.ndarray:
    total = jnp.concatenate([x, y], axis=0)
    total0 = total[:, None, :]
    total1 = total[None, :, :]
    L2 = jnp.sum((total0 - total1) ** 2, axis=-1)
    bandwidth = jnp.sum(L2) / (total.shape[0] ** 2 - total.shape[0])
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernels = [jnp.exp(-L2 / bw) for bw in bandwidth_list]
    K = sum(kernels)

    m = x.shape[0]
    XX = K[:m, :m].mean()
    YY = K[m:, m:].mean()
    XY = K[:m, m:].mean()
    return XX + YY - 2 * XY


class ModalityDomainAdapter:
    def __init__(self, adapters, encoders, ordered_modalities,
                 average_groups: List[Set[str]],
                 contrast_groups: List[Set[str]],
                 use_ot=True, use_mmd=False, use_contrastive=True):
        self.adapters = adapters
        self.encoders = encoders
        self.ordered_modalities = ordered_modalities
        self.average_groups = average_groups
        self.contrast_groups = contrast_groups  # Not used anymore
        self.use_ot = use_ot
        self.use_mmd = use_mmd
        self.use_contrastive = use_contrastive

    def encode(self, x_dict: Dict[str, jnp.ndarray], rng: jax.random.PRNGKey, is_training: bool = True) -> Dict[str, jnp.ndarray]:
        latents = {}
        subkeys = jax.random.split(rng, len(x_dict))
        for i, (name, x) in enumerate(x_dict.items()):
            if x is not None:
                adapter_fn, adapter_params, adapter_frozen = self.adapters[name]
                encoder_fn, encoder_params, encoder_state, encoder_frozen = self.encoders[name]
                x_proj = adapter_fn.apply(adapter_params, x)
                if encoder_frozen:
                    z = encoder_fn.apply(encoder_params, encoder_state, jax.random.PRNGKey(42), x_proj, is_training=False)[0]
                else:
                    z, _ = encoder_fn.apply(encoder_params, encoder_state, subkeys[i], x_proj, is_training=is_training)
                latents[name] = z
            else:
                latents[name] = None
        return latents

    def merge_embeddings(self, embeddings: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        latent_dim = next(z.shape[1] for z in embeddings.values() if z is not None)
        batch_size = next(z.shape[0] for z in embeddings.values() if z is not None)

        out = []
        used = set()
        for group in self.average_groups:
            stacked = []
            for mod in group:
                z = embeddings.get(mod)
                if z is not None:
                    mask = ~jnp.isnan(z).any(axis=1)
                    valid = jnp.where(mask[:, None], z, 0.0)
                    stacked.append((valid, mask))
            if stacked:
                values = jnp.stack([v for v, _ in stacked], axis=0)
                masks = jnp.stack([m for _, m in stacked], axis=0)
                mask_sum = jnp.sum(masks, axis=0)
                z_avg = jnp.sum(values, axis=0) / jnp.clip(mask_sum[:, None], a_min=1.0)
                z_avg = jnp.where(mask_sum[:, None] > 0, z_avg, jnp.nan)
                out.append(z_avg)
            else:
                out.append(jnp.full((batch_size, latent_dim), jnp.nan))
            used.update(group)

        for mod in self.ordered_modalities:
            if mod not in used:
                if embeddings[mod] is not None:
                    out.append(embeddings[mod])
                else:
                    out.append(jnp.full((batch_size, latent_dim), jnp.nan))
        return jnp.concatenate(out, axis=1)

    # def compute_alignment_losses(self, z_sim, z_real):
    #     losses = {}
    #     for mod in self.ordered_modalities:
    #         if (z_sim[mod] is not None) and (z_real[mod] is not None):
    #             if self.use_ot:
    #                 losses[f"ot_{mod}"] = compute_sinkhorn_ott(z_sim[mod], z_real[mod])
    #             if self.use_mmd:
    #                 losses[f"mmd_{mod}"] = compute_mmd(z_sim[mod], z_real[mod])
    #     return losses

    def compute_contrastive_losses(self, z_real):
        losses = {}
        if self.use_contrastive:
            for group in self.average_groups:
                for m1, m2 in itertools.combinations(sorted(group), 2):
                    if (z_real.get(m1) is not None) and (z_real.get(m2) is not None):
                        losses[f"real_contrast_{m1}_{m2}"] = safe_cosine_loss(z_real[m1], z_real[m2])
        return losses

    @staticmethod
    def filter_nans_pairwise(z1: jnp.ndarray, z2: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Filter rows with NaNs in either z1 or z2."""
        valid_mask = ~(jnp.any(jnp.isnan(z1), axis=1) | jnp.any(jnp.isnan(z2), axis=1))
        return z1[valid_mask], z2[valid_mask]

    @staticmethod
    def remove_nan_rows(z: jnp.ndarray) -> jnp.ndarray:
        return z[~jnp.any(jnp.isnan(z), axis=1)]

    def compute_alignment_losses(self, z_sim, z_real):
        losses = {}
        for mod in self.ordered_modalities:
            if (z_sim[mod] is not None) and (z_real[mod] is not None):
                z1 = self.remove_nan_rows(z_sim[mod])
                z2 = self.remove_nan_rows(z_real[mod])
                if z1.shape[0] > 0:
                    if self.use_ot:
                        losses[f"ot_{mod}"] = compute_sinkhorn_ott(z1, z2)
                    if self.use_mmd:
                        losses[f"mmd_{mod}"] = compute_mmd(z1, z2)
        return losses

    def forward(self, x_s, x_t, x_s_pair=None, x_t_pair=None, rng=None, is_training=True):
        rng, rng_sim, rng_real, rng_pair1, rng_pair2 = jax.random.split(rng, 5)
        z_sim = self.encode(x_s, rng=rng_sim, is_training=is_training)
        z_real = self.encode(x_t, rng=rng_real, is_training=is_training) if x_t is not None else None

        loss_dict = self.compute_alignment_losses(z_sim, z_real) if z_real is not None else {}
        contrastive_dict = self.compute_contrastive_losses(z_real) if z_real is not None else {}
        loss_dict.update(contrastive_dict)

        # if (x_s_pair is not None) and (x_t_pair is not None):
        #     z_s_pair = self.merge_embeddings(self.encode(x_s_pair, rng=rng_pair1, is_training=is_training))
        #     z_t_pair = self.merge_embeddings(self.encode(x_t_pair, rng=rng_pair2, is_training=is_training))
        #     loss_dict["pairwise"] = safe_cosine_loss(z_s_pair, z_t_pair)

        if (x_s_pair is not None) and (x_t_pair is not None):
            z_s_pair = self.merge_embeddings(self.encode(x_s_pair, rng=rng_pair1, is_training=is_training))
            z_t_pair = self.merge_embeddings(self.encode(x_t_pair, rng=rng_pair2, is_training=is_training))
            z_s_pair, z_t_pair = self.filter_nans_pairwise(z_s_pair, z_t_pair)
            if z_s_pair.shape[0] > 0:
                loss_dict["pairwise"] = safe_cosine_loss(z_s_pair, z_t_pair)

        z_merged = self.merge_embeddings(z_sim)
        return z_merged, loss_dict
