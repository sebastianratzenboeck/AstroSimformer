import torch
from typing import Optional, Sequence, Union, Tuple, List


def make_condition_mask_generator(
    batch_size: int,
    num_features: int,
    percent: Union[float, int, Tuple[float, float], torch.Tensor] = (0.1, 0.6),
    *,
    allowed_idx: Optional[Sequence[int]] = None,   # only choose among these cols
    always_on_idx: Optional[Sequence[int]] = None, # these cols always set to 1
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    seed: Optional[int] = None,
):
    B, D = batch_size, num_features
    device = device or torch.device("cpu")
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)

    cand = torch.arange(D, device=device, dtype=torch.long) if allowed_idx is None \
           else torch.as_tensor(allowed_idx, device=device, dtype=torch.long)
    C = cand.numel()

    always_on = torch.empty(0, dtype=torch.long, device=device) if always_on_idx is None \
                else torch.as_tensor(always_on_idx, device=device, dtype=torch.long)
    always_on_in_pool = always_on[torch.isin(always_on, cand)]
    k_always = always_on_in_pool.numel()

    if k_always > 0:
        keep = ~torch.isin(cand, always_on_in_pool)
        cand_eff = cand[keep]
    else:
        cand_eff = cand
    Ce = cand_eff.numel()

    def _row_counts() -> torch.Tensor:
        if isinstance(percent, tuple) and len(percent) == 2:
            pmin, pmax = float(percent[0]), float(percent[1])
            p_i = (pmin + (pmax - pmin) * torch.rand(B, device=device, generator=g)).clamp(0.0, 1.0)
            k_tot = torch.round(p_i * C).to(torch.long)
        elif torch.is_tensor(percent):
            k_tot = percent.to(device=device)
            if k_tot.numel() == 1:
                k_tot = k_tot.expand(B)
            if k_tot.dtype.is_floating_point:
                k_tot = torch.round(k_tot.clamp(0.0, 1.0) * C).to(torch.long)
            else:
                k_tot = k_tot.to(torch.long)
            if k_tot.numel() != B:
                raise ValueError(f"'percent' tensor must have shape [{B}] or scalar")
        elif isinstance(percent, float):
            k_tot = torch.full((B,), int(round(percent * C)), device=device, dtype=torch.long)
        elif isinstance(percent, int):
            k_tot = torch.full((B,), percent, device=device, dtype=torch.long)
        else:
            raise TypeError("Unsupported type for 'percent'.")
        return k_tot.clamp(min=0, max=k_always + Ce)

    while True:
        mask = torch.zeros(B, D, dtype=dtype, device=device)
        if always_on.numel() > 0:
            mask[:, always_on] = 1.0

        if Ce == 0:
            yield mask
            continue

        k_total = _row_counts()                   # [B]
        k_rem = (k_total - k_always).clamp(min=0, max=Ce)
        k_max = int(k_rem.max().item())
        if k_max == 0:
            yield mask
            continue

        scores = torch.rand(B, Ce, device=device, generator=g)
        top_idx = scores.topk(k_max, dim=1, largest=True, sorted=False).indices  # [B, k_max]
        chosen_cols = cand_eff[top_idx]                                          # [B, k_max]
        sel_mask = (torch.arange(k_max, device=device).unsqueeze(0) < k_rem.unsqueeze(1)).to(mask.dtype)
        mask.scatter_(dim=1, index=chosen_cols, src=sel_mask)
        yield mask


# --- helper: normalize blocks to device-long tensors ---
def _as_blocks(block_indices: Sequence[Sequence[int]], device) -> List[torch.Tensor]:
    return [torch.as_tensor(b, device=device, dtype=torch.long) for b in block_indices]

# ==============================
# Block CONDITION mask generator
# ==============================
def make_block_condition_mask_generator(
    batch_size: int,
    num_nodes: int,
    block_indices: Sequence[Sequence[int]],
    *,
    prob: Union[float, Tuple[float, float]] = 1/3,  # P(block is conditioned) or Uniform[pmin,pmax]
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    avoid_all_true: bool = True,   # prevent degenerate "everything conditioned"
) :
    """
    Yields a mask of shape [B, N, 1] (float in {0,1}) where **entire blocks** are conditioned together.
    """
    B, N = batch_size, num_nodes
    device = device or torch.device("cpu")
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)

    blocks = _as_blocks(block_indices, device=device)
    NB = len(blocks)

    def _draw_block_probs() -> torch.Tensor:
        if isinstance(prob, tuple):
            pmin, pmax = float(prob[0]), float(prob[1])
            return (pmin + (pmax - pmin) * torch.rand(B, NB, device=device, generator=g)).clamp(0.0, 1.0)
        p = float(prob)
        return torch.full((B, NB), p, device=device)

    while True:
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        p_mat = _draw_block_probs()  # [B, NB]
        block_bern = (torch.rand(B, NB, device=device, generator=g) < p_mat)  # [B, NB]

        # Write selected blocks into mask
        for j, idx in enumerate(blocks):
            if idx.numel() == 0:
                continue
            sel = block_bern[:, j].unsqueeze(1).expand(B, idx.numel())  # [B, |block_j|]
            # assignment with advanced indexing writes into original
            mask[:, idx] = mask[:, idx] | sel

        if avoid_all_true:
            all_true = mask.all(dim=1, keepdim=True)  # [B,1]
            mask = mask & ~all_true  # zero-out rows that turned fully True

        yield mask.unsqueeze(-1).to(dtype=torch.float32)  # [B, N, 1]
