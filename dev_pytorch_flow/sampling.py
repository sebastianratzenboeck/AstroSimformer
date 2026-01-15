import torch


class SimformerConditionalVelocity(torch.nn.Module):
    def __init__(self, model_fn, condition_mask, condition_value, node_ids, edge_mask):
        super().__init__()
        # Normalize shapes/dtypes/devices
        if condition_mask.dim() == 2:
            condition_mask = condition_mask.unsqueeze(-1)           # (B, M, 1)
        if condition_value.dim() == 2:
            condition_value = condition_value.unsqueeze(-1)         # (B, M, 1)

        device = node_ids.device
        self.model_fn = model_fn
        self.condition_mask = condition_mask.to(device=device, dtype=torch.float32)     # (B, M, 1)
        self.condition_value = condition_value.to(device=device, dtype=torch.float32)   # (B, M, 1)
        self.node_ids = node_ids.to(device)
        self.edge_mask = edge_mask.to(device)

    def forward(self, t, x_flat):
        B, M, _ = self.condition_mask.shape
        # Accept either (B*M,) or (B*M,1); reshape to (B, M, 1)
        x = x_flat.view(B, M, 1)
        # Clamp to conditional manifold
        x = x * (1.0 - self.condition_mask) + self.condition_value * self.condition_mask
        # Time as (B,1,1)
        t_batch = torch.full((B, 1, 1), float(t), dtype=x.dtype, device=x.device)
        # Model call expects (B, M, 1) mask
        v = self.model_fn(t_batch, x, self.node_ids, self.condition_mask, self.edge_mask)  # (B, M, 1)
        # Zero velocity on conditioned dims
        v = v * (1.0 - self.condition_mask)
        return v.view(-1)  # (B*M,)


@torch.no_grad()
def sample_batched_flow(
    model_fn,
    shape,
    condition_mask,
    condition_values,
    node_ids,
    edge_masks,
    steps=64,
    t0=0.0,
    t1=1.0,
    device="cpu",
):
    """
    Euler integration from t0→t1.
      shape: (B,)
      condition_mask:  (B, M) or (B, M, 1)
      condition_values:(B, M) or (B, M, 1)
      node_ids:        (B, M)
      edge_masks:      (B, M, M)
    """
    B, M = shape[0], node_ids.shape[1]
    device = torch.device(device)

    # Ensure (B,M,1) for mask/values
    if condition_mask.dim() == 2:
        condition_mask = condition_mask.unsqueeze(-1)
    if condition_values.dim() == 2:
        condition_values = condition_values.unsqueeze(-1)
    condition_mask = condition_mask.to(device=device, dtype=torch.float32)
    condition_values = condition_values.to(device=device, dtype=torch.float32)
    node_ids = node_ids.to(device)
    edge_masks = edge_masks.to(device)

    dt = (t1 - t0) / steps
    ts = torch.linspace(t0, t1, steps + 1, device=device)

    # Init state and clamp to conditional manifold
    x = torch.randn(B, M, 1, device=device)
    x = x * (1.0 - condition_mask) + condition_values * condition_mask
    x_flat = x.view(-1)  # keep 1-D throughout

    velocity_fn = SimformerConditionalVelocity(
        model_fn=model_fn,
        condition_mask=condition_mask,
        condition_value=condition_values,
        node_ids=node_ids,
        edge_mask=edge_masks,
    )

    for t in ts[:-1]:
        dx = velocity_fn(t, x_flat)      # (B*M,)
        x_flat = x_flat + dt * dx        # Euler step
        # Clamp after each step
        x = x_flat.view(B, M, 1)
        x = x * (1.0 - condition_mask) + condition_values * condition_mask
        x_flat = x.view(-1)

    # Final clamp and return (B, M, 1)
    x = x_flat.view(B, M, 1)
    x = x * (1.0 - condition_mask) + condition_values * condition_mask
    return x