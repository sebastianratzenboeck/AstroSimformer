import torch


class SimformerConditionalVelocity(torch.nn.Module):
    def __init__(self, model_fn, condition_mask, condition_value, node_ids, edge_mask):
        super().__init__()
        self.model_fn = model_fn
        self.condition_mask = condition_mask.float()
        self.condition_value = condition_value
        self.node_ids = node_ids
        self.edge_mask = edge_mask

    def forward(self, t, x_flat):
        B, M = self.condition_mask.shape[:2]
        x = x_flat.view(B, M).unsqueeze(-1)  # (B, M, 1)
        x = x * (1.0 - self.condition_mask) + self.condition_value * self.condition_mask
        t_batch = torch.full((B, 1, 1), t, dtype=x.dtype, device=x.device)
        v = self.model_fn(t_batch, x, self.node_ids, self.condition_mask, self.edge_mask)
        v = v * (1.0 - self.condition_mask)
        return v.view(-1)


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
    Run Euler integration to sample from the flow matching model.
    Inputs:
        - shape: tuple (B,) for batch size
        - condition_mask: (B, M, 1)
        - condition_values: (B, M, 1)
        - node_ids: (B, M)
        - edge_masks: (B, M, M)
    """
    B, M = shape[0], node_ids.shape[1]
    dt = (t1 - t0) / steps
    ts = torch.linspace(t0, t1, steps + 1, device=device)

    x0 = torch.randn(B, M, device=device)
    x0 = x0 * (1.0 - condition_mask.squeeze(-1)) + condition_values.squeeze(-1) * condition_mask.squeeze(-1)
    x_flat0 = x0.view(-1)

    velocity_fn = SimformerConditionalVelocity(
        model_fn=model_fn,
        condition_mask=condition_mask,
        condition_value=condition_values,
        node_ids=node_ids,
        edge_mask=edge_masks,
    )

    x_flat = x_flat0.clone()
    for t in ts[:-1]:
        dx = velocity_fn(t, x_flat)
        x_flat = x_flat + dt * dx

    return x_flat.view(B, M, 1)