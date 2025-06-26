import torch
import torch.nn as nn
from transformer import Tokenizer, TransformerBlock
from contextlib import contextmanager
from torch.utils.data import DataLoader


@contextmanager
def eval_mode(model: nn.Module):
    """Temporarily set `model` to eval(), restoring its original training state on exit."""
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        return x + self.fc2(self.relu(self.fc1(x)))


class TimeThetaEmbed(nn.Module):
    def __init__(self, dim_theta, embed_dim):
        super().__init__()
        # embed concatenated [t, theta] vector
        self.theta_emb = nn.Sequential(
            nn.Linear(dim_theta + 1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    def forward(self, t, theta):  # t: (B,), theta: (B, dim_theta)
        # concatenate time scalar and theta vector
        inp = torch.cat([t.unsqueeze(-1), theta], dim=1)  # (B, dim_theta+1)
        return self.theta_emb(inp)  # (B, embed_dim)


def edge_mask_from_marginal(marginal_mask):
    B, M = marginal_mask.shape
    edge_mask = torch.ones((B, M, M), dtype=torch.bool)
    for row, ei in zip(marginal_mask, edge_mask):
        ei[row.bool()] = False
        ei[:, row.bool()] = False
    return edge_mask


class ContextTransformer(nn.Module):
    def __init__(self,
                 num_nodes,  # Number of nodes (i.e., variables/features) in x
                 # Tokenizer parameters
                 dim_value,  # Dimension of the value embedding
                 dim_id,  # Dimension of the node ID embedding
                 dim_local, dim_global,  # Dimension of the marginal info embedding
                 # Attention embedding dimension
                 attn_embed_dim, # Dimension of the attention embedding
                 num_heads,      # Number of attention heads
                 num_layers,     # Number of transformer layers
                 widening_factor=4,  # Widening factor for feedforward layers
                 dropout=0.1):
        super().__init__()
        self.tokenizer = Tokenizer(num_nodes, dim_value, dim_id, dim_local, dim_global, attn_embed_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(attn_embed_dim, num_heads, widening_factor=widening_factor, dropout_rate=dropout)
            for _ in range(num_layers)
        ])
        # Project the pooled embedding (if you want to change its size, you can tweak in/out dims here)
        self.output_layer = nn.Linear(attn_embed_dim, attn_embed_dim)

    def forward(self, x, node_ids, marginal_mask):
        edge_mask = edge_mask_from_marginal(marginal_mask)
        tokens = self.tokenizer(x, node_ids, marginal_mask)
        for block in self.transformer_blocks:
            tokens = block(tokens, edge_mask)
        # Mean pooling, helps towards permutation-invariant aggregator
        ctx = tokens.mean(dim=1)
        # final projection → still (B, E)
        return self.output_layer(ctx)


class ContinuousFlowModel(nn.Module):
    def __init__(self,
                 num_nodes,   # Number of nodes (i.e., variables/features) in x
                 dim_theta,   # Dimension of the theta vector
                 # Tokenizer parameters
                 dim_value, dim_id, dim_local, dim_global,
                 attn_embed_dim, num_heads, num_layers, widening_factor, dropout,
                 # theta+t embedding parameters,
                 theta_embed_dim,
                 # Flow network parameters
                 flow_hidden_dim, flow_depth):
        super().__init__()
        # context transformer for x
        self.context_net = ContextTransformer(
            num_nodes, dim_value, dim_id, dim_local, dim_global,
            attn_embed_dim, num_heads, num_layers, widening_factor, dropout
        )
        # embed time+theta jointly
        self.time_theta_embed = TimeThetaEmbed(dim_theta, theta_embed_dim)
        # flow network now takes [ctx output = num_nodes, t+theta embed]
        in_dim = attn_embed_dim + theta_embed_dim  # context embedding + time+theta embedding
        layers = [ResidualBlock(flow_hidden_dim) for _ in range(flow_depth)]
        self.flow_net = nn.Sequential(
            nn.Linear(in_dim, flow_hidden_dim),
            nn.ReLU(),
            *layers,
            nn.Linear(flow_hidden_dim, dim_theta)
        )

    def forward(self, t, theta_t, x, node_ids, mask):
        # t: (B,), theta_t: (B, dim_theta)
        # x, node_ids, mask: (B, M)
        ctx_emb = self.context_net(x, node_ids, mask)         # (B, E)
        tt_emb = self.time_theta_embed(t, theta_t)            # (B, E)
        h = torch.cat([ctx_emb, tt_emb], dim=-1)      # (B, 2E)
        return self.flow_net(h)                               # (B, dim_theta)


class FlowMatchingTrainer:
    def __init__(self, model: ContinuousFlowModel, optimizer, sigma_min=0.001, time_prior_exponent=0., device='cpu'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.sigma_min = sigma_min
        self.time_prior_exponent = time_prior_exponent
        self.device = device
        self.loss_fn = nn.MSELoss()
        self.dim_theta = model.flow_net[-1].out_features

    def sample_t(self, batch_size):
        t = torch.rand(batch_size, device=self.device)
        return t.pow(1 / (1 + self.time_prior_exponent))

    def prepare_x(self, x):
        """Prepare the context data for training."""
        if len(x.shape) == 3:
            x = x.squeeze(-1)  # Remove the singleton dimension if present
        x_on_device = x.to(self.device)
        mask = torch.isnan(x_on_device)
        x_clean = torch.nan_to_num(x_on_device, nan=0.0)  # Replace NaNs with 0.0
        B, M = x_clean.shape
        node_ids = torch.arange(M, device=self.device).unsqueeze(0).repeat(B, 1)
        return x_clean, node_ids, mask

    def sample_theta0(self, batch_size, dim_theta=None):
        """Sample theta_0 from the gaussian prior."""
        if dim_theta is None:
            if self.dim_theta is None:
                raise ValueError("dim_theta must be set before sampling theta0.")
            dim_theta = self.dim_theta
        return torch.randn(batch_size, dim_theta, device=self.device)

    def prepare_vectorfield_pred(self, theta1, x):
        # Get mask and clean x
        x_clean, node_ids, mask = self.prepare_x(x)
        # Sample t and theta0
        if len(theta1.shape) == 3:
            theta1 = theta1.squeeze(-1)
        B, dim_theta = theta1.shape
        self.dim_theta = dim_theta  # Store the dimension of theta for later use
        t = self.sample_t(B)
        theta0 = self.sample_theta0(B)
        theta_t = (1 - (1 - self.sigma_min) * t)[:,None] * theta0 + t[:, None] * theta1
        true_vf = theta1 - (1 - self.sigma_min) * theta0
        return t, theta_t, x_clean, node_ids, mask, true_vf

    def validation_step(self, theta1, x):
        # Prepare inputs
        t, theta_t, x_clean, node_ids, mask, true_vf = self.prepare_vectorfield_pred(theta1, x)

        with torch.no_grad():
            pred_vf = self.model(t, theta_t, x_clean, node_ids, mask)
            loss = self.loss_fn(pred_vf, true_vf)

        return loss.item()

    def train_step(self, theta1, x):
        # Prepare inputs
        t, theta_t, x_clean, node_ids, mask, true_vf = self.prepare_vectorfield_pred(theta1, x)
        # Predict the vector field
        pred_vf = self.model(t, theta_t, x_clean, node_ids, mask)
        # Compute loss & backpropagate
        loss = self.loss_fn(pred_vf, true_vf)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
        # sigma_t = 1 - (1 - self.sigma_min) * t
        # theta_t = sigma_t.unsqueeze(-1) * theta0 + t.unsqueeze(-1) * theta1
        # true_vf = (theta1 - (1 - self.sigma_min) * theta0) / (sigma_t.unsqueeze(-1) + 1e-6)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        for theta1, x in dataloader:
            total_loss += self.train_step(theta1, x)
        return total_loss / len(dataloader)

    def evaluate_vectorfield(self, t, theta_t, x, node_ids, mask):
        # If t is a number (and thus the same for each element in this batch),
        # expand it as a tensor. This is required for the odeint solver.
        t = t * torch.ones(len(theta_t), device=self.device)
        return self.model(t, theta_t, x, node_ids, mask)

    def sample_batch(self, x, num_steps: int = 64):
        """
        Draw posterior samples by numerically integrating
        dθ/dt = v(t, θ) with forward Euler from t=0 to t=1.
        """
        # 1) prepare inputs
        x_clean, node_ids, mask = self.prepare_x(x)   # x_clean: (B,M)
        B, _ = x_clean.shape

        # 2) initialize at pure-noise
        theta = self.sample_theta0(B)                 # (B, dim_theta)

        # 3) set up Euler
        dt = 1.0 / num_steps
        t = torch.zeros(B, device=self.device)

        # 4) integrate
        with torch.no_grad(), eval_mode(self.model):
            for _ in range(num_steps):
                v = self.evaluate_vectorfield(t, theta, x_clean, node_ids, mask)
                theta = theta + dt * v
                t = t + dt

        return theta  # shape (B, dim_theta)


def train_with_checkpoint(
    trainer: FlowMatchingTrainer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    ckpt_path: str = "checkpoints/best_model.pt"
):
    best_val = float('inf')
    for epoch in range(1, num_epochs+1):
        train_loss = trainer.train_epoch(train_loader)

        # validation
        val_loss = 0.0
        for theta_val, x_val in val_loader:
            val_loss += trainer.validation_step(theta_val, x_val)
        val_loss /= len(val_loader)

        print(f"Epoch {epoch} — train {train_loss:.4f}, val {val_loss:.4f}", end='')
        if val_loss < best_val:
            best_val = val_loss
            torch.save(trainer.model.state_dict(), ckpt_path)
            print("  (improved, checkpoint saved)")
        else:
            print()

    # load best
    trainer.model.load_state_dict(torch.load(ckpt_path))
    print(f"Best model loaded (val_loss={best_val:.4f})")