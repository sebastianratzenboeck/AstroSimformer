import torch
import torch.optim as optim
import wandb
from sklearn.model_selection import train_test_split
from utils import make_condition_mask_generator


class FlowMatchingTrainer:
    def __init__(
            self,
            model,
            data,
            data_errors=None,
            condition_mask_generator=None,
            batch_size=128,
            lr=1e-3,
            sigma_min=0.001,
            time_prior_exponent=1.0,
            inner_train_loop_size=1000,
            early_stopping_patience=20,
            val_split=0.15,
            val_every_n_epochs=1,
            use_wandb=True,
            wandb_project="flow-matching",
            wandb_config=None,
            device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.condition_mask_generator = condition_mask_generator
        self.batch_size = batch_size
        self.lr = lr
        self.sigma_min = sigma_min
        self.time_prior_exponent = time_prior_exponent
        self.inner_train_loop_size = inner_train_loop_size
        self.early_stopping_patience = early_stopping_patience
        self.val_every_n_epochs = val_every_n_epochs
        self.use_wandb = use_wandb
        self.device = device

        # Split data into train and validation
        if val_split > 0:
            if data_errors is not None:
                train_data, val_data, train_errors, val_errors = train_test_split(
                    data, data_errors, test_size=val_split, random_state=42
                )
                self.train_errors = torch.tensor(train_errors, dtype=torch.float32).to(device)
                self.val_errors = torch.tensor(val_errors, dtype=torch.float32).to(device)
            else:
                train_data, val_data = train_test_split(
                    data, test_size=val_split, random_state=42
                )
                self.train_errors = None
                self.val_errors = None

            self.train_data = torch.tensor(train_data, dtype=torch.float32).to(device)
            self.val_data = torch.tensor(val_data, dtype=torch.float32).to(device)
        else:
            self.train_data = torch.tensor(data, dtype=torch.float32).to(device)
            self.val_data = None
            if data_errors is not None:
                self.train_errors = torch.tensor(data_errors, dtype=torch.float32).to(device)
                self.val_errors = None
            else:
                self.train_errors = None
                self.val_errors = None

        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.best_model_state = None

        self.num_nodes = self.train_data.shape[1]
        self.node_ids = torch.arange(self.num_nodes).unsqueeze(0).repeat(self.batch_size, 1).to(device)

        # Initialize wandb
        if self.use_wandb:
            config = wandb_config or {}
            config.update({
                "batch_size": batch_size,
                "lr": lr,
                "sigma_min": sigma_min,
                "time_prior_exponent": time_prior_exponent,
                "inner_train_loop_size": inner_train_loop_size,
                "num_nodes": self.num_nodes,
                "train_samples": len(self.train_data),
                "val_samples": len(self.val_data) if self.val_data is not None else 0,
                "has_errors": self.train_errors is not None,
            })
            wandb.init(project=wandb_project, config=config)
            wandb.watch(self.model, log="all", log_freq=100)

    def sample_batch(self, from_val=False):
        """Sample batch of data and corresponding errors."""
        if from_val:
            data = self.val_data
            errors = self.val_errors
        else:
            data = self.train_data
            errors = self.train_errors

        idx = torch.randint(0, data.shape[0], (self.batch_size,))
        batch_data = data[idx]
        batch_errors = errors[idx] if errors is not None else None
        return batch_data, batch_errors

    def build_edge_mask(self, dense_ratio=0.7):
        n_dense = int(self.batch_size * dense_ratio)
        dense = torch.ones((n_dense, self.num_nodes, self.num_nodes), dtype=torch.bool)
        sparse = []
        for _ in range(self.batch_size - n_dense):
            m = torch.ones((self.num_nodes, self.num_nodes), dtype=torch.bool)
            drop = torch.rand(self.num_nodes) < 0.2
            m[drop] = False
            m[:, drop] = False
            m[drop, drop] = True
            sparse.append(m)
        if sparse:
            sparse = torch.stack(sparse, dim=0)
            masks = torch.cat([dense, sparse], dim=0)
        else:
            masks = dense
        indices = torch.randint(0, masks.shape[0], (self.batch_size,))
        return masks[indices].to(self.device)

    def sample_t(self, batch_size):
        t = torch.rand(batch_size, device=self.device)
        return t.pow(1 / (1 + self.time_prior_exponent))

    def flow_matching_loss(self, x0, x1, node_ids, condition_mask, edge_mask, x_errors=None):
        """Compute standard flow matching loss."""
        B, N = x0.shape
        t = self.sample_t(B).reshape(B, 1, 1)
        k = (1.0 - self.sigma_min)

        # Geodesic mix + clamp to conditioned values (no grad into x1 on those dims)
        xt = (1 - k * t) * x0.unsqueeze(-1) + t * x1.unsqueeze(-1)
        # xt = torch.where(condition_mask.bool(), x1.unsqueeze(-1), xt)
        # clamp xt to conditioned values, no grad through x1 on those dims
        cond = condition_mask.to(x0.dtype).unsqueeze(-1)  # [B,D,1]
        free = 1.0 - cond
        xt = xt * free + x1.unsqueeze(-1).detach() * cond

        # predict velocity
        v_pred = self.model(t, xt, node_ids, condition_mask, edge_mask, x_errors).squeeze(-1)  # [B,D]
        v_tgt = (x1 - k * x0)  # [B,D]

        # Project BOTH velocities to the free subspace (stronger than masking the loss)
        free_s = free.squeeze(-1)  # [B,D]
        v_pred = v_pred * free_s
        v_tgt = v_tgt * free_s

        # MSE over free dims, normalized per sample by #free dims
        diff = (v_pred - v_tgt).pow(2)  # [B,D]
        counts = free_s.sum(-1).clamp_min(1.0)  # [B]
        loss = (diff.sum(-1) / counts).mean()  # scalar
        return loss

    def training_step(self, x0, x1, node_ids, condition_mask, edge_mask, x_errors):
        """Perform one training step."""
        loss = self.flow_matching_loss(x0, x1, node_ids, condition_mask, edge_mask, x_errors)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    @torch.no_grad()
    def validate(self, num_batches=50):
        """Compute validation loss over multiple batches."""
        if self.val_data is None:
            return None

        self.model.eval()
        total_val_loss = 0.0

        for _ in range(num_batches):
            x1, x_errors = self.sample_batch(from_val=True)
            x0 = torch.randn_like(x1)
            # condition_mask = self.build_condition_mask()
            condition_mask = next(self.condition_mask_generator)
            edge_mask = self.build_edge_mask(dense_ratio=0.6)

            loss = self.flow_matching_loss(x0, x1, self.node_ids, condition_mask, edge_mask, x_errors)
            total_val_loss += loss.item()

        self.model.train()
        return total_val_loss / num_batches

    def fit(self, epochs=100, verbose=True):
        """Train the model."""
        best_loss = float("inf")
        no_improve = 0
        global_step = 0

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0

            for step in range(self.inner_train_loop_size):
                x1, x_errors = self.sample_batch(from_val=False)
                x0 = torch.randn_like(x1)

                # condition_mask = self.build_condition_mask()
                condition_mask = next(self.condition_mask_generator)
                edge_mask = self.build_edge_mask(dense_ratio=0.6)

                step_metrics = self.training_step(x0, x1, self.node_ids, condition_mask, edge_mask, x_errors)

                total_loss += step_metrics['loss']
                global_step += 1

                # Log to wandb every N steps
                if self.use_wandb and step % 100 == 0:
                    wandb.log({
                        "train_loss_step": step_metrics['loss'],
                        "epoch": epoch + 1,
                        "step": global_step
                    })

            avg_train_loss = total_loss / self.inner_train_loop_size

            # Validate periodically
            val_loss = None
            if epoch % self.val_every_n_epochs == 0:
                val_loss = self.validate()

            # Logging
            if verbose:
                log_str = f"Epoch {epoch + 1}: train_loss = {avg_train_loss:.6f}"
                if val_loss is not None:
                    log_str += f", val_loss = {val_loss:.6f}"
                print(log_str)

            if self.use_wandb:
                log_dict = {
                    "train_loss_epoch": avg_train_loss,
                    "epoch": epoch + 1
                }
                if val_loss is not None:
                    log_dict["val_loss"] = val_loss
                wandb.log(log_dict)

            # Early stopping based on validation loss (if available) or training loss
            monitor_loss = val_loss if val_loss is not None else avg_train_loss

            if monitor_loss < best_loss:
                best_loss = monitor_loss
                no_improve = 0
                self.best_model_state = self.model.state_dict()
                if self.use_wandb:
                    wandb.run.summary["best_loss"] = best_loss
                    wandb.run.summary["best_epoch"] = epoch + 1
            else:
                no_improve += 1

            if no_improve >= self.early_stopping_patience:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

        # Load best model
        self.model.load_state_dict(self.best_model_state)

        if self.use_wandb:
            wandb.finish()

        return self.model


class BlockFlowMatchingTrainer(FlowMatchingTrainer):
    """
    Extension of FlowMatchingTrainer that supports block-wise conditioning and masking.

    Variables are grouped into blocks, and all variables in a block are always
    conditioned or dropped together in the edge mask.

    Args:
        block_dict: Dictionary mapping block names (str) to lists of column indices (list[int]).
                   Example: {'demographics': [0, 1, 2], 'vitals': [3, 4], 'labs': [5, 6, 7, 8]}
        All other arguments are inherited from FlowMatchingTrainer.
    """
    def __init__(
            self,
            model,
            data,
            block_dict,
            batch_size=128,
            lr=1e-3,
            sigma_min=0.001,
            time_prior_exponent=1.0,
            inner_train_loop_size=1000,
            early_stopping_patience=20,
            device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        # Validate and store block information before calling super().__init__
        self.block_dict = block_dict
        self._validate_blocks(data)

        # Call parent constructor
        super().__init__(
            model=model,
            data=data,
            batch_size=batch_size,
            lr=lr,
            sigma_min=sigma_min,
            time_prior_exponent=time_prior_exponent,
            inner_train_loop_size=inner_train_loop_size,
            early_stopping_patience=early_stopping_patience,
            device=device
        )

        # Create block-to-indices mapping on device
        self.block_names = list(block_dict.keys())
        self.num_blocks = len(self.block_names)

        # Store indices for each block as a list of tensors
        self.block_indices = [
            torch.tensor(block_dict[name], dtype=torch.long, device=device)
            for name in self.block_names
        ]

    def _validate_blocks(self, data):
        """Validate that block_dict is well-formed and covers all columns."""
        if not isinstance(self.block_dict, dict):
            raise ValueError("block_dict must be a dictionary")

        if not self.block_dict:
            raise ValueError("block_dict cannot be empty")

        # Check that all values are lists
        for name, indices in self.block_dict.items():
            if not isinstance(indices, list):
                raise ValueError(f"Block '{name}' must map to a list of column indices")
            if not indices:
                raise ValueError(f"Block '{name}' cannot be empty")
            if not all(isinstance(i, int) for i in indices):
                raise ValueError(f"Block '{name}' must contain only integer indices")

        # Get all indices and check for overlaps
        all_indices = []
        for indices in self.block_dict.values():
            all_indices.extend(indices)

        if len(all_indices) != len(set(all_indices)):
            raise ValueError("Blocks contain overlapping column indices")

        # Check that indices match data dimensions
        num_cols = data.shape[1]
        if set(all_indices) != set(range(num_cols)):
            raise ValueError(
                f"Block indices must cover all columns [0, {num_cols-1}]. "
                f"Got: {sorted(set(all_indices))}"
            )

    def build_condition_mask(self, prob=0.333):
        """
        Build condition mask where entire blocks are conditioned together.

        Args:
            prob: Probability that each block is conditioned (not the individual variables).

        Returns:
            mask: (batch_size, num_nodes, 1) float tensor
        """
        # Initialize mask
        mask = torch.zeros((self.batch_size, self.num_nodes), dtype=torch.bool, device=self.device)

        # For each sample in batch, decide which blocks to condition
        for b in range(self.batch_size):
            # Sample which blocks are conditioned for this batch item
            block_mask = torch.bernoulli(torch.full((self.num_blocks,), prob, device=self.device)).bool()

            # Set all variables in conditioned blocks to True
            for block_idx, is_conditioned in enumerate(block_mask):
                if is_conditioned:
                    mask[b, self.block_indices[block_idx]] = True

        # Ensure not all variables are conditioned (prevent all-true)
        all_true = mask.all(dim=1, keepdim=True)
        mask = mask & ~all_true

        return mask.unsqueeze(-1).float()

    def build_edge_mask(self, dense_ratio=0.7):
        """
        Build edge mask where entire blocks are dropped together.

        Args:
            dense_ratio: Ratio of samples in batch with dense (all edges) masks.

        Returns:
            masks: (batch_size, num_nodes, num_nodes) bool tensor
        """
        n_dense = int(self.batch_size * dense_ratio)

        # Dense masks (all edges present)
        dense = torch.ones((n_dense, self.num_nodes, self.num_nodes), dtype=torch.bool)

        # Sparse masks (some blocks dropped)
        sparse = []
        for _ in range(self.batch_size - n_dense):
            m = torch.ones((self.num_nodes, self.num_nodes), dtype=torch.bool)

            # For each block, decide if it should be dropped
            for block_indices in self.block_indices:
                if torch.rand(1).item() < 0.2:  # 20% chance to drop a block
                    # Convert to CPU for indexing, then back to original device
                    block_indices_cpu = block_indices.cpu()

                    # Drop edges involving this block
                    m[block_indices_cpu, :] = False
                    m[:, block_indices_cpu] = False

                    # Ensure self-loops within the block are True (diagonal elements)
                    for idx in block_indices_cpu:
                        m[idx, idx] = True

            sparse.append(m)

        # Combine dense and sparse masks
        if sparse:
            sparse = torch.stack(sparse, dim=0)
            masks = torch.cat([dense, sparse], dim=0)
        else:
            masks = dense

        # Shuffle the masks
        indices = torch.randint(0, masks.shape[0], (self.batch_size,))
        return masks[indices].to(self.device)


