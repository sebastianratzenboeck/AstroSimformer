import torch
import torch.optim as optim

class FlowMatchingTrainer:
    def __init__(
        self,
        model,
        data,
        batch_size=128,
        lr=1e-3,
        sigma_min=0.001,
        time_prior_exponent=1.0,
        inner_train_loop_size=1000,
        early_stopping_patience=20,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.data = torch.tensor(data, dtype=torch.float32).to(device)
        self.batch_size = batch_size
        self.lr = lr
        self.sigma_min = sigma_min
        self.time_prior_exponent = time_prior_exponent
        self.inner_train_loop_size = inner_train_loop_size
        self.early_stopping_patience = early_stopping_patience
        self.device = device

        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.best_model_state = None

        self.num_nodes = self.data.shape[1]
        self.node_ids = torch.arange(self.num_nodes).unsqueeze(0).repeat(self.batch_size, 1).to(device)

    def sample_batch(self):
        idx = torch.randint(0, self.data.shape[0], (self.batch_size,))
        return self.data[idx]

    def build_condition_mask(self, prob=0.333):
        mask = torch.bernoulli(torch.full((self.batch_size, self.num_nodes), prob)).bool().to(self.device)
        all_true = mask.all(dim=1, keepdim=True)
        mask = mask & ~all_true  # prevent all-true
        return mask.unsqueeze(-1).float()
        # Just for testing, return a zero mask
        # return torch.zeros((self.batch_size, self.num_nodes, 1), dtype=torch.float32).to(self.device)

    def build_edge_mask(self, dense_ratio=0.7):
        n_dense = int(self.batch_size * dense_ratio)
        dense = torch.ones((n_dense, self.num_nodes, self.num_nodes), dtype=torch.bool)
        sparse = []
        for _ in range(self.batch_size - n_dense):
            m = torch.ones((self.num_nodes, self.num_nodes), dtype=torch.bool)
            drop = torch.rand(self.num_nodes) < 0.2
            m[drop] = False
            m[:, drop] = False
            m[drop, drop] = True  # ensure diagonal is True
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

    def flow_matching_loss(self, x0, x1, node_ids, condition_mask, edge_mask):
        B, N = x0.shape
        t = self.sample_t(B).reshape(B, 1, 1)  #, x1.unsqueeze(-1).shape)

        xt = (1 - (1 - self.sigma_min) * t) * x0.unsqueeze(-1) + t * x1.unsqueeze(-1)
        xt = torch.where(condition_mask.bool(), x1.unsqueeze(-1), xt)

        pred_velocity = self.model(t, xt, node_ids, condition_mask, edge_mask)
        velocity = x1 - (1 - self.sigma_min) * x0

        loss_mask = condition_mask.bool()
        diff = (pred_velocity.squeeze(-1) - velocity) ** 2
        diff = torch.where(loss_mask.squeeze(-1), 0.0, diff)

        num_elements = torch.sum(~loss_mask, axis=-2, keepdim=True)
        diff = torch.where(num_elements > 0, diff / num_elements, 0.0)
        # Mean over batch
        loss = torch.mean(diff)
        return loss

    def fit(self, epochs=100):
        best_loss = float("inf")
        no_improve = 0

        for epoch in range(epochs):
            total_loss = 0.0
            for _ in range(self.inner_train_loop_size):
                x1 = self.sample_batch()
                x0 = torch.randn_like(x1)
                loss_mask = torch.isnan(x1)
                # x1_clean = torch.nan_to_num(x1, nan=0.0)

                condition_mask = self.build_condition_mask()
                edge_mask = self.build_edge_mask(dense_ratio=0.6)

                loss = self.flow_matching_loss(x0, x1, self.node_ids, condition_mask, edge_mask)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() / self.inner_train_loop_size

            print(f"Epoch {epoch + 1}: loss = {total_loss:.6f}")

            if total_loss < best_loss:
                best_loss = total_loss
                no_improve = 0
                self.best_model_state = self.model.state_dict()
            else:
                no_improve += 1

            if no_improve >= self.early_stopping_patience:
                print("Early stopping triggered.")
                break

        self.model.load_state_dict(self.best_model_state)
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


