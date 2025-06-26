import torch
import torch.optim as optim

class FlowMatchingTrainer:
    def __init__(
        self,
        model,
        data,
        batch_size=128,
        lr=1e-3,
        inner_train_loop_size=1000,
        early_stopping_patience=20,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.data = torch.tensor(data, dtype=torch.float32).to(device)
        self.batch_size = batch_size
        self.lr = lr
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

    def build_edge_mask(self, dense_ratio=0.8):
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

    def flow_matching_loss(self, x0, x1, node_ids, condition_mask, edge_mask):
        B, N = x0.shape
        t = torch.rand((B, 1, 1), device=self.device)
        xt = (1 - t) * x0.unsqueeze(-1) + t * x1.unsqueeze(-1)
        xt = torch.where(condition_mask.bool(), x1.unsqueeze(-1), xt)

        pred_velocity = self.model(t, xt, node_ids, condition_mask, edge_mask)
        velocity = x1 - x0

        loss_mask = condition_mask.bool()
        diff = (pred_velocity.squeeze(-1) - velocity) ** 2
        diff = torch.where(loss_mask.squeeze(-1), 0.0, diff)

        return diff.sum() / max((~loss_mask).sum(), 1)

    def fit(self, epochs=100):
        best_loss = float("inf")
        no_improve = 0

        for epoch in range(epochs):
            total_loss = 0.0
            for _ in range(self.inner_train_loop_size):
                x1 = self.sample_batch()
                x0 = torch.randn_like(x1)
                loss_mask = torch.isnan(x1)
                x1_clean = torch.nan_to_num(x1, nan=0.0)

                condition_mask = self.build_condition_mask()
                edge_mask = self.build_edge_mask(dense_ratio=0.8)

                loss = self.flow_matching_loss(x0, x1_clean, self.node_ids, condition_mask, edge_mask)

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