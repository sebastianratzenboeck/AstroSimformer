from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from torch.utils.data import DataLoader
from flow_base import ContextTransformer, TimeThetaEmbed, ResidualBlock, edge_mask_from_marginal
from transformer import MarginalEmbed, TransformerBlock
from geomloss import SamplesLoss

# ─────────────────────────────────────────────────────────────────────────────
# (Assume ValueEmbed, MarginalEmbed, Tokenizer, MaskedMultiheadAttention,
#  TransformerBlock, ResidualBlock, TimeThetaEmbed, and edge_mask_from_marginal
#  are imported or defined exactly as in your working code above.)
# ─────────────────────────────────────────────────────────────────────────────

class ModalityTokenizer(nn.Module):
    def __init__(
        self,
        modalities:       List[str],
        in_dims:          Dict[str,int],   # attn_embed_dims per modality
        value_dim:        int,            # after projecting each summary
        id_dim:           int,
        local_dim:        int,
        global_dim:       int,
        out_embed_dim:    int,            # common slot size E
    ):
        super().__init__()
        self.modalities = modalities
        M = len(modalities)
        # 1) project each per-mod summary into the same `value_dim`
        self.value_projs = nn.ModuleDict({
            mod: nn.Linear(in_dims[mod], value_dim)
            for mod in modalities
        })
        # 2) modality–ID embedding (M distinct tokens)
        self.id_embed = nn.Embedding(M, id_dim)
        # 3) marginal–presence embedding over slots
        # exactly like your MarginalEmbed: per-slot token + MLP global summary
        self.marg_embed = MarginalEmbed(M, dim_local=local_dim, dim_global=global_dim)
        # 4) final projection to slot size E
        total = value_dim + id_dim + local_dim + global_dim
        self.out_proj = nn.Linear(total, out_embed_dim)

    def forward(self, per_mod_embs, mod_mask):
        B = mod_mask.shape[0]
        M = len(self.modalities)
        # build value embeddings
        vals = []
        for i, mod in enumerate(self.modalities):
            v = self.value_projs[mod](per_mod_embs[mod])  # (B, value_dim)
            vals.append(v)
        vals = torch.stack(vals, dim=1)                   # (B, M, value_dim)
        # build ID embeddings
        ids = torch.arange(M, device=vals.device)         # (M,)
        ids = self.id_embed(ids).unsqueeze(0).expand(B,M,-1)  # (B, M, id_dim)
        # build marginal embeddings
        marg = self.marg_embed(mod_mask)                  # (B, M, local+global)
        # concatenate and project
        token = torch.cat([vals, ids, marg], dim=-1)  # (B, M, total_dim)
        return self.out_proj(token) # (B, M, out_embed_dim)

class ModalityTransformer(nn.Module):
    def __init__(self,
                 n_modalities, embed_dim,
                 num_heads, num_layers, widening_factor, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads,
                             widening_factor=widening_factor,
                             dropout_rate=dropout)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, slots, mod_mask):
        # slots:(B,M,E), mod_mask:(B,M)
        attn_mask = edge_mask_from_marginal(mod_mask)
        x = slots
        for blk in self.blocks:
            x = blk(x, attn_mask)
        pres = (~mod_mask).float().unsqueeze(-1)
        agg = (x * pres).sum(dim=1) / pres.sum(dim=1).clamp(min=1.0)
        # return self.out_proj(agg)
        return self.out_proj(agg)


class MultiContextFlowModel(nn.Module):
    """
    Multi‐modality flow model with per‐modality ContextTransformers,
    a modality‐level tokenizer/transformer, and a final flow network.
    """
    def __init__(self,
                 modalities: List[str],
                 num_nodes: dict,            # modality→#features
                 dim_values: dict,           # modality→dim_value for ContextTransformer
                 dim_ids: dict,              # modality→dim_id for ContextTransformer
                 dim_locals: dict,           # modality→dim_local
                 dim_globals: dict,          # modality→dim_global
                 attn_embed_dims: dict,      # modality→attention embedding size
                 num_heads: dict,            # modality→#attention heads
                 num_layers: dict,           # modality→#transformer layers
                 widening_factors: dict,     # modality→feedforward widening factor
                 dropout:      float,            # share dropout globally
                 # time+theta embedding (still global)
                 dim_theta: int,
                 theta_embed_dim: int,
                 flow_hidden_dim: int,
                 flow_depth: int,
                 final_embed_dim: int,
                 final_num_heads: int,
                 final_num_layers: int,
                 final_widening: int,
                 mod_value_dim: int = 128,
                 mod_id_dim: int = 16,  # default ID embedding size
                 mod_local_dim: int = 4,  # default local token size
                 mod_global_dim: int = 8,  # default global token size
                 ):
        super().__init__()
        self.modalities = modalities
        # 1) Per‐modality ContextTransformers
        self.context_nets = nn.ModuleDict({
            mod: ContextTransformer(
                num_nodes       = num_nodes[mod],
                dim_value       = dim_values[mod],
                dim_id          = dim_ids[mod],
                dim_local       = dim_locals[mod],
                dim_global      = dim_globals[mod],
                attn_embed_dim  = attn_embed_dims[mod],
                num_heads       = num_heads[mod],
                num_layers      = num_layers[mod],
                widening_factor = widening_factors[mod],
                dropout         = dropout
            )
            for mod in modalities
        })
        # 2) Modality‐level tokenizer & transformer
        self.mod_tokenizer = ModalityTokenizer(
            modalities    = modalities,
            in_dims       = attn_embed_dims,  # input dims per slot
            value_dim     = mod_value_dim,    # now taken from constructor
            id_dim        = mod_id_dim,
            local_dim     = mod_local_dim,
            global_dim    = mod_global_dim,
            out_embed_dim = final_embed_dim
        )
        self.mod_transformer = ModalityTransformer(
            n_modalities    = len(modalities),
            embed_dim       = final_embed_dim,
            num_heads       = final_num_heads,
            num_layers      = final_num_layers,
            widening_factor = final_widening,
            dropout         = dropout
        )

        # 3) time + θ embed
        self.time_theta_embed = TimeThetaEmbed(dim_theta, theta_embed_dim)

        # 4) final flow network
        in_dim = final_embed_dim + theta_embed_dim
        # in_dim = sum(attn_embed_dims[mod] for mod in modalities) + theta_embed_dim
        blocks = [ResidualBlock(flow_hidden_dim) for _ in range(flow_depth)]
        self.flow_net = nn.Sequential(
            nn.Linear(in_dim, flow_hidden_dim),
            nn.ReLU(),
            *blocks,
            nn.Linear(flow_hidden_dim, dim_theta)
        )

    def forward(self, t, theta_t, x_dict, node_ids_dict, mask_dict):
        # a) per‐modality summary embeddings
        per_mod_embs = {}
        mod_mask = []
        for mod in self.modalities:
            x, ids, mask = x_dict[mod], node_ids_dict[mod], mask_dict[mod]
            # Attention: entire modalities might be missing, so passing a row full of NaNs in there
            # Because ContextTransformer layers include dropout, want to keep the same per-sample dropout
            # masks for every slot in the batch to not introduce subtle train/val mismatches
            # --> we pass the entire thing in the transformer even with full NaN rows
            # emb_all: (B, E_mod) from ContextTransformer
            emb_all = self.context_nets[mod](x, ids, mask)
            # mask_all[i]==True <=> modality is 100% missing for sample i
            mask_all = mask.all(dim=1)  # (B,)
            # make a broadcastable keep‐mask
            keep = (~mask_all).unsqueeze(1).float()  # (B,1)
            # now selectively detach the “missing” rows to stop gradients from flowing when everything is NaN
            #  - rows where keep==1: emb_all stays as is (grad flows)
            #  - rows where keep==0: emb_all.detach() (grad stops here)
            emb_all = emb_all * keep + emb_all.detach() * (1 - keep)
            per_mod_embs[mod] = emb_all
            mod_mask.append(mask_all)

        mod_mask = torch.stack(mod_mask, dim=1)  # (B, #modalities)
        # b) tokenize & aggregate modalities
        slots = self.mod_tokenizer(per_mod_embs, mod_mask)  # (B, M, E)
        final_ctx = self.mod_transformer(slots, mod_mask)  # (B, E)
        
        # Delete after, this is just a test without the final transformer layer
        # final_ctx = torch.cat([per_mod_embs[mod] for mod in self.modalities], dim=1)
        
        # c) time+θ embedding
        tt_emb = self.time_theta_embed(t, theta_t)  # (B, θ_emb_dim)
        # d) predict vector field
        h = torch.cat([final_ctx, tt_emb], dim=1)
        v_pred = self.flow_net(h)  # (B, dim_theta)
        return v_pred, per_mod_embs, final_ctx

@contextmanager
def eval_mode(model: nn.Module):
    """Temporarily set model to eval(), then restore."""
    was_train = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_train)


class DomainAdaptiveTrainer:
    """
    Expects each batch as a dict:
      batch['source'] = (x_sim_dict, theta_sim)
      batch['target'] = x_real_dict
      (ignoring 'pair' for now)
    """
    def __init__(self,
                 model: MultiContextFlowModel,
                 optimizer: torch.optim.Optimizer,
                 sigma_min=0.001,
                 time_prior_exponent=1.,
                 device='cuda',
                 ot_w_final=1.0, # final‐context OT weight
                 ot_w_mod=1.0,   # per-modality OT weight
                 pair_weight=1.0,
                 warmup_steps=1000, # how many steps to ramp OT+pair from 0→full
                 max_loss=10.0,
                 clip_grad_norm=1.0,
                 # Blur annealing schedule
                 ot_blur_start=0.5, 
                 ot_blur_end=0.05, 
                 ot_blur_min=0.1,
                 **geomloss_kwargs):
        self.model = model.to(device)
        self.opt = optimizer
        self.device = device
        self.sigma_min = sigma_min
        self.tp_exp = time_prior_exponent
        self.ot_w_final = ot_w_final
        self.ot_w_mod = ot_w_mod
        self.pair_w = pair_weight
        
        # warm‐up + clamping + clipping
        self.warmup_steps   = warmup_steps
        self.max_loss       = max_loss
        self.clip_grad_norm = clip_grad_norm
        self.global_step    = 0
        
        # Use GeomLoss SamplesLoss for Sinkhorn OT
        # OT loss
        self.ot_blur_start = ot_blur_start
        self.ot_blur_end = ot_blur_end
        self.ot_blur_min = ot_blur_min

        self.ot_loss_fn = SamplesLoss(backend="online", **geomloss_kwargs)
        # 1) small per-modality normalizers
        self.layer_norm_mod = nn.ModuleDict({
            mod: nn.LayerNorm(self.model.context_nets[mod].output_layer.out_features)
            for mod in self.model.modalities
        })
        # 2) small final-context normalizer
        self.layer_norm_final = nn.LayerNorm(
            self.model.mod_transformer.out_proj.out_features
        )
        self.loss_fn = nn.MSELoss()
        self.dim_theta = model.flow_net[-1].out_features

    def _update_blur(self, epoch, num_epochs):
        # fraction of training completed
        frac = epoch / max(1, num_epochs - 1)
        # compute decay rate so that start→end over training
        decay_rate = (self.ot_blur_end / self.ot_blur_start) ** frac
        # exponential decay
        new_blur = self.ot_blur_start * decay_rate
        # clamp by minimum if desired
        self.ot_loss_fn.blur = max(new_blur, self.ot_blur_min)
        
    def sample_t(self, B):
        t = torch.rand(B, device=self.device)
        return t.pow(1/(1+self.tp_exp))

    def prepare_modality(self, x):
        """
        x: (B, M_mod) possibly with NaNs
        returns x_clean, node_ids, mask
        """
        mask = torch.isnan(x)
        x_clean = torch.nan_to_num(x, nan=0.0)
        B, M = x_clean.shape
        ids = torch.arange(M, device=self.device).unsqueeze(0).repeat(B,1)
        return x_clean, ids, mask

    def step(self, batch, is_training: bool = True):
        # 1) unpack
        x_s_dict, theta_s = batch['source']
        x_t_dict = batch['target']
        pair_batch = batch.get('pair', None)

        theta_s = theta_s.to(self.device)
        B, D = theta_s.shape

        # 2) build (t, theta_t, true_v) for flow‐matching
        t = self.sample_t(B)
        theta0 = torch.randn_like(theta_s)
        sigma_t = 1 - (1 - self.sigma_min) * t
        theta_t = theta0 * sigma_t.unsqueeze(1) + theta_s * t.unsqueeze(1)
        true_v = theta_s - (1 - self.sigma_min) * theta0

        # 3) prepare each modality for sim & real
        sim_clean, sim_ids, sim_mask = {},{},{}
        real_clean, real_ids, real_mask = {},{},{}
        for mod in self.model.modalities:
            sim_clean[mod], sim_ids[mod], sim_mask[mod] = self.prepare_modality(x_s_dict[mod].to(self.device))
            real_clean[mod], real_ids[mod], real_mask[mod] = self.prepare_modality(x_t_dict[mod].to(self.device))

        # 4) forward‐flow on sim batch
        v_pred, sim_embs, sim_final = self.model(t, theta_t, sim_clean, sim_ids, sim_mask)
        loss_flow = self.loss_fn(v_pred, true_v)

        # -------- DOMAIN ALIGNMENT ---------------
        # 5) domain‐alignment OT loss between sim vs. real joint contexts
        # get per‐modality real embeddings
        with torch.no_grad():
            _, real_embs, real_final = self.model(t, theta_t, real_clean, real_ids, real_mask)
            
        # — now **clone & normalize** copies just for OT —
        sim_embs_norm = {
            mod: self.layer_norm_mod[mod](sim_embs[mod])
            for mod in self.model.modalities
        }
        real_embs_norm = {
            mod: self.layer_norm_mod[mod](real_embs[mod])
            for mod in self.model.modalities
        }
        sim_final_norm = self.layer_norm_final(sim_final)
        real_final_norm = self.layer_norm_final(real_final)
    
        # 6) OT losses (per-modality + final)
        # — compute Sinkhorn OT on the **normalized** vectors only —
        loss_mod_ot = 0.0
        active_mods = 0
        for mod in self.model.modalities:
            # boolean masks of shape (B,) indicating which samples actually have this modality
            present_sim = ~sim_mask[mod].all(dim=1)
            present_real = ~real_mask[mod].all(dim=1)
            if present_sim.any() and present_real.any():
                s = sim_embs_norm[mod][present_sim]
                r = real_embs_norm[mod][present_real]
                loss_mod_ot += self.ot_loss_fn(s, r)
                active_mods += 1
        loss_mod_ot = loss_mod_ot / active_mods
        # final‐context OT
        loss_final_ot = self.ot_loss_fn(sim_final_norm, real_final_norm)
        # loss_final_ot = loss_final_ot.clamp(max=self.max_loss)
        loss_final_ot = torch.log1p(loss_final_ot)

        # 7) Pairwise *final-level* constraint (only if provided)
        loss_pair = 0.0
        if pair_batch is not None:
            x_sp, x_tp = pair_batch
            # prepare the *paired* inputs
            sp_clean, sp_ids, sp_mask = {},{},{}
            tp_clean, tp_ids, tp_mask = {},{},{}
            for mod in self.model.modalities:
                sp_clean[mod], sp_ids[mod], sp_mask[mod] = self.prepare_modality(x_sp[mod].to(self.device))
                tp_clean[mod], tp_ids[mod], tp_mask[mod] = self.prepare_modality(x_tp[mod].to(self.device))
            # **without** no_grad so gradients flow
            _, _, sp_final = self.model(t, theta_t, sp_clean, sp_ids, sp_mask)
            _, _, tp_final = self.model(t, theta_t, tp_clean, tp_ids, tp_mask)
            # Pair loss currently only on final context embedding
            # loss_pair = F.mse_loss(sp_final, tp_final).clamp(max=self.max_loss)
            cos_sim = F.cosine_similarity(sp_final, tp_final, dim=-1)   # (B,)
            loss_pair = (1.0 - cos_sim).mean()                         # scalar
            # then still clamp if you like:
            # loss_pair = loss_pair.clamp(max=self.max_loss)
            loss_pair = torch.log1p(loss_pair)   # Soft‐clamping of loss -> still differentiable

        # 8) Combine and step
        # total_loss = loss_flow + self.ot_w_mod * loss_mod_ot + self.ot_w_final * loss_final_ot + self.pair_w * loss_pair
        # total_loss = loss_flow + self.ot_w_final * loss_final_ot + self.pair_w * loss_pair
        
        # 7) ramp weights from 0→full over warmup_steps
        ramp = min(1.0, self.global_step / float(self.warmup_steps))
        w_ot = ramp * self.ot_w_final
        w_pair = ramp * self.pair_w

        total_loss = loss_flow + w_ot * loss_final_ot + w_pair * loss_pair

        # 9) only update in training mode
        if is_training:
            self.opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.opt.step()
            self.global_step += 1

        return {
            'total':     total_loss.item(),
            'flow':      loss_flow.item(),
            'ot_final':  loss_final_ot.item(),
            'pair':      loss_pair.item(),
            'ramp':      ramp,
        }
    
    def train_epoch(self, train_loader: DataLoader, epoch, num_epochs):
        self._update_blur(epoch, num_epochs)
        self.model.train()
        # stats = {'total': 0., 'flow': 0.} #, 'ot_mod': 0., 'pair': 0}
        stats = {'total': 0., 'flow': 0., 'ot_final': 0., 'pair': 0, 'ramp': 0}
        n = 0
        for batch in train_loader:
            out = self.step(batch, is_training=True)
            for k in stats:
                stats[k] += out[k]
            n += 1
        return {k: stats[k]/n for k in stats}

    def eval_epoch(self, val_loader: DataLoader):
        """Run a validation pass over val_loader."""
        self.model.eval()
        # stats = {'total': 0., 'flow': 0.} #, 'ot_mod': 0., 'pair': 0}
        stats = {'total': 0., 'flow': 0., 'ot_final': 0., 'pair': 0, 'ramp': 0}
        n = 0
        with torch.no_grad():
            for batch in val_loader:
                out = self.step(batch, is_training=False)
                for k in stats:
                    stats[k] += out[k]
                n += 1
        # Restore train mode (in case someone calls train again)
        self.model.train()
        return {k: stats[k]/n for k in stats}