from typing import Dict
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class DomainAdaptationDataset(Dataset):
    """
    Dataset for domain‐adaptation with multiple modalities.

    Each domain is represented as a dict:
      x_s_dict: {modality_name: Tensor[N_s, M_mod]}
      theta_s:  Tensor[N_s, dim_theta]
      x_t_dict: {modality_name: Tensor[N_t, M_mod]}
    Optionally, you can supply paired sim/real:
      x_sp_dict, x_tp_dict: each {modality_name: Tensor[N_p, M_mod]}

    __getitem__(i) returns:
      {
        'source': (x_s_dict_i, theta_s_i),
        'target': x_t_dict_j,
        'pair':   (x_sp_dict_k, x_tp_dict_k) or None
      }
    where i, j, k = idx % len(...)
    """
    def __init__(self,
                 x_s_dict: Dict[str, torch.Tensor],
                 theta_s:  torch.Tensor,
                 x_t_dict: Dict[str, torch.Tensor],
                 x_sp_dict: Dict[str, torch.Tensor] = None,
                 x_tp_dict: Dict[str, torch.Tensor] = None):
        self.modalities = list(x_s_dict.keys())
        # source
        N_s = theta_s.size(0)
        assert all(x_s_dict[mod].size(0) == N_s for mod in self.modalities), \
            "All source modalities must have same length as theta_s"
        # target
        N_t = next(iter(x_t_dict.values())).size(0)
        assert all(x_t_dict[mod].size(0) == N_t for mod in self.modalities), \
            "All target modalities must have same length"
        # paired
        if (x_sp_dict is not None) and (x_tp_dict is not None):
            N_p_s = next(iter(x_sp_dict.values())).size(0)
            N_p_t = next(iter(x_tp_dict.values())).size(0)
            assert N_p_s == N_p_t, "Paired source/target must have same length"
            self.x_sp = x_sp_dict
            self.x_tp = x_tp_dict
            self.N_p = N_p_s
        else:
            self.x_sp = self.x_tp = None
            self.N_p = 0

        self.x_s_dict  = x_s_dict
        self.theta_s   = theta_s
        self.x_t_dict  = x_t_dict
        # effective length
        self._len = max(N_s, N_t, self.N_p)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        # cycle through source
        i = idx % self.theta_s.size(0)
        x_s_i = {mod: self.x_s_dict[mod][i] for mod in self.modalities}
        theta_i = self.theta_s[i]

        # cycle through target
        j = idx % next(iter(self.x_t_dict.values())).size(0)
        x_t_j = {mod: self.x_t_dict[mod][j] for mod in self.modalities}

        # optional paired
        if self.N_p > 0:
            k = idx % self.N_p
            x_sp_k = {mod: self.x_sp[mod][k] for mod in self.modalities}
            x_tp_k = {mod: self.x_tp[mod][k] for mod in self.modalities}
            pair = (x_sp_k, x_tp_k)
        else:
            pair = None

        return {
            'source': (x_s_i, theta_i),
            'target':  x_t_j,
            'pair':    pair
        }


def get_train_val_loader(
        df_sim, df_real, df_sim_pair, df_real_pair, input_cols, feats_theta, modalities,
        train_frac=0.8, **kwargs):
    """
    Create train and validation DataLoaders for domain adaptation.

    Args:
        df_sim: DataFrame with simulated data.
        df_real: DataFrame with real data.
        df_sim_pair: DataFrame with paired simulated data.
        df_real_pair: DataFrame with paired real data.
        input_cols: List of input column names.
        modalities: List of modality names.

    Returns:
        train_loader, val_loader: DataLoaders for training and validation.
    """
    # 1) convert DataFrames to tensors
    x_s_dict = {mod: torch.from_numpy(df_sim[input_cols[mod]].values).float() for mod in modalities}  # simulated source data
    theta_s = torch.from_numpy(df_sim[feats_theta].values).float()  # simulated source theta
    x_t_dict = {mod: torch.from_numpy(df_real[input_cols[mod]].values).float() for mod in modalities}  # real target data
    # Pair wise data
    x_sp_dict = {mod: torch.from_numpy(df_sim_pair[input_cols[mod]].values).float() for mod in modalities}
    x_tp_dict = {mod: torch.from_numpy(df_real_pair[input_cols[mod]].values).float() for mod in modalities}

    # 2) split source
    N_s = theta_s.shape[0]
    idx_s = np.random.permutation(N_s)
    n_train_s = int(train_frac * N_s)
    train_idx_s, val_idx_s = idx_s[:n_train_s], idx_s[n_train_s:]

    # 3) split target
    N_t = next(iter(x_t_dict.values())).shape[0]
    idx_t = np.random.permutation(N_t)
    n_train_t = int(train_frac * N_t)
    train_idx_t, val_idx_t = idx_t[:n_train_t], idx_t[n_train_t:]

    # 4) (optional) split paired
    if x_sp_dict is not None:
        N_p = next(iter(x_sp_dict.values())).shape[0]
        idx_p = np.random.permutation(N_p)
        n_train_p = int(train_frac * N_p)
        train_idx_p, val_idx_p = idx_p[:n_train_p], idx_p[n_train_p:]
    else:
        train_idx_p = val_idx_p = None

    # 5) build sliced dicts
    def slice_dict(d, idx):
        return {mod: tensor[idx] for mod, tensor in d.items()}

    x_s_train = slice_dict(x_s_dict, train_idx_s)
    theta_s_train = theta_s[train_idx_s]
    x_t_train = slice_dict(x_t_dict, train_idx_t)

    x_s_val = slice_dict(x_s_dict, val_idx_s)
    theta_s_val = theta_s[val_idx_s]
    x_t_val = slice_dict(x_t_dict, val_idx_t)

    if x_sp_dict is not None:
        x_sp_train = slice_dict(x_sp_dict, train_idx_p)
        x_tp_train = slice_dict(x_tp_dict, train_idx_p)
        x_sp_val = slice_dict(x_sp_dict, val_idx_p)
        x_tp_val = slice_dict(x_tp_dict, val_idx_p)
    else:
        x_sp_train = x_tp_train = x_sp_val = x_tp_val = None

    # 6) instantiate DomainAdaptationDataset
    train_ds = DomainAdaptationDataset(
        x_s_train, theta_s_train,
        x_t_train,
        x_sp_train, x_tp_train
    )
    val_ds = DomainAdaptationDataset(
        x_s_val, theta_s_val,
        x_t_val,
        x_sp_val, x_tp_val
    )
    # 7) DataLoaders
    train_loader = DataLoader(train_ds, **kwargs)
    val_loader = DataLoader(val_ds, **kwargs)
    return train_loader, val_loader