from dataclasses import dataclass, fields


@dataclass
class SDEConfig:
    """
    SDEConfig
    Holds parameters for stochastic differential equations (SDEs):
        T: Total time
        T_min: Minimum time
        sigma_min: Minimum value for sigma
        sigma_max: Maximum value for sigma
    """
    T: float = 0.0
    T_min: float = 0.0
    sigma_min: float = 0.0
    sigma_max: float = 0.0

@dataclass
class SimformerConfig:
    """
    SimformerConfig
    Contains parameters for configuring a "Simformer" component (likely a variant of Transformer or a custom architecture):
        dim_value: Dimension for value embeddings
        dim_id: Dimension for identifier embeddings
        dim_condition: Dimension for conditional inputs
    """
    dim_value: int = 0
    dim_id: int = 0
    dim_condition: int = 0

@dataclass
class TransformerConfig:
    """
    TransformerConfig
    Stores parameters for Transformer-based architecture configurations:
        num_heads: Number of attention heads
        num_layers: Number of Transformer layers
        attn_size: Size of attention vectors
        widening_factor: Factor to widen feedforward layers
    """
    num_heads: int = 0
    num_layers: int = 0
    attn_size: int = 0
    widening_factor: int = 0

@dataclass
class TrainingConfig:
    """
    TrainingConfig
    Defines training-related parameters:
        epochs: Number of training epochs
    """
    epochs: int = 0

@dataclass
class DiffusionConfig:
    """
    DiffusionConfig
    Holds parameters for a diffusion process, typically in diffusion models:
        time_steps: Number of time steps
    """
    time_steps: int = 0


# Function to set the values in each configuration
def set_config_values(config, values):
    for field in fields(config):
        if field.name in values:
            setattr(config, field.name, values[field.name])