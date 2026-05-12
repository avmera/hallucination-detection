"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Student task: Aggregating multiple late layers to capture the 'Truth Direction' 
    more stably than just using the final layer.
    """
    # Find the index of the last real (non-padding) token.
    real_positions = attention_mask.nonzero(as_tuple=False)  
    last_pos = int(real_positions[-1].item())                

 
    late_layers = hidden_states[-6:, last_pos, :]  # Shape: (6, hidden_dim)
    
    # Apply Mean Pooling across the layer dimension to create a stable, 
    # unified representation vector without increasing the dimensionality.
    feature = late_layers.mean(dim=0)  # Shape: (hidden_dim,)

    return feature


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Student task: Extracting geometric trajectory features (Representation Drift)
    to detect the 'hesitation' or 'instability' characteristic of hallucinations.
    """
    # Find the last real token
    real_positions = attention_mask.nonzero(as_tuple=False)
    last_pos = int(real_positions[-1].item())

    # Get the representations of the last token across ALL layers
    # Shape: (n_layers, hidden_dim)
    all_layers = hidden_states[:, last_pos, :]

    # 1. Inter-layer Cosine Similarity (Representation Drift)
    # Measures how much the representation changes from layer L to L+1.
   
    layer_t = all_layers[:-1]      # Layers 0 to N-1
    layer_t_plus_1 = all_layers[1:] # Layers 1 to N
    
    cos_sim = F.cosine_similarity(layer_t, layer_t_plus_1, dim=1) # Shape: (n_layers - 1,)

    # 2. L2 Norms of the late layers (Activation Magnitude)
    
    late_norms = torch.norm(all_layers[-6:], p=2, dim=1) # Shape: (6,)

    # 3. Layer of Maximum Change
    # The index where the model underwent the most drastic shift in representation.
    max_change_layer = torch.argmin(cos_sim).float().unsqueeze(0) # Shape: (1,)

    # 4. Sequence Length Feature

    seq_length = torch.tensor([float(last_pos)], device=hidden_states.device) # Shape: (1,)

    # Combine all geometric features into a single 1D tensor
    geo_features = torch.cat([cos_sim, late_norms, max_change_layer, seq_length], dim=0)

    return geo_features


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    
    agg_features = aggregate(hidden_states, attention_mask)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features