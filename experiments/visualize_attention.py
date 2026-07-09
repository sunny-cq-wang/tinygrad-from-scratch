import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from engine.tensor import Tensor
from transformer.self_attention import MultiHeadAttention, causal_mask


def plot_attention(attn, batch_idx=0, title_prefix=""):
    num_heads = attn.data.shape[1]
    fig, axes = plt.subplots(1, num_heads, figsize=(4 * num_heads, 4))
    if num_heads == 1:
        axes = [axes]
    for h in range(num_heads):
        ax = axes[h]
        im = ax.imshow(attn.data[batch_idx, h], cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"{title_prefix}Head {h}")
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        ax.set_xticks(range(attn.data.shape[-1]))
        ax.set_yticks(range(attn.data.shape[-1]))
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    np.random.seed(0)

    batch, seq_len, d_model, num_heads = 1, 6, 8, 2
    x_data = np.random.randn(batch, seq_len, d_model)
    x = Tensor(x_data)

    mha = MultiHeadAttention(d_model, num_heads)

    # --- No mask ---
    out, attn_no_mask = mha(x, mask=None)
    print("=== No mask, head 0, query row sums (should all be 1.0) ===")
    print(attn_no_mask.data[0, 0].sum(axis=-1))
    plot_attention(attn_no_mask, title_prefix="No mask - ")

    # --- Causal mask ---
    mask = causal_mask(seq_len)
    out, attn_causal = mha(x, mask=mask)
    print("\n=== Causal mask, head 0 matrix (upper-right triangle should be ~0) ===")
    print(np.round(attn_causal.data[0, 0], 3))
    plot_attention(attn_causal, title_prefix="Causal - ")

    # --- Sanity check on the mask boundary itself (tiny 3x3 example) ---
    print("\n=== 3x3 causal_mask raw values (k=1 boundary check) ===")
    print(causal_mask(3))