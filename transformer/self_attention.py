import math
import numpy as np
from engine.tensor import Tensor


def softmax(x, axis=-1):
    shifted = x - x.max(axis=axis, keepdims=True)
    exp = shifted.exp()
    return exp / exp.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.data.shape[-1]
    scores = (Q @ K.transpose(0, 1, 3, 2)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores + mask  # mask is a plain numpy array, broadcasts fine

    attn = softmax(scores, axis=-1)
    out = attn @ V
    return out, attn


def causal_mask(seq_len):
    # upper-triangular, excluding the diagonal, filled with a large negative number
    mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)
    return mask  # broadcasts against (batch, num_heads, seq_len, seq_len)


def split_heads(x, num_heads):
    batch, seq_len, d_model = x.data.shape
    d_k = d_model // num_heads
    x = x.reshape(batch, seq_len, num_heads, d_k)
    return x.transpose(0, 2, 1, 3)  # (batch, num_heads, seq_len, d_k)


def combine_heads(x):
    batch, num_heads, seq_len, d_k = x.data.shape
    x = x.transpose(0, 2, 1, 3)          # (batch, seq_len, num_heads, d_k)
    return x.reshape(batch, seq_len, num_heads * d_k)


class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_model = d_model
        scale = 1 / math.sqrt(d_model)
        self.W_q = Tensor(np.random.randn(d_model, d_model) * scale)
        self.W_k = Tensor(np.random.randn(d_model, d_model) * scale)
        self.W_v = Tensor(np.random.randn(d_model, d_model) * scale)
        self.W_o = Tensor(np.random.randn(d_model, d_model) * scale)

    def __call__(self, x, mask=None):
        Q = split_heads(x @ self.W_q, self.num_heads)
        K = split_heads(x @ self.W_k, self.num_heads)
        V = split_heads(x @ self.W_v, self.num_heads)

        out, attn = scaled_dot_product_attention(Q, K, V, mask=mask)
        out = combine_heads(out)
        return out @ self.W_o, attn