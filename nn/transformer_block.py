from nn.Module import Module, Linear
from nn.LayerNorm import LayerNorm
from transformer.self_attention import MultiHeadAttention  # adjust import path to wherever yours lives
from nn.ffn import FeedForward


class TransformerBlock(Module):
    def __init__(self, d_model, num_heads, d_ff):
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x, mask=None):
        x1 = self.norm1(x + self.attn(x)[0])
        return self.norm2(x1 + self.ffn(x1))


if __name__ == "__main__":
    import numpy as np
    from engine.tensor import Tensor

    x = Tensor(np.random.randn(2, 5, 8))
    block = TransformerBlock(d_model=8, num_heads=2, d_ff=32)
    out = block(x)
    print(out.data.shape)
    out.sum().backward()
    print(block.attn.W_q.grad.shape)  # should be nonzero now that MHA inherits Module