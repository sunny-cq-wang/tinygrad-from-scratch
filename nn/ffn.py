from nn.Module import Module, Linear


class FeedForward(Module):
    def __init__(self, d_model, d_ff):
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)

    def forward(self, x):
        l1 = self.linear1(x)
        a1 = l1.relu()
        return self.linear2(a1)


if __name__ == "__main__":
    import numpy as np
    from engine.tensor import Tensor

    x = Tensor(np.random.randn(2, 5, 8))  # batch=2, seq_len=5, d_model=8
    ffn = FeedForward(d_model=8, d_ff=32)
    out = ffn(x)
    print(out.data.shape)