import numpy as np
from engine.tensor import Tensor
from nn.transformer_block import TransformerBlock


np.random.seed(0)

def numerical_grad_check(block, x_data, eps=1e-5, n_checks=20):
    x = Tensor(x_data.copy())
    out = block(x)
    loss = out.sum()
    loss.backward()
    analytic_grad = x.grad.copy()

    flat_idx = np.random.choice(x_data.size, size=n_checks, replace=False)
    max_diff = 0.0
    for idx in flat_idx:
        idx = np.unravel_index(idx, x_data.shape)

        orig = x_data[idx]
        x_data[idx] = orig + eps
        loss_plus = block(Tensor(x_data.copy())).sum().data

        x_data[idx] = orig - eps
        loss_minus = block(Tensor(x_data.copy())).sum().data

        x_data[idx] = orig  # restore

        numeric = (loss_plus - loss_minus) / (2 * eps)
        analytic = analytic_grad[idx]
        diff = abs(numeric - analytic)
        max_diff = max(max_diff, diff)
        print(f"idx={idx}  numeric={numeric:.6f}  analytic={analytic:.6f}  diff={diff:.2e}")

    print(f"\nMax abs diff: {max_diff:.2e}  (should be well under 1e-4)")


if __name__ == "__main__":
    batch, seq_len, d_model, num_heads, d_ff = 2, 4, 8, 2, 16
    x_data = np.random.randn(batch, seq_len, d_model)
    block = TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff)

    print("=== TransformerBlock grad check (input x) ===")
    numerical_grad_check(block, x_data.copy())