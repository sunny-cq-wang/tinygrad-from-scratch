import numpy as np
from engine.tensor import Tensor
from transformer.self_attention import MultiHeadAttention, causal_mask


np.random.seed(0)

def numerical_grad_check(mha, x_data, mask=None, eps=1e-5, n_checks=20):
    x = Tensor(x_data.copy())
    out, _ = mha(x, mask=mask)
    loss = out.sum()
    loss.backward()
    analytic_grad = x.grad.copy()

    flat_idx = np.random.choice(x_data.size, size=n_checks, replace=False)
    max_diff = 0.0
    for idx in flat_idx:
        idx = np.unravel_index(idx, x_data.shape)

        orig = x_data[idx]
        x_data[idx] = orig + eps
        out_plus, _ = mha(Tensor(x_data.copy()), mask=mask)
        loss_plus = out_plus.sum().data

        x_data[idx] = orig - eps
        out_minus, _ = mha(Tensor(x_data.copy()), mask=mask)
        loss_minus = out_minus.sum().data

        x_data[idx] = orig  # restore

        numeric = (loss_plus - loss_minus) / (2 * eps)
        analytic = analytic_grad[idx]
        diff = abs(numeric - analytic)
        max_diff = max(max_diff, diff)
        print(f"idx={idx}  numeric={numeric:.6f}  analytic={analytic:.6f}  diff={diff:.2e}")

    print(f"\nMax abs diff: {max_diff:.2e}  (should be well under 1e-4)")


if __name__ == "__main__":
    batch, seq_len, d_model, num_heads = 2, 4, 8, 2
    x_data = np.random.randn(batch, seq_len, d_model)
    mha = MultiHeadAttention(d_model, num_heads)
    mask = causal_mask(seq_len)

    print("=== No mask ===")
    numerical_grad_check(mha, x_data.copy(), mask=None)

    print("\n=== Causal mask ===")
    numerical_grad_check(mha, x_data.copy(), mask=mask)