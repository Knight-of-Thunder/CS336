import timeit
import torch
from cs336_basics.Model.scaled_dot_product_attention import scaled_dot_product_attention

torch.cuda.set_device(0)

batch = 8
d_models = [16, 32, 64, 128]
seqs = [256, 1024, 4096, 8192, 16384]


def run_once(seq_len, d_model):
    print(f"\n=== seq={seq_len}, d={d_model} ===")

    Q = torch.randn(batch, seq_len, d_model, device="cuda", requires_grad=True)
    K = torch.randn(batch, seq_len, d_model, device="cuda", requires_grad=True)
    V = torch.randn(batch, seq_len, d_model, device="cuda", requires_grad=True)

    # ---- warmup ----
    for _ in range(10):
        o = scaled_dot_product_attention(Q, K, V)
        l = o.sum()
        l.backward()
        torch.cuda.synchronize()
        Q.grad = K.grad = V.grad = None

    # ---- forward 100 times ----
    forward_times = []
    torch.cuda.synchronize()
    for _ in range(100):
        t0 = timeit.default_timer()
        o = scaled_dot_product_attention(Q, K, V)
        torch.cuda.synchronize()
        t1 = timeit.default_timer()
        forward_times.append((t1 - t0) * 1000)

    # ---- memory before backward ----
    mem_before = torch.cuda.memory_allocated()

    # ---- backward 100 times ----
    backward_times = []
    torch.cuda.synchronize()
    for _ in range(100):
        o = scaled_dot_product_attention(Q, K, V)
        l = o.sum()
        t0 = timeit.default_timer()
        l.backward()
        torch.cuda.synchronize()
        t1 = timeit.default_timer()
        backward_times.append((t1 - t0) * 1000)
        Q.grad = K.grad = V.grad = None

    # ---- output ----
    print(f"forward avg: {sum(forward_times)/100:.3f} ms")
    print(f"backward avg: {sum(backward_times)/100:.3f} ms")
    print(f"mem before backward: {mem_before / 1024**2:.2f} MB")

    return forward_times, backward_times


# ---- run all ----
for d in d_models:
    for L in seqs:
        try:
            run_once(L, d)
        except RuntimeError as e:
            print(f"OOM at seq={L}, d={d}: {e}")
