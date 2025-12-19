import math
import torch
import triton.testing as tt
from itertools import product

from cs336_systems.pytorch_flash_attention import FlashAttention2Algorithm
from cs336_systems.triton_flash_attention import TritonFlashAttention2Algorithm

# ============================================================
# 1. PyTorch baseline
# ============================================================

# def torch_attention(Q, K, V):
#     return FlashAttention2Algorithm.apply(Q, K, V, True)

def torch_attention(Q, K, V, causal=True):
    """
    Q, K, V: (B, L, D)
    """
    d = Q.shape[-1]
    S = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d)

    if causal:
        mask = torch.triu(
            torch.ones_like(S, dtype=torch.bool), diagonal=1
        )
        S = S.masked_fill(mask, -1e9)

    P = torch.softmax(S, dim=-1)
    return torch.matmul(P, V)

# ============================================================
# 2. Triton FlashAttention wrapper
# ============================================================

def triton_attention(Q, K, V):
    return TritonFlashAttention2Algorithm.apply(Q, K, V, True)


# ============================================================
# 3. Benchmark helpers
# ============================================================

def bench_forward(fn, Q, K, V):
    return tt.do_bench(lambda: fn(Q, K, V))


def bench_backward(fn, Q, K, V):
    O = fn(Q, K, V)
    dO = torch.randn_like(O)

    def run():
        torch.autograd.grad(O, (Q, K, V), dO, retain_graph=True)

    return tt.do_bench(run)


def bench_fwd_bwd(fn, Q, K, V):
    def run():
        O = fn(Q, K, V)
        loss = O.sum()
        loss.backward()

    return tt.do_bench(run)


# ============================================================
# 4. Main benchmark loop
# ============================================================

def run_bench():
    device = "cuda"
    batch_size = 1

    seq_lens = [2**i for i in range(7, 17)]   # 128 → 65536
    dims = [2**i for i in range(4, 8)]        # 16 → 128
    dtypes = [torch.float32]

    results = []

    for L, D, dtype in product(seq_lens, dims, dtypes):
        print(f"\nRunning: L={L}, D={D}, dtype={dtype}")

        Q = torch.randn(batch_size, L, D, device=device, dtype=dtype, requires_grad=True)
        K = torch.randn(batch_size, L, D, device=device, dtype=dtype, requires_grad=True)
        V = torch.randn(batch_size, L, D, device=device, dtype=dtype, requires_grad=True)

        # ---------------- PyTorch ----------------
        try:
            t_torch_fwd = bench_forward(torch_attention, Q, K, V)
            t_torch_bwd = bench_backward(torch_attention, Q, K, V)
            t_torch_fwbw = bench_fwd_bwd(torch_attention, Q, K, V)
        except RuntimeError as e:
            print("PyTorch OOM or failure:", e)
            t_torch_fwd = t_torch_bwd = t_torch_fwbw = float("nan")

        # clear grads
        for x in (Q, K, V):
            if x.grad is not None:
                x.grad.zero_()

        # ---------------- Triton ----------------
        try:
            t_triton_fwd = bench_forward(triton_attention, Q, K, V)
            t_triton_bwd = bench_backward(triton_attention, Q, K, V)
            t_triton_fwbw = bench_fwd_bwd(triton_attention, Q, K, V)
        except RuntimeError as e:
            print("Triton OOM or failure:", e)
            t_triton_fwd = t_triton_bwd = t_triton_fwbw = float("nan")

        results.append((
            L, D, str(dtype).replace("torch.", ""),
            t_torch_fwd, t_torch_bwd, t_torch_fwbw,
            t_triton_fwd, t_triton_bwd, t_triton_fwbw
        ))

    return results


# ============================================================
# 5. Pretty print results
# ============================================================

def print_table(results):
    header = (
        "seq_len | d | dtype | "
        "torch_fwd | torch_bwd | torch_fwbw | "
        "triton_fwd | triton_bwd | triton_fwbw"
    )
    print("\n" + header)
    print("-" * len(header))

    for row in results:
        print(
            f"{row[0]:6d} | {row[1]:3d} | {row[2]:8s} | "
            f"{row[3]:9.3f} | {row[4]:9.3f} | {row[5]:10.3f} | "
            f"{row[6]:10.3f} | {row[7]:10.3f} | {row[8]:11.3f}"
        )


# ============================================================
# 6. Entry
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    results = run_bench()
    print_table(results)
