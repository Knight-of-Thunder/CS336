import timeit
import torch
import pandas as pd
from cs336_basics.Model.scaled_dot_product_attention import scaled_dot_product_attention

torch.cuda.set_device(0)

batch = 8
d_models = [16, 32, 64, 128]
seqs = [256, 1024, 4096, 8192, 16384]

results = []

COMPILE = True
if COMPILE:
    self_attention = torch.compile(scaled_dot_product_attention)
else:
    self_attention = scaled_dot_product_attention

def run_once(seq_len, d_model):

    print(f"\n=== seq={seq_len}, d={d_model} ===")
    try:
        Q = torch.randn(batch, seq_len, d_model, device="cuda", requires_grad=True)
        K = torch.randn(batch, seq_len, d_model, device="cuda", requires_grad=True)
        V = torch.randn(batch, seq_len, d_model, device="cuda", requires_grad=True)

        # ---- warmup ----
        for _ in range(10):
            o = self_attention(Q, K, V)
            l = o.sum()
            l.backward()
            torch.cuda.synchronize()
            Q.grad = K.grad = V.grad = None

        # ---- forward 100 times ----
        forward_times = []
        torch.cuda.synchronize()
        for _ in range(100):
            t0 = timeit.default_timer()
            o = self_attention(Q, K, V)
            torch.cuda.synchronize()
            t1 = timeit.default_timer()
            forward_times.append((t1 - t0) * 1000)

        avg_fwd = sum(forward_times) / 100

        # ---- memory before backward ----
        mem_before = torch.cuda.memory_allocated() / (1024**2)  # MB

        # ---- backward 100 times ----
        backward_times = []
        torch.cuda.synchronize()
        for _ in range(100):
            o = self_attention(Q, K, V)
            l = o.sum()
            t0 = timeit.default_timer()
            l.backward()
            torch.cuda.synchronize()
            t1 = timeit.default_timer()
            backward_times.append((t1 - t0) * 1000)
            Q.grad = K.grad = V.grad = None

        avg_bwd = sum(backward_times) / 100

        print(f"forward avg: {avg_fwd:.3f} ms")
        print(f"backward avg: {avg_bwd:.3f} ms")
        print(f"mem before backward: {mem_before:.2f} MB")

        return {
            "d_model": d_model,
            "seq_len": seq_len,
            "forward_ms": round(avg_fwd, 3),
            "backward_ms": round(avg_bwd, 3),
            "mem_mb": round(mem_before, 2),
            "oom": False,
            "error": None
        }

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"OOM at seq={seq_len}, d={d_model}")
            return {
                "d_model": d_model,
                "seq_len": seq_len,
                "forward_ms": None,
                "backward_ms": None,
                "mem_mb": None,
                "oom": True,
                "error": "OOM"
            }
        else:
            raise e
    finally:
        torch.cuda.empty_cache()

# ---- run all ----
for d in d_models:
    for L in seqs:
        result = run_once(L, d)
        results.append(result)

# ---- create DataFrame ----
df = pd.DataFrame(results)

# ---- print as Markdown table ----
print("\n" + "="*80)
print("Benchmark Results (Markdown Table)")
print("="*80)

markdown_table = df.to_markdown(index=False, tablefmt="github")
print(markdown_table)

