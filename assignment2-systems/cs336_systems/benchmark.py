from contextlib import nullcontext
from cs336_basics.Model.TransformerLM import TransformerLM
from cs336_basics.Training.AdamW import AdamW
from cs336_basics.Training.cross_entropy_loss import cross_entropy
import config
import torch
import timeit
import statistics


def create_random_batch(model_config, profile_config, device):
    return torch.randint(
        0,
        model_config["vocab_size"],
        (profile_config["batch_size"], model_config["context_length"]),
        device=device
    )


def create_model_with_config(model_config, device):
    return TransformerLM(
        vocab_size=model_config["vocab_size"],
        context_length=model_config["context_length"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        d_model=model_config["d_model"],
        d_ff=model_config["d_ff"],
        rope_theta=model_config["rope_theta"],
    ).to(device)


def benchmark_one(model_config, profile_config, use_bf16=None, compile=False):

    device = torch.device("cuda")

    model = create_model_with_config(model_config, device)
    if compile:
        model = torch.compile(model)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    if use_bf16 is None:
        use_bf16 = profile_config["use_bf16"]
    memory_profile = profile_config["memory_profile"]
    
    # choose autocast or nullcontext
    amp_context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if use_bf16
        else nullcontext()
    )

    # warmup
    for _ in range(profile_config["warmup_steps"]):
        x = create_random_batch(model_config, profile_config, device)
        with amp_context:
            outputs = model(x)
            loss = outputs.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()

    # Benchmark
    forward_times = []
    backward_times = []
    
    if memory_profile:
        torch.cuda.memory._record_memory_history(max_entries=1000000)
    for _ in range(profile_config["total_steps"]):
        x = create_random_batch(model_config, profile_config, device)

        # forward
        start = timeit.default_timer()
        with torch.cuda.nvtx.range("Forward"), amp_context:
            outputs = model(x)
            torch.cuda.synchronize()
        end = timeit.default_timer()
        forward_times.append(end - start)

        # backward
        optimizer.zero_grad()
        loss = outputs.mean()

        start = timeit.default_timer()
        if not profile_config["forward_only"]:
            with torch.cuda.nvtx.range("Backward"):
                loss.backward()
                torch.cuda.synchronize()
            end = timeit.default_timer()
            backward_times.append(end - start)

            optimizer.step()
    
    if memory_profile:
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
    
    # statistics
    result = {}
    result["fw_mean"] = sum(forward_times) / len(forward_times)
    result["fw_std"] = statistics.stdev(forward_times)
    if not profile_config["forward_only"]:
        result["bw_mean"] = sum(backward_times) / len(backward_times)
        result["bw_std"] = statistics.stdev(backward_times)
    
    return result


if __name__ == "__main__":
    profile_config = config.profile
    sizes = config.model_size
    torch.set_float32_matmul_precision('high')  # 或 'medium'

    for name, mcfg in sizes.items():
        print(f"\n================ MODEL: {name} ================")

        model_cfg = config.model_default.copy()
        model_cfg.update(mcfg)

        # FP32 baseline
        print("\n--- FP32 baseline ---")
        result_fp32 = benchmark_one(model_cfg, profile_config, use_bf16=False, compile=True)
        print(
            f"Forward: {result_fp32['fw_mean']:.6f} ± {result_fp32['fw_std']:.6f}"
        )
        print(
            f"Backward:{result_fp32['bw_mean']:.6f} ± {result_fp32['bw_std']:.6f}"
        )

        # BF16
        print("\n--- BF16 autocast ---")
        result_bf16 = benchmark_one(model_cfg, profile_config, use_bf16=True)
        print(
            f"Forward: {result_bf16['fw_mean']:.6f} ± {result_bf16['fw_std']:.6f}"
        )
        print(
            f"Backward:{result_bf16['bw_mean']:.6f} ± {result_bf16['bw_std']:.6f}"
        )
