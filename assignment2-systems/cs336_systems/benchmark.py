# from cs336_basics.Model.TransformerLM import TransformerLM
# from cs336_basics.Training.AdamW import AdamW
# from cs336_basics.Training.cross_entropy_loss import cross_entropy
# import config
# import torch
# import timeit
# import statistics 
# import torch.cuda.nvtx as nvtx

# model_config = config.model_default.copy()
# model_config.update(config.model_size["large"])
# profile_config = config.profile

# def create_random_batch(device=None):
#     if device is None:
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     return torch.randint(0, model_config["vocab_size"], \
#                          (profile_config["batch_size"], model_config["context_length"]), device=device)

# def create_model_with_config():
#     model = TransformerLM(
#         vocab_size=model_config["vocab_size"],
#         context_length=model_config["context_length"],
#         num_layers=model_config["num_layers"],
#         num_heads=model_config["num_heads"],
#         d_model=model_config["d_model"],
#         d_ff=model_config["d_ff"],
#         rope_theta=model_config["rope_theta"],
#         # device=device, all in cuda device by default
#     )
#     return model

# def bench_mark():
#     model = create_model_with_config()
#     optimizer = AdamW(model.parameters())
#     for group in optimizer.param_groups:
#         group['lr'] = 1e-4
    
#     # Warmup
#     for i in range(profile_config["warmup_steps"]):
#         random_batch = create_random_batch()
#         # Forward
#         outputs = model(random_batch)
#         if not profile_config["forward_only"]:
#             # Backward
#             optimizer.zero_grad()
#             loss = outputs.mean()
#             loss.backward()
#             optimizer.step()
    
#     torch.cuda.synchronize()

#     # Benchmark
#     forward_times = []
#     backward_times = []
#     for _ in range(profile_config["total_steps"]):
#         random_batch = create_random_batch()
#         # Forward
#         start_time = timeit.default_timer()
#         with torch.cuda.nvtx.range("Forward Pass"):
#             outputs = model(random_batch)
#             torch.cuda.synchronize()
#         end_time = timeit.default_timer()
#         forward_times.append(end_time - start_time)

#         if not profile_config["forward_only"]:
#             # Backward
#             optimizer.zero_grad()
#             loss = outputs.mean()
#             start_time = timeit.default_timer()
#             with torch.cuda.nvtx.range("Backward Pass"):
#                 loss.backward()
#                 torch.cuda.synchronize()
#             end_time = timeit.default_timer()
#             backward_times.append(end_time - start_time)
#             optimizer.step()

#     # Calculate statistics    
#     avg_forward_time = sum(forward_times) / len(forward_times)
#     std_forward_time = statistics.stdev(forward_times) if len(forward_times) > 1 else 0.0
#     print(f"Average Forward Time per step: {avg_forward_time:.6f} seconds ± {std_forward_time:.6f}")
#     if not profile_config["forward_only"]:
#         avg_backward_time = sum(backward_times) / len(backward_times)
#         std_backward_time = statistics.stdev(backward_times) if len(backward_times) > 1 else 0.0
#         print(f"Average Backward Time per step: {avg_backward_time:.6f} seconds ± {std_backward_time:.6f}")

# if __name__ == "__main__":
#     bench_mark()


from contextlib import nullcontext
from cs336_basics.Model.TransformerLM import TransformerLM
from cs336_basics.Training.AdamW import AdamW
from cs336_basics.Training.cross_entropy_loss import cross_entropy
import config
import torch
import timeit
import statistics
import torch.cuda.nvtx as nvtx


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


def benchmark_one(model_config, profile_config, use_bf16=False):

    device = torch.device("cuda")

    model = create_model_with_config(model_config, device)
    optimizer = AdamW(model.parameters(), lr=1e-4)

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
        with torch.cuda.nvtx.range("Backward"):
            loss.backward()
            torch.cuda.synchronize()
        end = timeit.default_timer()
        backward_times.append(end - start)

        optimizer.step()

    # statistics
    return {
        "fw_mean": sum(forward_times) / len(forward_times),
        "fw_std": statistics.stdev(forward_times),
        "bw_mean": sum(backward_times) / len(backward_times),
        "bw_std": statistics.stdev(backward_times),
    }


if __name__ == "__main__":
    profile_config = config.profile
    sizes = config.model_size

    for name, mcfg in sizes.items():
        print(f"\n================ MODEL: {name} ================")

        model_cfg = config.model_default.copy()
        model_cfg.update(mcfg)

        # FP32 baseline
        print("\n--- FP32 baseline ---")
        result_fp32 = benchmark_one(model_cfg, profile_config, use_bf16=False)
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
