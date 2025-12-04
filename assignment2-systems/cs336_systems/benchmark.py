from cs336_basics.Model.TransformerLM import TransformerLM
from cs336_basics.Training.AdamW import AdamW
from cs336_basics.Training.cross_entropy_loss import cross_entropy
import config
import torch
import timeit



def create_random_batch(device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.randint(0, config.model["vocab_size"], \
                         (config.profile["batch_size"], config.model["context_length"]), device=device)

def create_model_with_config():
    model = TransformerLM(
        vocab_size=config.model["vocab_size"],
        context_length=config.model["context_length"],
        num_layers=config.model["num_layers"],
        num_heads=config.model["num_heads"],
        d_model=config.model["d_model"],
        d_ff=config.model["d_ff"],
        rope_theta=config.model["rope_theta"],
        # device=device, all in cuda device by default
    )
    return model

def bench_mark():
    model = create_model_with_config()
    optimizer = AdamW(model.parameters())
    for group in optimizer.param_groups:
        group['lr'] = 1e-4

    for i in range(config.profile["warmup_steps"]):
        random_batch = create_random_batch()
        # Forward
        outputs = model(random_batch)
        if not config.profile["forward_only"]:
            # Backward
            optimizer.zero_grad()
            loss = outputs.mean()
            loss.backward()
            optimizer.step()
    
    torch.cuda.synchronize()

    forward_times = []
    backward_times = []
    for _ in range(config.profile["total_steps"]):
        random_batch = create_random_batch()
        # Forward
        start_time = timeit.default_timer()
        outputs = model(random_batch)
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        forward_times.append(end_time - start_time)

        if not config.profile["forward_only"]:
            # Backward
            optimizer.zero_grad()
            loss = outputs.mean()
            start_time = timeit.default_timer()
            loss.backward()
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            backward_times.append(end_time - start_time)
            optimizer.step()
        
    avg_forward_time = sum(forward_times) / len(forward_times)
    print(f"Average Forward Time per step: {avg_forward_time:.6f} seconds")
    if not config.profile["forward_only"]:
        avg_backward_time = sum(backward_times) / len(backward_times)
        print(f"Average Backward Time per step: {avg_backward_time:.6f} seconds")

if __name__ == "__main__":
    bench_mark()
