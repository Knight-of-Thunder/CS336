# config.py
model = {
    "vocab_size": 10000,
    "context_length": 256,
    "num_layers": 4,
    "num_heads": 16,
    "d_model": 512,
    "d_ff": 1344,
    "rope_theta": 10000
}

profile = {
    "warmup_steps": 5,
    "total_steps": 10,
    "batch_size" :4,
    "forward_only":False
}