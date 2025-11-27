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

optimizer = {
    "lr": 3e-4,
    "weight_decay": 1e-2,
    "betas": (0.9, 0.999),
    "max_norm": 1.0
}

train = {
    "batch_size": 16,
    "total_epochs": 0.5,
    "checkpoint_freq": 2000,
    "log_freq": 10,
    "val_freq": 400,
    "val_batch_size": 16,
    "val_batches": 20
}

paths = {
    "training_dataset_path": "./data/train.npy",
    "validation_dataset_path": "./data/valid.npy",
    "checkpoint_load_path": None,
    "checkpoint_save_format": "./data/model/checkpoint_{}.pt",
    "final_model_path": "./data/model/final_model.pt"
}
