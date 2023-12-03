import os
import torch
import logging
from dataclasses import dataclass

logging.basicConfig(
    filename="training_logs",
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def get_logger(name):
    logger = logging.getLogger(name)
    return logger


def save_model(model, optimizer, loss, name, train_losses, val_losses, logger):
    filename = f"checkpoint_{name}.pt"
    logger.info(f"| saving {filename}...")
    torch.save(
        {
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "loss": loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
        },
        filename,
    )
    logger.info(f"| saving {filename} finished")


@dataclass
class Config:
    device = "cpu" if (cuda_index := os.getenv("CUDA_DEVICE")) is None else f"cuda:{cuda_index}"
    data_folder = "data"

    num_epoches = 100
    validate_freq = 1000
    vocab_size = 5120
    max_len = 256
    bsz = 512

    embed_dim = 512
    hidden_dim = 2048
    num_heads = 8
    num_layers = 6
    dropout = 0.1

    lr = 3e-4
    weight_decay = 0.0
    lr_patience = 10
    lr_factor = 0.3
