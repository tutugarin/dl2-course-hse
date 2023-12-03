import os
import torch

from torch import optim

from dataset import get_dataloaders
from model import LanguageModel
from train import train
from utils import get_logger, Config

logger = get_logger(__name__)


def main():
    config = Config()
    logger.info(f"| using device: {config.device}")

    tokenizer, train_dataloader, val_dataloader = get_dataloaders(config)
    logger.info(f"| {len(train_dataloader) = }")
    logger.info(f"| {len(val_dataloader) = }")

    model = LanguageModel(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )
    logger.info(f"| number of model params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if os.path.isfile("checkpoint_last.pt"):
        logger.info("| loading model state from checkpoint_last.pt")
        state = torch.load("checkpoint_last.pt", map_location='cpu')
        model.load_state_dict(state["model"])
        model = model.to(config.device)

        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        optimizer.load_state_dict(state["optim"])
    else:
        logger.info("| there is no checkpoint_last.pt to load model state")
        model = model.to(config.device)
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.lr_factor, patience=config.lr_patience)
    scaler = torch.cuda.amp.GradScaler()

    train(model, optimizer, lr_scheduler, scaler, train_dataloader, val_dataloader, config, tokenizer)

    logger.info("| generating examples...")
    for _ in range(5):
        logger.info(
            model.inference(tokenizer, "Once upon a time ", max_new_tokens=config.max_len, temperature=0.1, top_k=3)
        )


if __name__ == "__main__":
    main()
