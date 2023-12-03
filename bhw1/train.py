import torch

from typing import Optional, Any
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from time import perf_counter


from model import LanguageModel
from dataset import Tokenizer, create_mask
from utils import get_logger, save_model, Config

logger = get_logger(__name__)


@torch.no_grad()
def _validate(
    model: LanguageModel,
    criterion: nn.Module,
    loader: DataLoader,
):
    device = next(model.parameters()).device
    val_loss = 0.0

    model.eval()
    for indices, targets in loader:
        mask, _ = create_mask(indices)
        indices, mask, targets = indices.to(device), mask.to(device), targets.to(device)
        logits = model(indices, mask)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        val_loss += loss.item() * indices.shape[0]

    val_loss /= len(loader.dataset)
    return val_loss


def validate_step(
    model: LanguageModel,
    criterion: nn.Module,
    loader: DataLoader,
    config: Config,
    tokenizer: Tokenizer,
):
    model.eval()
    val_loss = _validate(model, criterion, loader)
    start_generation = perf_counter()
    logger.info(f'| generation example:')
    logger.info(
        model.inference(tokenizer, "Once upon a time ", max_new_tokens=config.max_len, temperature=0.1, top_k=3)
    )
    logger.info(f"| generation 1 sample total time: {perf_counter() - start_generation}")
    return val_loss


def train_step(
    indices,
    targets,
    criterion: nn.CrossEntropyLoss,
    model: LanguageModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
):
    model.train()
    device = next(model.parameters()).device

    optimizer.zero_grad()
    mask, _ = create_mask(indices)
    indices, mask, targets = indices.to(device), mask.to(device), targets.to(device)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        logits = model(indices, mask)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
    scaler.scale(loss).backward()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    return loss.item()


def train_epoch(
    model: LanguageModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: torch.cuda.amp.GradScaler,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config,
    tokenizer: Tokenizer,
    best_loss: float,
    train_losses: list[float],
    val_losses: list[float],
):
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    for i, (indices, targets) in enumerate(train_loader, start=1):
        train_loss = train_step(indices, targets, criterion, model, optimizer, scaler)
        train_losses.append(train_loss)

        if i % config.validate_freq == 0:
            save_model(model, optimizer, train_loss, "last", train_losses, val_losses, logger)
            logger.info(f"| train loss after {i} iterations: {train_loss}")

            logger.info(f"| validating after {i} iterations")
            val_loss = validate_step(model, criterion, val_loader, config, tokenizer)
            val_losses.append(val_loss)
            scheduler.step(val_loss)
            if val_loss < best_loss:
                logger.info(f'| best loss was updated from {best_loss} to {val_loss}')
                save_model(model, optimizer, val_loss, "best", train_losses, val_losses, logger)
                best_loss = val_loss

    return best_loss, train_losses, val_losses


def train(
    model: LanguageModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: torch.cuda.amp.GradScaler,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config,
    tokenizer: Tokenizer,
):
    train_losses, val_losses = [], []
    best_loss = float('+inf')
    for epoch in range(1, config.num_epoches + 1):
        logger.info(f"| training epoch #{epoch}")
        best_loss, train_losses, val_losses = train_epoch(
            model,
            optimizer,
            scheduler,
            scaler,
            train_loader,
            val_loader,
            config,
            tokenizer,
            best_loss,
            train_losses,
            val_losses,
        )
