import logging
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from surgphase.config import UNLABELLED
from surgphase.eval import compute_metrics
from surgphase.finetuning.sequence_head import SequenceHead, sequence_loss

logger = logging.getLogger(__name__)

Split = dict[str, tuple[torch.Tensor, torch.Tensor]]
SelectMetric = Literal["f1_macro", "accuracy", "loss"]
SEQUENCE_METRICS = (
    "transition_error",
    "temporal_stability",
    "edit_score",
    "f1_at_10",
    "f1_at_25",
    "f1_at_50",
)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-3
    min_lr_ratio: float = 1e-2
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    class_weights: bool = False
    select_metric: SelectMetric = "f1_macro"
    patience: int = 15
    min_delta: float = 1e-4
    log_every: int = 10
    smoothing_weight: float = 0.15
    smoothing_max: float = 4.0


@dataclass
class TrainingResult:
    best_epoch: int
    best_score: float
    best_state: dict[str, torch.Tensor]
    history: list[dict] = field(default_factory=list)


def balanced_class_weights(labels: torch.Tensor, num_phases: int) -> torch.Tensor:
    labels = labels[labels != UNLABELLED].numpy()
    present = np.unique(labels)
    weights = torch.zeros(num_phases)
    weights[present] = torch.from_numpy(
        compute_class_weight("balanced", classes=present, y=labels)
    ).float()
    return weights


@torch.no_grad()
def predict_logits(
    head: nn.Module,
    embeddings: torch.Tensor,
    device: str | torch.device,
    batch_size: int = 4096,
) -> torch.Tensor:
    head.eval()
    if getattr(head, "is_sequence_model", False):
        return head(embeddings.to(device)).cpu()
    return torch.cat([head(chunk.to(device)).cpu() for chunk in embeddings.split(batch_size)])


def evaluate(
    head: nn.Module,
    split: Split,
    device: str | torch.device,
    loss_fn: nn.Module | None = None,
) -> dict:
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    per_surgery, all_true, all_pred = {}, [], []
    total_loss, total = 0.0, 0
    for surgery_id, (embeddings, labels) in split.items():
        logits = predict_logits(head, embeddings, device)
        labelled = labels != UNLABELLED
        logits, labels = logits[labelled], labels[labelled]
        preds = logits.argmax(dim=1).numpy()
        total_loss += loss_fn(logits, labels).item() * len(labels)
        total += len(labels)
        per_surgery[surgery_id] = asdict(compute_metrics(labels.numpy(), preds))
        all_true.append(labels.numpy())
        all_pred.append(preds)

    overall = asdict(compute_metrics(np.concatenate(all_true), np.concatenate(all_pred)))
    for key in SEQUENCE_METRICS:
        overall[key] = float(np.mean([metrics[key] for metrics in per_surgery.values()]))
    return {"loss": total_loss / total, "overall": overall, "per_surgery": per_surgery}


def _loss_functions(
    train_labels: torch.Tensor, num_phases: int, config: TrainingConfig, device: str | torch.device
) -> tuple[nn.Module, nn.Module]:
    weights = balanced_class_weights(train_labels, num_phases) if config.class_weights else None
    train_loss_fn = nn.CrossEntropyLoss(
        weight=None if weights is None else weights.to(device),
        label_smoothing=config.label_smoothing,
        ignore_index=UNLABELLED,
    )
    return train_loss_fn, nn.CrossEntropyLoss(weight=weights)


def _train(
    head: nn.Module,
    train_epoch: Callable[[torch.optim.Optimizer], float],
    val_split: Split,
    val_loss_fn: nn.Module,
    config: TrainingConfig,
    device: str | torch.device,
) -> TrainingResult:
    head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.lr * config.min_lr_ratio
    )

    higher_is_better = config.select_metric != "loss"
    sign = 1.0 if higher_is_better else -1.0
    result = TrainingResult(best_epoch=0, best_score=-sign * float("inf"), best_state={})
    stale = 0

    for epoch in range(1, config.epochs + 1):
        head.train()
        train_loss = train_epoch(optimizer)
        scheduler.step()

        val = evaluate(head, val_split, device, val_loss_fn)
        score = (
            val["loss"] if config.select_metric == "loss" else val["overall"][config.select_metric]
        )
        result.history.append(
            {
                "epoch": epoch,
                "lr": scheduler.get_last_lr()[0],
                "train_loss": train_loss,
                "val_loss": val["loss"],
                "val_accuracy": val["overall"]["accuracy"],
                "val_f1_macro": val["overall"]["f1_macro"],
            }
        )

        improved = sign * (score - result.best_score) > config.min_delta
        if improved:
            result.best_score, result.best_epoch, stale = score, epoch, 0
            result.best_state = {
                key: value.detach().cpu().clone() for key, value in head.state_dict().items()
            }
        else:
            stale += 1

        if epoch == 1 or epoch % config.log_every == 0 or improved:
            logger.info(
                "Epoch %3d: train_loss=%.4f val_loss=%.4f val_acc=%.4f val_f1=%.4f%s",
                epoch,
                train_loss,
                val["loss"],
                val["overall"]["accuracy"],
                val["overall"]["f1_macro"],
                "  *" if improved else "",
            )
        if config.patience and stale >= config.patience:
            logger.info(
                "Early stopping at epoch %d (no %s improvement for %d epochs)",
                epoch,
                config.select_metric,
                stale,
            )
            break

    head.load_state_dict(result.best_state)
    return result


def fit(
    head: nn.Module,
    train_split: Split,
    val_split: Split,
    num_phases: int,
    config: TrainingConfig,
    device: str | torch.device,
) -> TrainingResult:
    train_x = torch.cat([embeddings for embeddings, _ in train_split.values()])
    train_y = torch.cat([labels for _, labels in train_split.values()])
    labelled = train_y != UNLABELLED
    train_x, train_y = train_x[labelled], train_y[labelled]
    has_batch_norm = any(isinstance(module, nn.BatchNorm1d) for module in head.modules())
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=has_batch_norm and len(train_y) % config.batch_size == 1,
    )
    train_loss_fn, val_loss_fn = _loss_functions(train_y, num_phases, config, device)

    def train_epoch(optimizer: torch.optim.Optimizer) -> float:
        running, seen = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = train_loss_fn(head(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * len(y)
            seen += len(y)
        return running / seen

    return _train(head, train_epoch, val_split, val_loss_fn, config, device)


def fit_sequence(
    head: SequenceHead,
    train_split: Split,
    val_split: Split,
    num_phases: int,
    config: TrainingConfig,
    device: str | torch.device,
) -> TrainingResult:
    sequences = list(train_split.values())
    train_y = torch.cat([labels for _, labels in sequences])
    train_loss_fn, val_loss_fn = _loss_functions(train_y, num_phases, config, device)

    def train_epoch(optimizer: torch.optim.Optimizer) -> float:
        running = 0.0
        for embeddings, labels in random.sample(sequences, len(sequences)):
            stage_logits = head.stage_logits(embeddings.unsqueeze(0).to(device))
            loss = sequence_loss(
                stage_logits,
                labels.unsqueeze(0).to(device),
                train_loss_fn,
                config.smoothing_weight,
                config.smoothing_max,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
        return running / len(sequences)

    return _train(head, train_epoch, val_split, val_loss_fn, config, device)
