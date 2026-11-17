from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, get_args

import torch
import torch.nn as nn

from surgphase.finetuning.sequence_head import SequenceHead, SequenceHeadConfig

HeadType = Literal["linear", "simple", "robust"]
HEAD_TYPES: tuple[str, ...] = get_args(HeadType)


@dataclass(frozen=True)
class HeadConfig:
    embedding_dim: int
    num_phases: int
    hidden_dims: tuple[int, ...] = ()
    dropout: float = 0.0
    batch_norm: bool = False


HEAD_PRESETS: dict[str, dict] = {
    "linear": {"hidden_dims": (), "dropout": 0.0, "batch_norm": False},
    "simple": {"hidden_dims": (256,), "dropout": 0.1, "batch_norm": False},
    "robust": {"hidden_dims": (512, 256), "dropout": 0.3, "batch_norm": True},
}


class PhaseHead(nn.Sequential):
    def __init__(self, config: HeadConfig):
        layers: list[nn.Module] = []
        in_dim = config.embedding_dim
        for hidden_dim in config.hidden_dims:
            layers += [nn.Dropout(config.dropout), nn.Linear(in_dim, hidden_dim)]
            if config.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers += [nn.Dropout(config.dropout), nn.Linear(in_dim, config.num_phases)]
        super().__init__(*layers)
        self.config = config

    @property
    def embedding_dim(self) -> int:
        return self.config.embedding_dim

    @property
    def num_phases(self) -> int:
        return self.config.num_phases


def build_head(
    head_type: HeadType,
    embedding_dim: int,
    num_phases: int,
    dropout: float | None = None,
    hidden_dims: Sequence[int] | None = None,
) -> PhaseHead:
    if head_type not in HEAD_PRESETS:
        raise ValueError(f"Unknown head type {head_type!r}; choose from {HEAD_TYPES}")
    preset = HEAD_PRESETS[head_type]
    return PhaseHead(
        HeadConfig(
            embedding_dim=embedding_dim,
            num_phases=num_phases,
            hidden_dims=tuple(preset["hidden_dims"] if hidden_dims is None else hidden_dims),
            dropout=preset["dropout"] if dropout is None else dropout,
            batch_norm=preset["batch_norm"],
        )
    )


def save_head(head: PhaseHead | SequenceHead, path: str | Path) -> None:
    state_dict = {key: value.detach().cpu() for key, value in head.state_dict().items()}
    kind = "sequence" if isinstance(head, SequenceHead) else "frame"
    torch.save({"kind": kind, "config": asdict(head.config), "state_dict": state_dict}, Path(path))


def load_head(path: str | Path, device: str | torch.device = "cpu") -> PhaseHead | SequenceHead:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or not {"config", "state_dict"} <= checkpoint.keys():
        raise ValueError(f"{path} is not a head checkpoint written by save_head")
    config = dict(checkpoint["config"])
    if checkpoint.get("kind", "frame") == "sequence":
        head = SequenceHead(SequenceHeadConfig(**config))
    else:
        config["hidden_dims"] = tuple(config["hidden_dims"])
        head = PhaseHead(HeadConfig(**config))
    head.load_state_dict(checkpoint["state_dict"])
    return head.to(device).eval()
