from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SequenceHeadConfig:
    embedding_dim: int
    num_phases: int
    num_stages: int = 4
    num_layers: int = 10
    hidden_dim: int = 64
    dropout: float = 0.5
    causal: bool = True


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation: int, channels: int, dropout: float, causal: bool):
        super().__init__()
        self.padding = (2 * dilation, 0) if causal else (dilation, dilation)
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.mix = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.mix(F.relu(self.conv(F.pad(x, self.padding)))))


class SingleStage(nn.Module):
    def __init__(self, in_channels: int, config: SequenceHeadConfig):
        super().__init__()
        self.input = nn.Conv1d(in_channels, config.hidden_dim, kernel_size=1)
        self.layers = nn.Sequential(
            *(
                DilatedResidualLayer(2**i, config.hidden_dim, config.dropout, config.causal)
                for i in range(config.num_layers)
            )
        )
        self.output = nn.Conv1d(config.hidden_dim, config.num_phases, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.layers(self.input(x)))


class SequenceHead(nn.Module):
    is_sequence_model = True

    def __init__(self, config: SequenceHeadConfig):
        super().__init__()
        self.config = config
        self.stages = nn.ModuleList(
            [SingleStage(config.embedding_dim, config)]
            + [SingleStage(config.num_phases, config) for _ in range(config.num_stages - 1)]
        )

    @property
    def embedding_dim(self) -> int:
        return self.config.embedding_dim

    @property
    def num_phases(self) -> int:
        return self.config.num_phases

    def stage_logits(self, embeddings: torch.Tensor) -> list[torch.Tensor]:
        x = embeddings.transpose(1, 2)
        outputs = []
        for index, stage in enumerate(self.stages):
            x = stage(x if index == 0 else F.softmax(x, dim=1))
            outputs.append(x)
        return outputs

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        batched = embeddings.dim() == 3
        x = embeddings if batched else embeddings.unsqueeze(0)
        logits = self.stage_logits(x)[-1].transpose(1, 2)
        return logits if batched else logits.squeeze(0)


def sequence_loss(
    stage_logits: list[torch.Tensor],
    labels: torch.Tensor,
    classification_loss: nn.Module,
    smoothing_weight: float,
    smoothing_max: float,
) -> torch.Tensor:
    loss = labels.new_zeros((), dtype=torch.float)
    for logits in stage_logits:
        loss = loss + classification_loss(logits, labels)
        log_probs = F.log_softmax(logits, dim=1)
        jumps = (log_probs[:, :, 1:] - log_probs.detach()[:, :, :-1]) ** 2
        loss = loss + smoothing_weight * jumps.clamp(max=smoothing_max**2).mean()
    return loss
