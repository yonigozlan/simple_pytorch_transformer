import torch
from torch import nn


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator: nn.Module, criterion: nn.Module):
        self.generator = generator
        self.criterion = criterion

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, norm: float
    ) -> tuple[torch.Tensor]:
        x = self.generator(x)
        sloss = (
            self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1))
            / norm
        )
        return sloss.data * norm, sloss


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size: int, padding_idx: int, smoothing: float = 0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist

        return self.criterion(x, true_dist.clone().detach())
