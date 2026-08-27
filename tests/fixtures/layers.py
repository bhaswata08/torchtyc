import math

import torch
from einops import einsum
from jaxtyping import Float
from torch import Tensor, nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        stdev = math.sqrt(2 / (in_features + out_features))
        self.W: Float[nn.Parameter, "out_features in_features"] = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty((out_features, in_features), dtype=dtype, device=device),
                mean=0.0,
                std=stdev,
                a=-3 * stdev,
                b=3 * stdev,
            )
        )

    def forward(self, x: Float[Tensor, "... in_features"]) -> Float[Tensor, "... in_features"]:
        return einsum(x, self.W, "... in_features, out_features in_features -> ... out_features")


def scale(x: Float[Tensor, "batch seq d_model"]) -> Float[Tensor, "batch seq d_model"]:
    return x * 2


def flatten_wrong(x: Float[Tensor, "batch seq d_model"]) -> Float[Tensor, "batch seq d_model"]:
    return x.reshape(x.shape[0], -1)


class Transposed(nn.Module):
    """The weight is built with its axes the wrong way round."""

    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.W: Float[nn.Parameter, "d_out d_in"] = nn.Parameter(
            torch.empty((d_in, d_out))
        )

    def forward(self, x: Float[Tensor, "batch d_in"]) -> Float[Tensor, "batch d_out"]:
        return x @ self.W
