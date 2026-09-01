from jaxtyping import Float
from torch import Tensor


def squeeze_wrong(x: Float[Tensor, "batch seq d"]) -> Float[Tensor, "batch seq d"]:
    undefined_name_for_pyright()
    return x.sum(dim=1)
