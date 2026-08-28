"""The in-process lint pass: no import, no torch, just the parsed file."""

from pathlib import Path

from torchtyc.config import Config
from torchtyc.discovery import scan_source
from torchtyc.engine import lint_scan


def unused_dims(source: str) -> set[tuple[str, str]]:
    scan = scan_source(source, "model.py")
    return {
        (d.function, d.message.split("`")[1])
        for d in lint_scan(scan, Config(root=Path(".")))
        if d.rule == "unused-dim"
    }


HEADER = """
from jaxtyping import Float
from torch import Tensor, nn
"""


def test_a_dimension_bound_by_an_annotated_attribute_is_not_unused():
    """A forward legitimately gets `d_out` from `self.W`, so the class is one scope."""
    source = (
        HEADER
        + """
class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.W: Float[nn.Parameter, "d_out d_in"] = nn.Parameter(torch.empty(4, 3))

    def forward(self, x: Float[Tensor, "b d_in"]) -> Float[Tensor, "b d_out"]:
        return x @ self.W.T
"""
    )
    assert unused_dims(source) == set()


def test_a_dimension_bound_by_an_init_parameter_is_not_unused():
    source = (
        HEADER
        + """
class Net(nn.Module):
    def __init__(self, d_out: int) -> None:
        super().__init__()

    def forward(self, x: Float[Tensor, "b d_in"]) -> Float[Tensor, "b d_out"]:
        return x
"""
    )
    assert unused_dims(source) == {("Net.forward", "d_in")}


def test_a_free_function_dimension_is_unused_even_if_a_sibling_reuses_the_name():
    """Two module-level functions share nothing, so one cannot excuse the other."""
    source = (
        HEADER
        + """
def pool(x: Float[Tensor, "batch d"]) -> Float[Tensor, "batch"]:
    return x.sum(-1)

def widen(y: Float[Tensor, "d"]) -> Float[Tensor, "d two"]:
    return y[:, None].expand(-1, 2)
"""
    )
    assert unused_dims(source) == {("pool", "d"), ("widen", "two")}


def test_a_free_function_that_constrains_its_own_dimensions_is_quiet():
    source = (
        HEADER
        + """
def double(x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
    return x * 2
"""
    )
    assert unused_dims(source) == set()
