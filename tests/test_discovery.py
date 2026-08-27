from torchtyc.annotations import ArraySpec
from torchtyc.discovery import scan_source

SOURCE = """
import torch
from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor, nn


class Net(nn.Module):
    def __init__(self, d_in: int, d_out: int, bias: bool = True) -> None:
        self.W: Float[nn.Parameter, "d_out d_in"] = nn.Parameter(torch.empty(d_out, d_in))

    def forward(self, x: Float[Tensor, "b d_in"]) -> Float[Tensor, "b d_out"]:
        y = rearrange(x, "b d -> d b")
        return einsum(x, self.W, "b d_in, d_out d_in -> b d_out")

    def helper(self, n: int) -> int:  # torchtyc: ignore
        return n


def free(x: Float[Tensor, "a"]) -> Float[Tensor, "a"]:
    return x
"""


def test_finds_methods_and_functions():
    scan = scan_source(SOURCE, "net.py")
    assert {t.qualname for t in scan.targets} == {"Net.forward", "Net.helper", "free"}


def test_dunder_init_is_not_a_target():
    scan = scan_source(SOURCE, "net.py")
    assert all(t.name != "__init__" for t in scan.targets)


def test_init_params_and_attributes():
    scan = scan_source(SOURCE, "net.py")
    info = next(c for c in scan.classes if c.name == "Net")
    assert [p.name for p in info.init_params] == ["d_in", "d_out", "bias"]
    assert info.init_params[2].has_default is True
    assert [a.name for a in info.attributes] == ["W"]
    assert info.dim_names == {"d_out", "d_in"}
    assert info.is_module is True


def test_param_specs_and_plain_types():
    scan = scan_source(SOURCE, "net.py")
    forward = next(t for t in scan.targets if t.qualname == "Net.forward")
    assert isinstance(forward.params[1].spec, ArraySpec)
    helper = next(t for t in scan.targets if t.qualname == "Net.helper")
    assert helper.params[1].plain_type == "int"
    assert helper.has_array_annotation is False


def test_dim_names_span_params_and_return():
    scan = scan_source(SOURCE, "net.py")
    forward = next(t for t in scan.targets if t.qualname == "Net.forward")
    assert forward.dim_names == {"b", "d_in", "d_out"}


def test_einops_calls_captured():
    scan = scan_source(SOURCE, "net.py")
    forward = next(t for t in scan.targets if t.qualname == "Net.forward")
    funcs = {c.func for c in forward.einops_calls}
    assert funcs == {"rearrange", "einsum"}
    einsum_call = next(c for c in forward.einops_calls if c.func == "einsum")
    assert einsum_call.tensor_args == 2


def test_suppressions():
    scan = scan_source(SOURCE, "net.py")
    assert len(scan.suppressions) == 1
    assert scan.suppressions[0].rules is None


def test_scoped_suppression():
    scan = scan_source("x = 1  # torchtyc: ignore[shape-mismatch, unused-dim]\n", "a.py")
    assert scan.suppressions[0].rules == frozenset({"shape-mismatch", "unused-dim"})


def test_syntax_error_is_reported_not_raised():
    scan = scan_source("def broken(:\n", "bad.py")
    assert scan.syntax_error is not None
    assert scan.targets == []
