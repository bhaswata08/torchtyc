from torchtyc.annotations import ArraySpec
from torchtyc.diagnostics import Diagnostic, Severity
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
    # Class-wide: the attribute names two axes and `forward` adds `b`, and a
    # constructor parameter is matched against all of them.
    assert info.dim_names == {"d_out", "d_in", "b"}
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


def test_end_line_covers_the_whole_function():
    scan = scan_source(SOURCE, "net.py")
    free = next(t for t in scan.targets if t.qualname == "free")
    lines = SOURCE.splitlines()
    assert lines[free.position.line].lstrip().startswith("def free")
    assert lines[free.end_line].strip() == "return x"


def test_positional_only_params_are_marked():
    scan = scan_source("def f(a, /, b, *, c=1): ...\n", "a.py")
    target = scan.targets[0]
    assert [(p.name, p.positional_only) for p in target.params] == [
        ("a", True),
        ("b", False),
        ("c", False),
    ]


def test_torch_einsum_is_not_an_einops_call():
    source = """
import torch


def f(x, y):
    return torch.einsum("bij,bjk->bik", x, y)
"""
    scan = scan_source(source, "a.py")
    assert scan.targets[0].einops_calls == []


def test_aliased_einops_imports_are_recognised():
    source = """
import einops as E
from einops import rearrange as rr


def f(x):
    y = rr(x, "a b -> b a")
    return E.reduce(y, "a b -> a", "sum")
"""
    scan = scan_source(source, "a.py")
    assert {c.func for c in scan.targets[0].einops_calls} == {"rearrange", "reduce"}


def test_a_local_function_sharing_an_einops_name_is_not_claimed():
    source = """
def rearrange(x, pattern):
    return x


def f(x):
    return rearrange(x, "not a pattern")
"""
    scan = scan_source(source, "a.py")
    assert scan.targets[1].einops_calls == []


def _suppressed(source: str, rules: list[str]) -> list[str]:
    """Which of these rules survive the suppressions written in this source."""
    from torchtyc.engine import apply_suppressions

    scan = scan_source(source, "a.py")
    diagnostics = [
        Diagnostic(path="a.py", line=0, column=0, rule=rule, severity=Severity.ERROR, message="")
        for rule in rules
    ]
    return [d.rule for d in apply_suppressions(diagnostics, scan.suppressions)]


def test_two_scoped_ignores_on_one_line_both_apply():
    source = "x = 1  # torchtyc: ignore[rank-mismatch]  # torchtyc: ignore[dtype-mismatch]\n"
    assert _suppressed(source, ["rank-mismatch", "dtype-mismatch", "unused-dim"]) == ["unused-dim"]


def test_a_bare_ignore_beside_a_scoped_one_covers_every_rule():
    source = "x = 1  # torchtyc: ignore[rank-mismatch]  # torchtyc: ignore\n"
    assert _suppressed(source, ["rank-mismatch", "unused-dim"]) == []


def test_class_position_spans_the_class_name():
    source = "class Net:\n    x = 1\n    y = 2\n"
    info = scan_source(source, "a.py").classes[0]
    assert (info.position.line, info.position.end_line) == (0, 0)
    assert source.splitlines()[0][info.position.column : info.position.end_column] == "class Net"


SIGNATURES = """
from jaxtyping import Float
from torch import Tensor


def one_line(x: Float[Tensor, "a"]) -> Float[Tensor, "a"]:
    return x


def wrapped(
    x: Float[Tensor, "a b"],
    y: Float[Tensor, "b c"],
) -> Float[Tensor, "a c"]:
    return x @ y


def compact(x: Float[Tensor, "a"]) -> Float[Tensor, "a"]: return x


def unannotated_return(x: Float[Tensor, "a"]):
    return x
"""


def signature_end(name: str) -> int:
    scan = scan_source(SIGNATURES, "sig.py")
    return next(t for t in scan.targets if t.name == name).signature_end_line


def test_signature_end_of_a_single_line_def():
    # `def one_line(...) -> ...:` is on the 6th line, 0-based 5.
    assert signature_end("one_line") == 5


def test_signature_end_of_a_wrapped_def():
    lines = SIGNATURES.splitlines()
    # The header finishes on the line carrying the return annotation, not on
    # the `def` line.
    assert lines[signature_end("wrapped")].strip() == ') -> Float[Tensor, "a c"]:'


def test_signature_end_never_runs_past_the_body():
    lines = SIGNATURES.splitlines()
    assert lines[signature_end("compact")].lstrip().startswith("def compact")


def test_signature_end_without_a_return_annotation():
    lines = SIGNATURES.splitlines()
    assert lines[signature_end("unannotated_return")].lstrip().startswith("def unannotated_return")


NESTED = """
from jaxtyping import Float
from torch import Tensor


class Outer:
    class Inner:
        def forward(self, x: Float[Tensor, "a b"]) -> Float[Tensor, "a b"]:
            return x


def factory():
    class Made:
        def forward(self, x: Float[Tensor, "a b"]) -> Float[Tensor, "a b"]:
            return x

    return Made


if True:
    class Conditional:
        def forward(self, x: Float[Tensor, "a b"]) -> Float[Tensor, "a b"]:
            return x


class Callable_:
    def __call__(self, x: Float[Tensor, "a b"]) -> Float[Tensor, "a b"]:
        return x

    def __repr__(self) -> str:
        return "Callable_()"
"""


def test_a_class_nested_in_a_class_is_discovered():
    scan = scan_source(NESTED, "n.py")
    assert "Outer.Inner.forward" in {t.qualname for t in scan.targets}
    assert "Outer.Inner" in {c.qualname for c in scan.classes}


def test_a_class_in_a_factory_function_is_discovered_under_locals():
    scan = scan_source(NESTED, "n.py")
    assert "factory.<locals>.Made.forward" in {t.qualname for t in scan.targets}


def test_a_class_under_a_module_level_if_is_discovered():
    scan = scan_source(NESTED, "n.py")
    assert "Conditional.forward" in {t.qualname for t in scan.targets}


def test_call_is_traced_but_other_dunders_are_not():
    names = {t.qualname for t in scan_source(NESTED, "n.py").targets}
    assert "Callable_.__call__" in names
    assert "Callable_.__repr__" not in names


def test_a_class_under_type_checking_is_not_discovered():
    source = """
from typing import TYPE_CHECKING

from jaxtyping import Float
from torch import Tensor

if TYPE_CHECKING:
    class Phantom:
        def forward(self, x: Float[Tensor, "a b"]) -> Float[Tensor, "a b"]:
            return x
"""
    scan = scan_source(source, "t.py")
    assert scan.targets == []
    assert scan.classes == []
