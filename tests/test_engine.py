"""End-to-end checks: a real file, a real torch import, a real subprocess."""

import sys
import textwrap
from pathlib import Path

import pytest

from torchtyc.config import Config
from torchtyc.engine import check_paths


@pytest.fixture
def project(tmp_path: Path):
    def build(source: str) -> tuple[list[str], Config]:
        path = tmp_path / "model.py"
        path.write_text(textwrap.dedent(source))
        config = Config(root=tmp_path, python=sys.executable)
        return [str(path)], config

    return build


HEADER = """
    import torch
    from einops import einsum
    from jaxtyping import Float, Int
    from torch import Tensor, nn
"""


def rules(report) -> list[str]:
    return [d.rule for d in report.diagnostics]


def test_correct_function_is_clean(project):
    paths, config = project(
        HEADER
        + """
    def double(x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
        return x * 2
    """
    )
    report = check_paths(paths, config)
    assert report.diagnostics == []
    assert report.ok


def test_swapped_return_dims(project):
    paths, config = project(
        HEADER
        + """
    def linear(
        x: Float[Tensor, "b d_in"], w: Float[Tensor, "d_out d_in"]
    ) -> Float[Tensor, "b d_in"]:
        return einsum(x, w, "b d_in, d_out d_in -> b d_out")
    """
    )
    report = check_paths(paths, config)
    assert "shape-mismatch" in rules(report)
    assert not report.ok


def test_flatten_is_reported_as_a_rank_error(project):
    paths, config = project(
        HEADER
        + """
    def flat(x: Float[Tensor, "b s d"]) -> Float[Tensor, "b s d"]:
        return x.reshape(x.shape[0], -1)
    """
    )
    report = check_paths(paths, config)
    diagnostic = next(d for d in report.diagnostics if d.rule == "rank-mismatch")
    assert "*" in (diagnostic.got or "")


def test_dtype_mismatch(project):
    paths, config = project(
        HEADER
        + """
    def to_int(x: Float[Tensor, "b"]) -> Float[Tensor, "b"]:
        return x.long()
    """
    )
    assert "dtype-mismatch" in rules(check_paths(paths, config))


def test_module_is_constructed_from_matching_param_names(project):
    paths, config = project(
        HEADER
        + """
    class Linear(nn.Module):
        def __init__(self, in_features: int, out_features: int) -> None:
            super().__init__()
            self.W = nn.Parameter(torch.empty((out_features, in_features)))

        def forward(self, x: Float[Tensor, "... in_features"]) -> Float[Tensor, "... in_features"]:
            return einsum(x, self.W, "... in_features, out_features in_features -> ... out_features")
    """
    )
    report = check_paths(paths, config)
    diagnostic = next(d for d in report.diagnostics if d.rule == "shape-mismatch")
    assert "out_features" in diagnostic.message


def test_annotated_attribute_is_checked(project):
    paths, config = project(
        HEADER
        + """
    class Net(nn.Module):
        def __init__(self, d_in: int, d_out: int) -> None:
            super().__init__()
            self.W: Float[nn.Parameter, "d_out d_in"] = nn.Parameter(torch.empty((d_in, d_out)))
    """
    )
    assert "attribute-mismatch" in rules(check_paths(paths, config))


def test_tuple_return_arity(project):
    paths, config = project(
        HEADER
        + """
    def split(x: Float[Tensor, "b d"]) -> tuple[Float[Tensor, "b d"], Float[Tensor, "b d"]]:
        return (x,)
    """
    )
    assert "tuple-arity" in rules(check_paths(paths, config))


def test_trace_error_anchors_to_user_code(project):
    paths, config = project(
        HEADER
        + """
    def bad(x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
        y = x @ x
        return y
    """
    )
    report = check_paths(paths, config)
    diagnostic = next(d for d in report.diagnostics if d.rule == "trace-error")
    # The failing line is `y = x @ x`, not anything inside torch.
    assert diagnostic.line == 7


def test_import_error_is_reported(project):
    paths, config = project(
        HEADER
        + """
    import a_module_that_does_not_exist

    def f(x: Float[Tensor, "b"]) -> Float[Tensor, "b"]:
        return x
    """
    )
    assert "import-error" in rules(check_paths(paths, config))


def test_suppression_silences_a_rule(project):
    paths, config = project(
        HEADER
        + """
    def linear(x: Float[Tensor, "b d_in"]) -> Float[Tensor, "b d_in"]:  # torchtyc: ignore[rank-mismatch]
        return x.reshape(-1)
    """
    )
    assert "rank-mismatch" not in rules(check_paths(paths, config))


def test_unused_suppression_is_reported(project):
    paths, config = project(
        HEADER
        + """
    def fine(x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:  # torchtyc: ignore[shape-mismatch]
        return x
    """
    )
    assert "suppression-unused" in rules(check_paths(paths, config))


def test_config_ignore_drops_a_rule(project):
    paths, config = project(
        HEADER
        + """
    def flat(x: Float[Tensor, "b s d"]) -> Float[Tensor, "b s d"]:
        return x.reshape(x.shape[0], -1)
    """
    )
    config.ignore = frozenset({"rank-mismatch"})
    assert "rank-mismatch" not in rules(check_paths(paths, config))


def test_hover_shapes_are_produced(project):
    paths, config = project(
        HEADER
        + """
    def double(x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
        return x * 2
    """
    )
    report = check_paths(paths, config, hover=True)
    assert "b" in report.hovers["double"]["x"]


def test_variadic_rank_is_configurable(project):
    paths, config = project(
        HEADER
        + """
    def needs_three_batch_dims(x: Float[Tensor, "... d"]) -> Float[Tensor, "d"]:
        return x[0, 0, 0]
    """
    )
    config.variadic_rank = 3
    assert check_paths(paths, config).diagnostics == []


def test_broken_interpreter_reports_worker_error(project):
    paths, config = project(
        HEADER
        + """
    def f(x: Float[Tensor, "b"]) -> Float[Tensor, "b"]:
        return x
    """
    )
    config.python = "/nonexistent/python"
    report = check_paths(paths, config)
    assert report.worker_error is not None
    assert not report.ok


def test_positional_only_parameter_traces(project):
    paths, config = project(
        HEADER
        + """
    def half(x: Float[Tensor, "b d"], /) -> Float[Tensor, "b d"]:
        return x * 0.5
    """
    )
    assert check_paths(paths, config).diagnostics == []


def test_positional_only_parameter_still_catches_a_bad_shape(project):
    paths, config = project(
        HEADER
        + """
    def swap(x: Float[Tensor, "b d"], /) -> Float[Tensor, "d b"]:
        return x
    """
    )
    assert "shape-mismatch" in rules(check_paths(paths, config))


def test_variadic_conflict_is_reported_as_dim_inconsistent(project):
    paths, config = project(
        HEADER
        + """
    def drop(x: Float[Tensor, "*batch d"]) -> Float[Tensor, "*batch d"]:
        return x[0]
    """
    )
    assert "dim-inconsistent" in rules(check_paths(paths, config))


def test_unsaved_buffer_is_what_gets_traced(project):
    paths, config = project(
        HEADER
        + """
    def saved(x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
        return x
    """
    )
    buffer = textwrap.dedent(
        HEADER
        + """
    def renamed(x: Float[Tensor, "b d"]) -> Float[Tensor, "d b"]:
        return x
    """
    )
    report = check_paths(paths, config, sources={paths[0]: buffer})
    found = rules(report)
    assert "shape-mismatch" in found
    assert "trace-error" not in found


def test_an_emptied_buffer_reports_nothing(project):
    paths, config = project(
        HEADER
        + """
    def flat(x: Float[Tensor, "b s d"]):
        return x.reshape(x.shape[0], -1)
    """
    )
    assert "anonymous-return" in rules(check_paths(paths, config))
    assert check_paths(paths, config, sources={paths[0]: ""}).diagnostics == []


def test_a_buffer_inside_a_package_keeps_its_relative_imports(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "helpers.py").write_text("SCALE = 2\n")
    model = package / "model.py"
    model.write_text(textwrap.dedent(HEADER))

    buffer = textwrap.dedent(
        HEADER
        + """
    from .helpers import SCALE

    def scale(x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
        return x * SCALE
    """
    )
    config = Config(root=tmp_path, python=sys.executable)
    report = check_paths([str(model)], config, sources={str(model): buffer})
    assert report.diagnostics == []
    assert report.worker_error is None
