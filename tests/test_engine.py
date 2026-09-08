"""End-to-end checks: a real file, a real torch import, a real subprocess."""

import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from torchtyc import cli, tracing, worker
from torchtyc.binding import FIRST_PRIME
from torchtyc.config import Config
from torchtyc.discovery import scan_source
from torchtyc.engine import check_paths, collect_files


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
    assert "b" in report.shapes_in(paths[0])["double"]["x"]


def test_two_files_sharing_a_qualname_keep_separate_shapes(tmp_path: Path):
    paths = []
    for index, dims in enumerate(('"b d"', '"b d e"')):
        path = tmp_path / f"pkg{index}.py"
        path.write_text(
            textwrap.dedent(HEADER)
            + textwrap.dedent(f"""
            def double(x: Float[Tensor, {dims}]) -> Float[Tensor, {dims}]:
                return x * 2
            """)
        )
        paths.append(str(path))

    config = Config(root=tmp_path, python=sys.executable)
    report = check_paths(paths, config, hover=True)
    first = report.shapes_in(paths[0])["double"]["return"]
    second = report.shapes_in(paths[1])["double"]["return"]
    assert first != second
    assert first.count(",") == 1
    assert second.count(",") == 2


def test_an_unresolvable_init_parameter_skips_the_class_both_ways(project):
    paths, config = project(
        HEADER
        + """
    class Block(nn.Module):
        def __init__(self, d_model, dropout):
            super().__init__()
            self.W: Float[nn.Parameter, "d_model d_model"] = nn.Parameter(
                torch.empty((d_model, d_model))
            )
            self.drop = nn.Dropout(dropout)

        def forward(self, x: Float[Tensor, "b d_model"]) -> Float[Tensor, "b d_model"]:
            return self.drop(x @ self.W)
    """
    )
    report = check_paths(paths, config)
    found = rules(report)
    assert "trace-error" not in found
    assert found.count("uninstantiable") == 2


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


def test_a_size_one_dimension_does_not_stall_the_worker(project):
    paths, config = project(
        HEADER
        + """
    class Net(nn.Module):
        def __init__(self, d_in: int) -> None:
            super().__init__()
            self.bias: Float[nn.Parameter, "one"] = nn.Parameter(torch.empty(1))
            self.W: Float[nn.Parameter, "d_in one"] = nn.Parameter(torch.empty((7, 1)))
    """
    )
    config.timeout = 20.0
    report = check_paths(paths, config)
    assert report.worker_error is None
    assert "attribute-mismatch" in rules(report)


def test_a_package_init_is_imported_once(tmp_path):
    """`pkg/__init__.py` is the module `pkg`, so its body must not run twice."""
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text(
        textwrap.dedent(HEADER)
        + textwrap.dedent(
            """
            from pathlib import Path

            LOG = Path(__file__).with_name("imports.log")
            LOG.write_text(LOG.read_text() + "x" if LOG.exists() else "x")

            def scale(x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
                return x * 2
            """
        )
    )
    # The count is kept in a file the imported body writes, so the guard has to
    # be off for this one check.
    config = Config(root=tmp_path, python=sys.executable, allow_effects=True)
    report = check_paths([str(package / "__init__.py")], config)

    assert report.worker_error is None
    assert report.diagnostics == []
    assert (package / "imports.log").read_text() == "x"


def test_a_module_that_prints_on_import_still_reports_its_diagnostics(project):
    paths, config = project(
        HEADER
        + """
    print("loading the model")

    def linear(
        x: Float[Tensor, "b d_in"], w: Float[Tensor, "d_out d_in"]
    ) -> Float[Tensor, "b d_in"]:
        return einsum(x, w, "b d_in, d_out d_in -> b d_out")
    """
    )
    report = check_paths(paths, config)
    assert report.worker_error is None
    assert "shape-mismatch" in rules(report)


def test_an_excluded_name_above_the_target_does_not_hide_the_tree(tmp_path):
    """`build` is excluded, but only below the directory the user asked for."""
    project_dir = tmp_path / "build" / "proj"
    project_dir.mkdir(parents=True)
    (project_dir / "model.py").write_text("")
    (project_dir / "build").mkdir()
    (project_dir / "build" / "generated.py").write_text("")

    config = Config(root=tmp_path, python=sys.executable)
    assert collect_files([str(project_dir)], config) == [str(project_dir / "model.py")]


def test_shape_mismatch_suggests_the_traced_annotation(project):
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
    assert diagnostic.suggestion == 'Float[Tensor, "... out_features"]'


def test_message_does_not_leak_the_prime_when_both_axes_are_named(project):
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
    assert "101" not in diagnostic.message
    assert "in_features" in diagnostic.message and "out_features" in diagnostic.message


def test_traced_shape_renders_the_variadic_as_ellipsis(project):
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
    assert diagnostic.got == "(..., out_features)"


def test_no_suggestion_when_the_traced_shape_has_a_product(project):
    paths, config = project(
        HEADER
        + """
    def flat(x: Float[Tensor, "b s d"]) -> Float[Tensor, "b s d"]:
        return x.reshape(x.shape[0], -1)
    """
    )
    report = check_paths(paths, config)
    diagnostic = next(d for d in report.diagnostics if d.rule == "rank-mismatch")
    assert diagnostic.suggestion is None


def test_dtype_mismatch_suggests_the_right_dtype_class(project):
    paths, config = project(
        HEADER
        + """
    def to_int(x: Float[Tensor, "b"]) -> Float[Tensor, "b"]:
        return x.long()
    """
    )
    report = check_paths(paths, config)
    diagnostic = next(d for d in report.diagnostics if d.rule == "dtype-mismatch")
    assert diagnostic.suggestion == 'Int[Tensor, "b"]'


def test_trace_command_reports_a_failed_trace(tmp_path, capsys):
    from torchtyc.cli import build_parser, cmd_trace

    path = tmp_path / "model.py"
    path.write_text(
        textwrap.dedent(
            """
            from jaxtyping import Float
            from torch import Tensor


            def bad(x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
                y = x @ x
                return y
            """
        )
    )
    args = build_parser().parse_args(["trace", f"{path}::bad", "--python", sys.executable])

    assert cmd_trace(args) != 0
    captured = capsys.readouterr()
    assert "trace-error" in captured.err
    assert "primes" not in captured.out


def test_a_nested_class_is_traced(project):
    paths, config = project(
        HEADER
        + """
    class Outer:
        class Inner(nn.Module):
            def forward(self, x: Float[Tensor, "b d"]) -> Float[Tensor, "d b"]:
                return x
    """
    )
    report = check_paths(paths, config)
    diagnostic = next(d for d in report.diagnostics if d.rule == "shape-mismatch")
    assert diagnostic.function == "Outer.Inner.forward"


def test_a_class_under_a_module_level_if_is_traced(project):
    paths, config = project(
        HEADER
        + """
    if True:

        class Conditional(nn.Module):
            def forward(self, x: Float[Tensor, "b d"]) -> Float[Tensor, "d b"]:
                return x
    """
    )
    report = check_paths(paths, config)
    assert "shape-mismatch" in rules(report)


def test_a_class_in_a_factory_function_is_reported_not_skipped_silently(project):
    paths, config = project(
        HEADER
        + """
    def factory():
        class Made(nn.Module):
            def forward(self, x: Float[Tensor, "b d"]) -> Float[Tensor, "d b"]:
                return x

        return Made
    """
    )
    report = check_paths(paths, config)
    diagnostic = next(d for d in report.diagnostics if d.rule == "local-definition")
    assert "factory.<locals>.Made" in diagnostic.message


def test_call_is_traced(project):
    paths, config = project(
        HEADER
        + """
    class Wrong(nn.Module):
        def __call__(self, x: Float[Tensor, "b d"]) -> Float[Tensor, "d b"]:
            return x
    """
    )
    report = check_paths(paths, config)
    diagnostic = next(d for d in report.diagnostics if d.rule == "shape-mismatch")
    assert diagnostic.function == "Wrong.__call__"


def test_an_unannotated_tensor_argument_never_leaks_a_prime(project):
    paths, config = project(
        HEADER
        + """
    def slice_to(bias: Tensor, x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
        return x[:, : bias.shape[0]]
    """
    )
    report = check_paths(paths, config)
    diagnostic = next(d for d in report.diagnostics if d.rule == "shape-mismatch")
    assert "_" in (diagnostic.got or "")
    assert not any(char.isdigit() for char in diagnostic.message)


def test_an_init_parameter_named_args_is_still_passed(project):
    paths, config = project(
        HEADER
        + """
    class Block(nn.Module):
        def __init__(self, args: int) -> None:
            super().__init__()
            self.W: Float[nn.Parameter, "args"] = nn.Parameter(torch.empty((args,)))

        def forward(self, x: Float[Tensor, "b args"]) -> Float[Tensor, "b args"]:
            return x + self.W
    """
    )
    report = check_paths(paths, config)
    assert report.diagnostics == []


def test_a_trace_error_underlines_the_statement_not_the_indentation(project):
    paths, config = project(
        HEADER
        + """
    class Block(nn.Module):
        def forward(self, x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
            return x @ x
    """
    )
    report = check_paths(paths, config)
    diagnostic = next(d for d in report.diagnostics if d.rule == "trace-error")
    line = Path(paths[0]).read_text().splitlines()[diagnostic.line]
    # The span is the failing expression itself, never the leading indentation.
    assert line[diagnostic.column : diagnostic.end_column] == "x @ x"


def test_a_class_under_a_guard_that_never_runs_is_not_reported(project):
    paths, config = project(
        """
    import sys

    from jaxtyping import Float
    from torch import Tensor, nn

    if sys.version_info >= (3, 99):

        class Legacy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.W: Float[nn.Parameter, "d d"] = nn.Parameter(torch.empty((3, 3)))

            def forward(self, x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
                return x
    """
    )
    report = check_paths(paths, config)
    assert report.diagnostics == []
    assert report.ok


def test_only_the_branch_the_import_took_is_traced(project):
    paths, config = project(
        HEADER
        + """
    import sys

    if sys.version_info >= (3, 0):

        def widen(x: Float[Tensor, "a"]) -> Float[Tensor, "a b"]:
            return x[:, None] * torch.ones((1, 4))

    else:

        def widen(x: Float[Tensor, "a"]) -> Float[Tensor, "a"]:
            return x
    """
    )
    report = check_paths(paths, config)
    # The losing branch annotates one axis; tracing it against the live
    # two-axis function is what used to raise a false rank-mismatch.
    assert "rank-mismatch" not in rules(report)
    assert report.ok


def test_a_guarded_fallback_shadowed_by_an_import_is_not_traced(tmp_path: Path):
    (tmp_path / "fast.py").write_text(
        textwrap.dedent("""
        from jaxtyping import Float
        from torch import Tensor, nn


        class Block(nn.Module):
            def forward(self, x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
                return x
        """)
    )
    path = tmp_path / "model.py"
    path.write_text(
        textwrap.dedent(HEADER)
        + textwrap.dedent("""
        try:
            from fast import Block
        except ImportError:

            class Block(nn.Module):
                def forward(self, x: Float[Tensor, "b d"]) -> Float[Tensor, "d b"]:
                    return x
        """)
    )
    config = Config(root=tmp_path, python=sys.executable)
    report = check_paths([str(path)], config)
    assert report.diagnostics == []


def test_a_guarded_definition_that_is_live_is_still_checked(project):
    paths, config = project(
        HEADER
        + """
    import sys

    if sys.version_info >= (3, 0):

        def flip(x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
            return x.T
    """
    )
    report = check_paths(paths, config)
    assert "shape-mismatch" in rules(report)


def test_a_trace_error_reports_axis_names_not_primes(project):
    paths, config = project(
        HEADER
        + """
    class Block(nn.Module):
        def __init__(self, d_in: int, d_out: int) -> None:
            super().__init__()
            self.W = nn.Parameter(torch.empty((d_out, d_in)))

        def forward(self, x: Float[Tensor, "b d_in"]) -> Float[Tensor, "b d_out"]:
            return x @ self.W
    """
    )
    report = check_paths(paths, config)
    diagnostic = next(d for d in report.diagnostics if d.rule == "trace-error")
    assert "d_in" in diagnostic.message
    assert "d_out" in diagnostic.message
    assert "101" not in diagnostic.message
    assert "103" not in diagnostic.message


def test_a_local_class_is_reported_once(project):
    paths, config = project(
        HEADER
        + """
    def factory():
        class Made(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.W: Float[nn.Parameter, "d d"] = nn.Parameter(torch.empty((3, 3)))

            def forward(self, x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
                return x

        return Made
    """
    )
    report = check_paths(paths, config)
    local = [d for d in report.diagnostics if d.rule == "local-definition"]
    assert len(local) == 1
    assert local[0].line == next(
        index
        for index, line in enumerate(Path(paths[0]).read_text().splitlines())
        if line.strip().startswith("class Made")
    )


def test_a_guarded_method_that_never_ran_is_not_traced(project):
    paths, config = project(
        HEADER
        + """
    FLAG = False


    class Block(nn.Module):
        if FLAG:

            def forward(self, x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
                return x
    """
    )
    report = check_paths(paths, config)
    assert report.diagnostics == []
    assert report.ok


def test_only_the_live_branch_of_a_method_is_traced(project):
    paths, config = project(
        HEADER
        + """
    FLAG = True


    class Block(nn.Module):
        if FLAG:

            def forward(self, x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
                return x

        else:

            def forward(self, x: Float[Tensor, "b d"]) -> Float[Tensor, "b"]:
                return x.sum(-1)
    """
    )
    report = check_paths(paths, config)
    assert "rank-mismatch" not in rules(report)
    assert report.ok


def test_a_local_class_with_nothing_annotated_is_not_reported(project):
    paths, config = project(
        HEADER
        + """
    def train(x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
        class _Ctx:
            pass

        _Ctx()
        return x
    """
    )
    report = check_paths(paths, config)
    assert "local-definition" not in rules(report)
    assert report.diagnostics == []


def test_a_failure_inside_init_reports_axis_names(project):
    paths, config = project(
        HEADER
        + """
    class Block(nn.Module):
        def __init__(self, d_in: int, d_out: int) -> None:
            super().__init__()
            self.W = nn.Parameter(torch.empty(d_out, d_in)).view(d_in, d_out, 2)

        def forward(self, x: Float[Tensor, "b d_in"]) -> Float[Tensor, "b d_out"]:
            return x
    """
    )
    report = check_paths(paths, config)
    diagnostic = next(d for d in report.diagnostics if d.rule == "trace-error")
    assert "d_in" in diagnostic.message
    assert "101" not in diagnostic.message


def test_json_output_carries_no_synthetic_primes(project, capsys):
    import json as json_module

    from torchtyc.binding import _PRIME_POOL
    from torchtyc.formats import render

    paths, config = project(
        HEADER
        + """
    class Block(nn.Module):
        def __init__(self, d_in: int, d_out: int) -> None:
            super().__init__()
            self.W = nn.Parameter(torch.empty((d_out, d_in)))

        def forward(self, x: Float[Tensor, "b d_in"]) -> Float[Tensor, "b d_out"]:
            return x @ self.W
    """
    )
    report = check_paths(paths, config)
    print(render(report, "json", config.root))
    payload = json_module.loads(capsys.readouterr().out)

    # Only the fields that carry torchtyc's own rendering are checked. Paths
    # live under the pytest session directory, so a prime session number is a
    # position the test was given and not a shape it let through. Frame lines
    # in a traceback are positions for the same reason, while the message
    # lines beside them went through the binder and stay in the check.
    rendered: list[str] = []
    for entry in payload["diagnostics"]:
        for key in ("message", "expected", "got", "hint", "note", "suggestion"):
            value = entry.get(key)
            if value:
                rendered.append(value)
        lines = (entry.get("traceback") or "").splitlines()
        rendered.extend(line for line in lines if line and not line[0].isspace())
    for per_file in payload.get("hovers", {}).values():
        for shapes in per_file.values():
            rendered.extend(shapes.values())

    primes = {str(p) for p in _PRIME_POOL[:8]}
    numbers = set(re.findall(r"\d+", "\n".join(rendered)))
    assert not (numbers & primes), f"a synthetic prime reached the json output: {numbers & primes}"

    # The traceback still names the real source lines it points at, which are
    # positions and not shapes, so renaming must have left them alone.
    diagnostic = next(d for d in payload["diagnostics"] if d["rule"] == "trace-error")
    assert f"line {diagnostic['line'] + 1}" in diagnostic["traceback"]


def test_a_check_run_from_a_subdirectory_finds_the_file(tmp_path, monkeypatch, capsys):
    """The worker runs at the project root, so a relative path has to survive it."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    package = tmp_path / "src"
    package.mkdir()
    (package / "model.py").write_text(
        textwrap.dedent(HEADER)
        + textwrap.dedent(
            """
            def linear(
                x: Float[Tensor, "b d_in"], w: Float[Tensor, "d_out d_in"]
            ) -> Float[Tensor, "b d_in"]:
                return einsum(x, w, "b d_in, d_out d_in -> b d_out")
            """
        )
    )
    monkeypatch.chdir(package)

    code = cli.main(["check", "model.py", "--python", sys.executable])
    out = capsys.readouterr().out

    assert code == 1  # findings, not a worker failure
    assert "shape-mismatch" in out
    assert "src/model.py" in out


def test_a_bare_variadic_is_the_same_batch_in_every_argument(project):
    paths, config = project(
        HEADER
        + """
    def add(x: Float[Tensor, "... d"], y: Float[Tensor, "... d"]) -> Float[Tensor, "... d"]:
        return x + y
    """
    )
    report = check_paths(paths, config)
    assert report.diagnostics == []
    assert report.worker_error is None


def test_a_correct_async_forward_is_clean(project):
    paths, config = project(
        HEADER
        + """
    class Net(nn.Module):
        async def forward(self, x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
            return x * 2
    """
    )
    report = check_paths(paths, config)
    assert report.diagnostics == []
    assert report.worker_error is None


def test_a_wrong_async_forward_reports_the_shape_not_the_coroutine(project):
    paths, config = project(
        HEADER
        + """
    class Net(nn.Module):
        async def forward(self, x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
            return x.sum(dim=-1)
    """
    )
    found = rules(check_paths(paths, config))
    assert "rank-mismatch" in found
    assert "not-a-tensor" not in found


def test_an_async_target_leaves_no_unawaited_coroutine_warning(tmp_path):
    """The worker's JSON protocol: stdout is the result, stderr is the user's."""
    path = tmp_path / "model.py"
    path.write_text(
        textwrap.dedent(HEADER)
        + textwrap.dedent(
            """
            async def good(x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
                return x * 2

            async def bad(x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
                raise ValueError("boom")
            """
        )
    )
    completed = subprocess.run(
        [sys.executable, "-m", "torchtyc.worker"],
        input=json.dumps({"paths": [str(path)], "variadic_rank": 2, "hover": False}),
        capture_output=True,
        text=True,
        check=False,
    )
    assert "never awaited" not in completed.stderr
    found = {d["rule"] for d in json.loads(completed.stdout)["diagnostics"]}
    assert found == {"trace-error"}


def test_a_flattened_pair_of_anonymous_axes_leaks_no_prime(project):
    paths, config = project(
        HEADER
        + """
    def flat(x: Float[Tensor, "... d"]) -> Float[Tensor, "... d"]:
        merged = x.flatten(0, 1)
        return merged @ merged
    """
    )
    report = check_paths(paths, config)
    message = next(d.message for d in report.diagnostics if d.rule == "trace-error")
    assert not [n for n in re.findall(r"\d+", message) if int(n) >= FIRST_PRIME]


def test_a_class_with_a_guarded_init_uses_the_one_that_ran(project):
    paths, config = project(
        HEADER
        + """
    FAST = True

    class Block(nn.Module):
        if FAST:
            def __init__(self, d_model: int) -> None:
                super().__init__()
                self.W: Float[nn.Parameter, "d_model d_model"] = nn.Parameter(
                    torch.empty((d_model, d_model))
                )
        else:
            def __init__(self, d_model: int, extra: int) -> None:
                super().__init__()
                self.W: Float[nn.Parameter, "d_model d_model"] = nn.Parameter(
                    torch.empty((d_model, extra))
                )
    """
    )
    report = check_paths(paths, config)
    assert report.diagnostics == []
    assert report.worker_error is None


def test_a_forgotten_await_in_a_sync_forward_is_still_reported(project):
    paths, config = project(
        HEADER
        + """
    class Net(nn.Module):
        async def _helper(self, x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
            return x * 2

        def forward(self, x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
            return self._helper(x)
    """
    )
    report = check_paths(paths, config)
    diagnostic = next(d for d in report.diagnostics if d.function == "Net.forward")
    assert diagnostic.rule == "not-a-tensor"
    assert "coroutine" in diagnostic.message


def test_an_attribute_of_a_guarded_init_is_checked_when_that_arm_is_live(project):
    paths, config = project(
        HEADER
        + """
    FAST = True

    class Block(nn.Module):
        if FAST:
            def __init__(self, d: int) -> None:
                super().__init__()
                self.W: Float[nn.Parameter, "d d"] = nn.Parameter(torch.empty((d, 3)))
        else:
            def __init__(self, d: int) -> None:
                super().__init__()
                self.W = nn.Parameter(torch.empty((d, d)))
    """
    )
    report = check_paths(paths, config)
    assert "attribute-mismatch" in rules(report)
    assert report.worker_error is None


def test_a_class_whose_only_init_did_not_run_is_left_alone(project):
    paths, config = project(
        HEADER
        + """
    FAST = False

    class Block(nn.Module):
        if FAST:
            def __init__(self, d_model: int) -> None:
                super().__init__()
                self.W: Float[nn.Parameter, "d_model d_model"] = nn.Parameter(
                    torch.empty(d_model, d_model)
                )

        def forward(self, x: Float[Tensor, "b d_model"]) -> Float[Tensor, "b d_model"]:
            return x * 2
    """
    )
    report = check_paths(paths, config)
    assert report.diagnostics == []
    assert report.worker_error is None


def test_a_guarded_definition_behind_a_same_file_decorator_is_still_traced(project):
    paths, config = project(
        HEADER
        + """
    import functools
    import sys

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapper

    if sys.version_info >= (3, 0):

        @deco
        def flip(x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
            return x.T
    """
    )
    report = check_paths(paths, config)
    assert "shape-mismatch" in rules(report)


def test_a_guarded_definition_behind_a_bare_decorator_is_still_traced(project):
    paths, config = project(
        HEADER
        + """
    import sys

    def deco(fn):
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapper

    if sys.version_info >= (3, 0):

        @deco
        def flip(x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
            return x.T
    """
    )
    report = check_paths(paths, config)
    assert "shape-mismatch" in rules(report)


def test_a_path_with_no_python_files_exits_as_a_tool_failure(tmp_path, monkeypatch, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(tmp_path)

    code = cli.main(["check", "empty"])
    err = capsys.readouterr().err

    # 2 is "torchtyc could not do the job", which is what a mistyped path is.
    # Exit 1 would read as findings and let a misconfigured CI step pass quietly.
    assert code == 2
    assert "no python files found" in err


def test_a_constructor_default_is_used_rather_than_synthesised(project):
    paths, config = project(
        HEADER
        + """
    class Block(nn.Module):
        def __init__(self, d_model: int, scale: float = 2.0, wide: bool = True) -> None:
            super().__init__()
            self.out = d_model * 2 if wide else d_model
            self.scale = scale
            self.W = nn.Parameter(torch.empty((self.out, d_model)))

        def forward(self, x: Float[Tensor, "b d_model"]) -> Float[Tensor, "b d_model"]:
            return x @ self.W.T * self.scale
    """
    )
    report = check_paths(paths, config)
    # `wide` defaults to True, so `self.out` is `d_model * 2` and the return
    # names the wrong axis. Synthesising `False` for the bool instead would
    # take the other branch and report nothing.
    assert "shape-mismatch" in rules(report)


def test_a_dimension_name_still_outranks_its_own_default(project):
    paths, config = project(
        HEADER
        + """
    class Block(nn.Module):
        def __init__(self, d_model: int = 4) -> None:
            super().__init__()
            self.W = nn.Parameter(torch.empty((d_model, d_model)))

        def forward(self, x: Float[Tensor, "b d_model"]) -> Float[Tensor, "b d_model"]:
            return x @ self.W
    """
    )
    report = check_paths(paths, config)
    assert not [d for d in report.diagnostics if d.severity.name == "ERROR"]


def test_a_device_default_does_not_pull_the_trace_off_meta(project):
    paths, config = project(
        HEADER
        + """
    class Block(nn.Module):
        def __init__(self, d_model: int, device: torch.device | None = None) -> None:
            super().__init__()
            # Resolving `None` to a concrete device would allocate for real,
            # which is what tracing on meta exists to avoid.
            self.dev = device if device is not None else torch.device("cpu")
            self.W = nn.Parameter(torch.empty((d_model, d_model), device=self.dev))

        def forward(self, x: Float[Tensor, "b d_model"]) -> Float[Tensor, "b d_model"]:
            return x @ self.W
    """
    )
    report = check_paths(paths, config)
    assert not [d for d in report.diagnostics if d.severity.name == "ERROR"]


def test_a_tuple_parameter_keeps_its_non_array_members(project):
    paths, config = project(
        HEADER
        + """
    def unpack(pair: tuple[Float[Tensor, "a b"], int]) -> Float[Tensor, "a b"]:
        x, n = pair
        return x * n
    """
    )
    report = check_paths(paths, config)
    # Dropping the `int` would build a 1-tuple and raise "not enough values to
    # unpack" against code that is correct.
    assert not [d for d in report.diagnostics if d.severity.name == "ERROR"]


def test_a_cached_property_is_not_traced(project):
    paths, config = project(
        HEADER
        + """
    from functools import cached_property

    class Block(nn.Module):
        def __init__(self, n: int) -> None:
            super().__init__()
            self.n = n

        @cached_property
        def mask(self) -> Float[Tensor, "n n"]:
            return torch.ones(self.n, self.n)

        def forward(self, x: Float[Tensor, "b n"]) -> Float[Tensor, "b n"]:
            return x
    """
    )
    report = check_paths(paths, config)
    assert not [d for d in report.diagnostics if d.severity.name == "ERROR"]


def test_a_traceback_keeps_a_frame_line_that_looks_like_a_prime(tmp_path):
    """`line 101` is a source position, not a size, so renaming leaves it alone.

    The failing statement is put on line `FIRST_PRIME` on purpose: renaming the
    whole traceback would turn that frame's line number into an axis name.
    """
    head = [
        "import torch",
        "from jaxtyping import Float",
        "from torch import Tensor",
        "",
    ]
    tail = [
        'def boom(x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:',
        "    return helper(x)",
        "",
        "def helper(x):",
    ]
    failing = "    return x.reshape(3, 5, 7)"
    padding = ["# pad"] * (FIRST_PRIME - 1 - len(head) - len(tail))
    lines = head + padding + tail + [failing]
    assert lines.index(failing) + 1 == FIRST_PRIME

    path = tmp_path / "model.py"
    path.write_text("\n".join(lines) + "\n")
    config = Config(root=tmp_path, python=sys.executable)
    report = check_paths([str(path)], config)

    diagnostic = next(d for d in report.diagnostics if d.rule == "trace-error")
    assert diagnostic.line + 1 == FIRST_PRIME
    assert f"line {FIRST_PRIME}" in (diagnostic.traceback or "")
    # The message beside it still has the primes taken back out.
    assert str(FIRST_PRIME) not in diagnostic.message
    assert "b" in diagnostic.message and "d" in diagnostic.message


def test_trace_error_anchors_to_the_caller_not_the_shared_layer(project):
    paths, config = project(
        HEADER
        + """
    class Linear(nn.Module):
        def __init__(self, in_features: int, out_features: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.empty((out_features, in_features)))

        def forward(self, x: Float[Tensor, "... in_features"]) -> Float[Tensor, "... out_features"]:
            return einsum(
                x, self.weight, "... in_features, out_features in_features -> ... out_features"
            )

    class Block(nn.Module):
        def __init__(self, d_model: int) -> None:
            super().__init__()
            self.w1 = Linear(64, d_model)

        def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
            return self.w1(x)
    """
    )
    report = check_paths(paths, config)
    diagnostic = next(d for d in report.diagnostics if d.rule == "trace-error")
    # `Linear.forward` is correct and shared. The wrong shape is passed by
    # `Block.forward`, so that is the line worth underlining.
    assert diagnostic.function == "Block.forward"
    assert diagnostic.line == 22
    assert diagnostic.hint is not None
    # The shapes on the failing line say which side carries the wrong width: the
    # weight was built as (d_model, 64), so its input axis is 64, not d_model.
    assert diagnostic.hint == (
        "raised further down, in `Linear.forward` at line 13, "
        "where x is (..., d_model), self.weight is (d_model, 64)"
    )


def test_a_trace_error_reports_the_shapes_on_the_line_that_raised(project):
    paths, config = project(
        HEADER
        + """
    class Block(nn.Module):
        def __init__(self, d_model: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.empty((64, 32)))

        def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
            return einsum(x, self.weight, "... d, out d -> ... out")
    """
    )
    report = check_paths(paths, config)
    diagnostic = next(d for d in report.diagnostics if d.rule == "trace-error")
    # Nothing was raised further down, so the hint is the shapes alone. `d` and
    # `out` are einops axis names inside a string, not locals, and stay out of it.
    assert diagnostic.hint == "x is (..., d_model), self.weight is (64, 32)"


def test_a_width_computed_in_init_is_named_by_what_it_follows(project):
    paths, config = project(
        HEADER
        + """
    class Linear(nn.Module):
        def __init__(self, in_features: int, out_features: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.empty((out_features, in_features)))

        def forward(self, x: Float[Tensor, "... in_features"]) -> Float[Tensor, "... out_features"]:
            return einsum(x, self.weight, "... a, b a -> ... b")

    class SwiGLU(nn.Module):
        def __init__(self, d_model: int) -> None:
            super().__init__()
            d_ff = round(((8 / 3) * d_model) / 64) * 64
            self.w1 = Linear(d_ff, d_model)

        def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
            return self.w1(x)
    """
    )
    report = check_paths(paths, config)
    diagnostic = next(d for d in report.diagnostics if d.rule == "trace-error")
    # `d_ff` is 256 only because the trace ran with a stand-in `d_model`. At any
    # real width it is a different number, so the number is not what to print.
    assert "256" not in diagnostic.message
    assert "<from d_model>" in diagnostic.message
    assert diagnostic.hint is not None
    assert "self.weight is (d_model, <from d_model>)" in diagnostic.hint
    assert diagnostic.note is not None
    assert "your __init__ computed from d_model" in diagnostic.note


def test_a_size_written_in_the_code_keeps_its_number(project):
    paths, config = project(
        HEADER
        + """
    class Block(nn.Module):
        def __init__(self, d_model: int) -> None:
            super().__init__()
            self.up = nn.Parameter(torch.empty((64, d_model)))
            self.down = nn.Parameter(torch.empty((d_model, 512)))

        def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
            h = einsum(x, self.up, "... d, up d -> ... up")
            return einsum(h, self.down, "... up, d up -> ... d")
    """
    )
    report = check_paths(paths, config)
    diagnostic = next(d for d in report.diagnostics if d.rule == "trace-error")
    # 64 and 512 do not move when d_model moves, so they are real widths the
    # code writes down, and the reader can go and find them.
    assert diagnostic.hint == "h is (..., 64), self.down is (d_model, 512)"
    assert diagnostic.note is None


def test_a_suggestion_keeps_the_leading_space_the_file_writes(project):
    paths, config = project(
        HEADER
        + """
    class Linear(nn.Module):
        def __init__(self, d_in: int, d_out: int) -> None:
            super().__init__()
            self.w: Float[nn.Parameter, " d_out d_in"] = nn.Parameter(torch.empty((d_out, d_in)))

        def forward(self, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_in"]:
            return einsum(x, self.w, "... d_in, d_out d_in -> ... d_out")
    """
    )
    report = check_paths(paths, config)
    diagnostic = next(d for d in report.diagnostics if d.rule == "shape-mismatch")
    # Pasting a suggestion that dropped the space would hand the reader a UP037
    # to fix, so the annotation goes back the way the file writes them.
    assert diagnostic.suggestion == 'Float[Tensor, " ... d_out"]'


def test_a_leading_space_annotation_traces_clean(project):
    paths, config = project(
        HEADER
        + """
    class Linear(nn.Module):
        def __init__(self, d_in: int, d_out: int) -> None:
            super().__init__()
            self.w: Float[nn.Parameter, " d_out d_in"] = nn.Parameter(torch.empty((d_out, d_in)))

        def forward(self, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
            return einsum(x, self.w, "... d_in, d_out d_in -> ... d_out")

    def tokens(ids: Int[Tensor, " ..."]) -> Int[Tensor, " ..."]:
        return ids
    """
    )
    report = check_paths(paths, config)
    assert rules(report) == []


def test_multi_head_attention_splitting_a_width_is_clean(project):
    paths, config = project(
        HEADER
        + """
    class MHA(nn.Module):
        def __init__(self, d_model: int, n_heads: int = 8) -> None:
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.qkv = nn.Parameter(torch.empty((3 * d_model, d_model)))
            self.out = nn.Parameter(torch.empty((d_model, d_model)))

        def forward(
            self, x: Float[Tensor, "batch seq d_model"]
        ) -> Float[Tensor, "batch seq d_model"]:
            b, s, _ = x.shape
            qkv = (x @ self.qkv.T).view(b, s, 3, self.n_heads, self.head_dim)
            q, k, v = qkv.unbind(2)
            scores = q.transpose(1, 2) @ k.transpose(1, 2).transpose(-2, -1)
            h = torch.softmax(scores, dim=-1) @ v.transpose(1, 2)
            return h.transpose(1, 2).reshape(b, s, -1) @ self.out.T
    """
    )
    report = check_paths(paths, config)
    # `d_model // n_heads` loses a remainder at any prime width, so tracing on
    # primes alone reports this correct block as a shape error. The retry runs
    # it at a width the eight the constructor writes down divides.
    assert report.diagnostics == []


def test_a_dimension_named_by_a_defaulted_parameter_takes_the_default(project):
    paths, config = project(
        HEADER
        + """
    class Heads(nn.Module):
        def __init__(self, d_model: int, n_heads: int = 8) -> None:
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads

        def forward(
            self, x: Float[Tensor, "batch seq d_model"]
        ) -> Float[Tensor, "batch n_heads seq head_dim"]:
            b, s, _ = x.shape
            return x.view(b, s, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
    """
    )
    report = check_paths(paths, config)
    # `n_heads` names an axis and states a width. On the retry the width wins,
    # so the split comes out whole and the axis binds to eight.
    assert rules(report) == ["unused-dim"]


def test_a_width_split_twice_is_clean(project):
    paths, config = project(
        HEADER
        + """
    class Rotary(nn.Module):
        def __init__(self, d_model: int, n_heads: int = 8) -> None:
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads

        def forward(
            self, x: Float[Tensor, "batch seq d_model"]
        ) -> Float[Tensor, "batch seq d_model"]:
            b, s, _ = x.shape
            pairs = x.view(b, s, self.n_heads, self.head_dim // 2, 2)
            return pairs.reshape(b, s, -1)
    """
    )
    report = check_paths(paths, config)
    # The head width is split again, so a factor that survives one division is
    # not enough. The second retry multiplies the written widths instead.
    assert report.diagnostics == []


def test_a_wrong_width_beside_a_split_is_still_reported(project):
    paths, config = project(
        HEADER
        + """
    class MHA(nn.Module):
        def __init__(self, d_model: int, n_heads: int = 8) -> None:
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.out = nn.Parameter(torch.empty((d_model, self.head_dim)))

        def forward(
            self, x: Float[Tensor, "batch seq d_model"]
        ) -> Float[Tensor, "batch seq d_model"]:
            b, s, _ = x.shape
            h = x.view(b, s, self.n_heads, self.head_dim).reshape(b, s, -1)
            return h @ self.out.T
    """
    )
    report = check_paths(paths, config)
    diagnostic = next(d for d in report.diagnostics if d.rule == "trace-error")
    # The `view` above is correct and only failed while widths did not divide.
    # The projection built at the head width is the real mistake, so that is
    # the line to underline.
    assert diagnostic.line == 18
    assert diagnostic.hint == "h is (batch, seq, d_model), self.out.T is (<from d_model>, d_model)"


def test_a_split_that_no_written_width_repairs_is_reported(project):
    paths, config = project(
        HEADER
        + """
    class Split(nn.Module):
        def forward(self, x: Float[Tensor, "batch d"]) -> Float[Tensor, "batch d"]:
            b, d = x.shape
            return x.reshape(b, d // 7, 7)
    """
    )
    report = check_paths(paths, config)
    # Seven is written down, so it is tried, and the reshape still adds an axis
    # the annotation does not have.
    assert rules(report) == ["rank-mismatch"]


def test_a_returned_width_the_init_computed_is_named_not_numbered(project):
    paths, config = project(
        HEADER
        + """
    class FF(nn.Module):
        def __init__(self, d_model: int) -> None:
            super().__init__()
            self.up = nn.Parameter(torch.empty((4 * d_model, d_model)))

        def forward(self, x: Float[Tensor, "batch d_model"]) -> Float[Tensor, "batch d_model"]:
            return x @ self.up.T
    """
    )
    report = check_paths(paths, config)
    diagnostic = next(d for d in report.diagnostics if d.rule == "shape-mismatch")
    # The return traces four times d_model, and that number exists only at the
    # width torchtyc traced with, so the message says what it follows instead.
    assert "<from d_model>" in diagnostic.message
    assert diagnostic.got == "(batch, <from d_model>)"
    assert diagnostic.note is not None
    assert "your __init__ computed from d_model" in diagnostic.note
    # It is a computed width, not another axis the annotation could have named,
    # so there is no swap to point at.
    assert diagnostic.hint is None


def test_a_width_kept_only_as_an_integer_is_named_by_what_it_follows(project):
    paths, config = project(
        HEADER
        + """
    class Heads(nn.Module):
        def __init__(self, d_model: int, n_heads: int = 8) -> None:
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads

        def forward(
            self, x: Float[Tensor, "batch seq d_model"]
        ) -> Float[Tensor, "batch seq d_model"]:
            b, s, _ = x.shape
            return x.view(b, s, self.n_heads, self.head_dim).sum(2)
    """
    )
    report = check_paths(paths, config)
    diagnostic = next(d for d in report.diagnostics if d.rule == "shape-mismatch")
    # `head_dim` is an integer attribute and no weight is ever built at that
    # width, so following the tensors alone would leave the message quoting a
    # number the model does not contain.
    assert "<from d_model>" in diagnostic.message
    assert diagnostic.got == "(batch, seq, <from d_model>)"


def test_a_parity_guard_beside_a_keyword_call_is_clean(project):
    paths, config = project(
        HEADER
        + """
    def check_positive(**kwargs):
        for key, value in kwargs.items():
            if value <= 0:
                raise ValueError(f"{key} must be positive")


    class RoPE(nn.Module):
        def __init__(self, theta: float, d_k: int, max_seq_len: int) -> None:
            super().__init__()
            check_positive(theta=theta, d_k=d_k, max_seq_len=max_seq_len)
            if d_k % 2 != 0:
                raise ValueError("RoPE dimension d_k should be divisible by 2")

        def forward(self, x: Float[Tensor, "... seq d_k"]) -> Float[Tensor, "... seq d_k"]:
            return x
    """
    )
    report = check_paths(paths, config)
    # The `2` in the guard compiles to an inline instruction and never reaches
    # the constants beside the keyword call, so reading the constants alone
    # finds no divisor and the retry never runs. Reading the bytecode sees it.
    assert report.diagnostics == []


def test_a_guarded_constructor_after_a_dead_one_is_used(project):
    paths, config = project(
        HEADER
        + """
    FAST = True

    class Block(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.W = nn.Parameter(torch.empty((d, d)))

        if FAST:

            def __init__(self, d: int, out: int) -> None:
                super().__init__()
                self.W = nn.Parameter(torch.empty((out, d)))

        def forward(self, x: Float[Tensor, "... d"]) -> Float[Tensor, "... out"]:
            return einsum(x, self.W, "... d, out d -> ... out")
    """
    )
    # The unconditional `__init__` never ran: the guard's arm reassigned the
    # name after it. Building with its parameters reports a missing `out`
    # against code that is correct.
    report = check_paths(paths, config)
    assert report.diagnostics == []
    assert report.ok


def test_a_dead_unconditional_constructor_contributes_no_attributes(project):
    paths, config = project(
        HEADER
        + """
    FAST = True

    class Block(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.W: Float[nn.Parameter, "d d"] = nn.Parameter(torch.empty((d, 3)))

        if FAST:

            def __init__(self, d: int, out: int) -> None:
                super().__init__()
                self.W: Float[nn.Parameter, "out d"] = nn.Parameter(torch.empty((out, d)))

        def forward(self, x: Float[Tensor, "... d"]) -> Float[Tensor, "... out"]:
            return einsum(x, self.W, "... d, out d -> ... out")
    """
    )
    # Neither the dead constructor's parameters nor its annotated attributes
    # take part in the check, so the wrong shape it writes down stays silent.
    report = check_paths(paths, config)
    assert report.diagnostics == []
    assert report.ok


def test_check_attributes_resolves_the_live_constructor_once(tmp_path):
    source = textwrap.dedent(
        HEADER
        + """
    class Block(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.W: Float[nn.Parameter, "d d"] = nn.Parameter(torch.empty((d, d)))
    """
    )
    path = tmp_path / "model.py"
    path.write_text(source)
    module = worker.import_from_path(path)
    info = next(c for c in scan_source(source, str(path)).classes if c.qualname == "Block")
    with (
        patch.object(worker, "live_init", wraps=tracing.live_init) as worker_init,
        patch.object(tracing, "live_init", wraps=tracing.live_init) as tracing_init,
    ):
        worker.check_attributes(module, info, str(path), 2)
    # `check_attributes` picks the constructor and hands it to `instantiate`,
    # so the resolution happens here and is not repeated inside.
    assert worker_init.call_count + tracing_init.call_count == 1


def test_constructor_thread_does_not_survive_trace(tmp_path):
    source = textwrap.dedent(
        HEADER
        + """
    import threading

    spawned = None

    class CooperativeWatcher(threading.Thread):
        def __init__(self) -> None:
            super().__init__(daemon=True)
            self._stopped = threading.Event()

        def run(self) -> None:
            self._stopped.wait(10.0)

        def stop(self) -> None:
            self._stopped.set()

    class Block(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            global spawned
            spawned = CooperativeWatcher()
            spawned.start()
            self.W = nn.Parameter(torch.empty((d, d)))

        def forward(self, x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
            return x @ self.W
    """
    )
    path = tmp_path / "model.py"
    path.write_text(source)
    module = worker.import_from_path(path)
    scan = scan_source(source, str(path))
    target = next(t for t in scan.targets if t.has_array_annotation)
    diags, _ = worker.check_target(module, target, str(path), 2, scan.targets)
    assert diags == []
    assert module.spawned is not None
    # The worker thread was started in construction, but was stopped cooperatively
    # and joined before the trace returned, so it does not outlive the check.
    assert not module.spawned.is_alive()


def test_unstoppable_constructor_thread_is_reported(project):
    paths, config = project(
        HEADER
        + """
    import threading
    import time

    class Stubborn(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            def loop():
                while True:
                    try:
                        time.sleep(0.01)
                    except BaseException:
                        pass

            t = threading.Thread(target=loop, name="StubbornWatcher", daemon=True)
            t.start()
            self.W = nn.Parameter(torch.empty((d, d)))

        def forward(self, x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
            return x @ self.W
    """
    )
    report = check_paths(paths, config)
    # A thread that ignores shutdown cannot be stopped cleanly, so it is reported
    # as uninstantiable rather than left running silently in the background.
    assert "uninstantiable" in rules(report)
    diagnostic = next(d for d in report.diagnostics if d.rule == "uninstantiable")
    assert "StubbornWatcher" in diagnostic.message


def test_constructor_error_wins_over_thread_check(project):
    paths, config = project(
        HEADER
        + """
    import threading
    import time

    class Broken(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            def loop():
                time.sleep(5.0)
            t = threading.Thread(target=loop, name="Watcher", daemon=True)
            t.start()
            raise ValueError("d must be even, this is the real error")

        def forward(self, x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
            return x
    """
    )
    report = check_paths(paths, config)
    assert len(report.diagnostics) == 1
    diag = report.diagnostics[0]
    assert diag.rule == "trace-error"
    assert "ValueError: d must be even, this is the real error" in diag.message


def test_constructor_thread_blocked_in_c_does_not_deadlock(project):
    paths, config = project(
        HEADER
        + """
    import threading

    lock = threading.Lock()
    lock.acquire()

    class BlockedInLock(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            def wait_for_lock():
                lock.acquire()
            t = threading.Thread(target=wait_for_lock, name="LockWaiter", daemon=True)
            t.start()
            self.W = nn.Parameter(torch.empty((d, d)))

        def forward(self, x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
            return x @ self.W
    """
    )
    report = check_paths(paths, config)
    assert "uninstantiable" in rules(report)
    diag = next(d for d in report.diagnostics if d.rule == "uninstantiable")
    assert "LockWaiter" in diag.message


def test_trace_retry_construction_count_is_capped(tmp_path):
    source = textwrap.dedent(
        HEADER
        + """
    class MultiFail(nn.Module):
        def __init__(self, d_model: int, n_heads: int = 8, d_k: int = 2, d_v: int = 4) -> None:
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.out = nn.Parameter(torch.empty((d_model, self.head_dim)))

        def forward(
            self, x: Float[Tensor, "batch seq d_model"]
        ) -> Float[Tensor, "batch seq d_model"]:
            b, s, _ = x.shape
            h = x.view(b, s, self.n_heads, self.head_dim).reshape(b, s, -1)
            return h @ self.out.T
    """
    )
    path = tmp_path / "model.py"
    path.write_text(source)
    module = worker.import_from_path(path)
    scan = scan_source(source, str(path))
    target = next(t for t in scan.targets if t.has_array_annotation)

    with patch.object(tracing, "instantiate", wraps=tracing.instantiate) as mock_instantiate:
        worker.check_target(module, target, str(path), 2, scan.targets)

    # One initial trace, two divisible scale retries, and one derived size probe.
    # Verify that retries occurred and that total constructions are bounded.
    assert mock_instantiate.call_count > 1
    assert mock_instantiate.call_count == 4
