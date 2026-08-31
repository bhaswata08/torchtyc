"""End-to-end checks: a real file, a real torch import, a real subprocess."""

import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from torchtyc import cli
from torchtyc.binding import FIRST_PRIME
from torchtyc.config import Config
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
    config = Config(root=tmp_path, python=sys.executable)
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

    text = json_module.dumps(payload)
    primes = {str(p) for p in _PRIME_POOL[:8]}
    numbers = set(re.findall(r"\d+", text))
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
