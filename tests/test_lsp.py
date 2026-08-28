import pytest

pytest.importorskip("pygls")

from lsprotocol import types as lsp

from torchtyc.diagnostics import Diagnostic, Severity
from torchtyc.lsp import _target_at, path_to_uri, to_lsp, uri_to_path


def test_uri_roundtrip(tmp_path):
    path = tmp_path / "a b" / "model.py"
    path.parent.mkdir()
    path.write_text("")
    assert uri_to_path(path_to_uri(str(path))) == str(path.resolve())


def test_severity_mapping():
    for severity, expected in [
        (Severity.ERROR, lsp.DiagnosticSeverity.Error),
        (Severity.WARNING, lsp.DiagnosticSeverity.Warning),
        (Severity.INFO, lsp.DiagnosticSeverity.Information),
    ]:
        diagnostic = Diagnostic(
            path="a.py", line=0, column=0, rule="r", message="m", severity=severity
        )
        assert to_lsp(diagnostic).severity == expected


def test_message_carries_expected_got_and_hint():
    diagnostic = Diagnostic(
        path="a.py",
        line=2,
        column=4,
        rule="shape-mismatch",
        message="bad",
        expected="(a, b)",
        got="(b, a)",
        hint="swapped",
    )
    converted = to_lsp(diagnostic)
    assert "expected: (a, b)" in converted.message
    assert "got:      (b, a)" in converted.message
    assert "hint: swapped" in converted.message
    assert converted.code == "shape-mismatch"
    assert converted.source == "torchtyc"


def test_range_defaults_when_end_is_missing():
    diagnostic = Diagnostic(path="a.py", line=3, column=5, rule="r", message="m")
    converted = to_lsp(diagnostic)
    assert converted.range.start == lsp.Position(line=3, character=5)
    assert converted.range.end == lsp.Position(line=3, character=6)


SOURCE = """
from jaxtyping import Float
from torch import Tensor


def first(x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
    return x


CONSTANT = 3
"""


def scanned_server():
    from types import SimpleNamespace

    from torchtyc.discovery import scan_source

    return SimpleNamespace(scans={"file:///a.py": scan_source(SOURCE, "a.py")})


def test_target_at_finds_the_enclosing_function():
    line = SOURCE.splitlines().index("    return x")
    target = _target_at(scanned_server(), "file:///a.py", line)
    assert target is not None
    assert target.qualname == "first"


def test_target_at_stops_at_the_end_of_the_function():
    line = SOURCE.splitlines().index("CONSTANT = 3")
    assert _target_at(scanned_server(), "file:///a.py", line) is None
