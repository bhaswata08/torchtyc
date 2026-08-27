import pytest

pytest.importorskip("pygls")

from lsprotocol import types as lsp

from torchtyc.diagnostics import Diagnostic, Severity
from torchtyc.lsp import path_to_uri, to_lsp, uri_to_path


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
