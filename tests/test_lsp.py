import pytest

pytest.importorskip("pygls")

from lsprotocol import types as lsp

from torchtyc.config import Overrides
from torchtyc.diagnostics import Diagnostic, Severity
from torchtyc.lsp import TorchtycServer, _target_at, path_to_uri, to_lsp, uri_to_path


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


PYPROJECT = """
[tool.torchtyc]
python = "/from/toml/python"
variadic-rank = 5
severity = "error"
timeout = 42.0
"""


def project_root(tmp_path):
    (tmp_path / "pyproject.toml").write_text(PYPROJECT)
    return tmp_path


def test_config_for_reads_the_project_the_file_belongs_to(tmp_path):
    root = project_root(tmp_path)
    server = TorchtycServer()
    config = server.config_for(str(root / "model.py"))
    assert config.python == "/from/toml/python"
    assert config.variadic_rank == 5
    assert config.timeout == 42.0


def test_command_line_options_win_over_the_project_file(tmp_path):
    root = project_root(tmp_path)
    server = TorchtycServer()
    server.overrides = Overrides(
        python="/from/cli/python",
        variadic_rank=3,
        ignore=frozenset({"unused-dim"}),
        timeout=7.0,
    )
    config = server.config_for(str(root / "model.py"))
    assert config.python == "/from/cli/python"
    assert config.variadic_rank == 3
    assert config.timeout == 7.0
    assert "unused-dim" in config.ignore
    # Untouched by the command line, so the project file still decides.
    assert config.severity is Severity.ERROR


def test_serve_hands_the_command_line_options_to_the_server(monkeypatch, tmp_path):
    from torchtyc import lsp as lsp_module
    from torchtyc.cli import build_parser, cmd_lsp

    root = project_root(tmp_path)
    monkeypatch.setattr(lsp_module.server, "overrides", lsp_module.server.overrides)
    monkeypatch.setattr(lsp_module.server, "start_io", lambda: None)

    args = build_parser().parse_args(
        ["lsp", "--python", "/from/cli/python", "--variadic-rank", "4"]
    )
    assert cmd_lsp(args) == 0

    config = lsp_module.server.config_for(str(root / "model.py"))
    assert config.python == "/from/cli/python"
    assert config.variadic_rank == 4


LINT_SOURCE = """\
from jaxtyping import Float
from torch import Tensor


def pool(x: Float[Tensor, "batch d"]) -> Float[Tensor, "batch"]:{ignore}
    return x.sum(-1)
"""


def lint_rules(tmp_path, source: str) -> list[str]:
    """Rules the fast, on-every-keystroke pass would publish for this buffer."""
    path = tmp_path / "model.py"
    path.write_text(source)

    server = TorchtycServer()
    published: list[list[Diagnostic]] = []
    server.source_of = lambda _uri: source
    server.publish = lambda _uri, diagnostics: published.append(diagnostics)
    server.lint_now(path_to_uri(str(path)))
    return [d.rule for d in published[0]]


def test_the_fast_pass_reports_an_unsuppressed_lint_rule(tmp_path):
    assert "unused-dim" in lint_rules(tmp_path, LINT_SOURCE.format(ignore=""))


def test_the_fast_pass_honours_an_ignore_comment(tmp_path):
    source = LINT_SOURCE.format(ignore="  # torchtyc: ignore[unused-dim]")
    assert "unused-dim" not in lint_rules(tmp_path, source)


def test_suggestion_reaches_the_lsp_message():
    diagnostic = Diagnostic(
        path="a.py",
        line=1,
        column=0,
        rule="shape-mismatch",
        message="bad",
        suggestion='Float[Tensor, "... out_features"]',
    )
    assert 'try: Float[Tensor, "... out_features"]' in to_lsp(diagnostic).message


def quick_fix_edits(
    line_text: str, diagnostic: Diagnostic, anchor: str | None = None
) -> list[tuple[str, str]]:
    """Run the real code action handler over one line and one diagnostic.

    `anchor` is the substring the diagnostic points at, which is what the
    worker anchors real diagnostics to: the return annotation node, or the
    annotation of a `self.X` assignment. Passing it here rather than letting
    the handler search the line is the whole point, since a line can hold
    several annotations.

    Returns (replaced text, replacement) pairs so a test can assert the edit
    lands on the right span and not merely that the line ends up right.
    """
    from types import SimpleNamespace

    from torchtyc.lsp import code_action

    uri = "file:///a.py"
    document = SimpleNamespace(lines=[line_text + "\n"])
    ls = SimpleNamespace(workspace=SimpleNamespace(get_text_document=lambda _: document))

    if anchor is not None:
        start = line_text.index(anchor)
        diagnostic.column = start
        diagnostic.end_column = start + len(anchor)
        diagnostic.end_line = diagnostic.line

    params = lsp.CodeActionParams(
        text_document=lsp.TextDocumentIdentifier(uri=uri),
        range=lsp.Range(
            start=lsp.Position(line=0, character=0), end=lsp.Position(line=0, character=0)
        ),
        context=lsp.CodeActionContext(diagnostics=[to_lsp(diagnostic)]),
    )

    pairs = []
    for action in code_action(ls, params):
        if not action.title.startswith("Change the annotation"):
            continue
        edit = action.edit.changes[uri][0]
        replaced = line_text[edit.range.start.character : edit.range.end.character]
        pairs.append((replaced, edit.new_text))
    return pairs


def test_quick_fix_replaces_the_anchored_annotation():
    diagnostic = Diagnostic(
        path="a.py",
        line=0,
        column=0,
        rule="shape-mismatch",
        message="bad",
        suggestion='Float[Tensor, "b out_features"]',
    )
    line = 'def f(x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:'
    assert quick_fix_edits(line, diagnostic, anchor='Float[Tensor, "b d"]') == [
        ('Float[Tensor, "b d"]', 'Float[Tensor, "b out_features"]')
    ]


def test_quick_fix_for_a_dtype_mismatch_changes_the_dtype_class():
    diagnostic = Diagnostic(
        path="a.py",
        line=0,
        column=0,
        rule="dtype-mismatch",
        message="bad",
        suggestion='Int[Tensor, "b d"]',
    )
    line = 'def f(x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:'
    # The dims are unchanged, so rewriting only the dim string would be a
    # no-op. The whole annotation is replaced instead.
    assert quick_fix_edits(line, diagnostic, anchor='Float[Tensor, "b d"]') == [
        ('Float[Tensor, "b d"]', 'Int[Tensor, "b d"]')
    ]


def test_quick_fix_is_not_offered_when_it_would_change_nothing():
    diagnostic = Diagnostic(
        path="a.py",
        line=0,
        column=0,
        rule="shape-mismatch",
        message="bad",
        suggestion='Float[Tensor, "b d"]',
    )
    line = 'def f(x: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:'
    assert quick_fix_edits(line, diagnostic, anchor='Float[Tensor, "b d"]') == []


def test_quick_fix_rewrites_an_annotated_attribute():
    diagnostic = Diagnostic(
        path="a.py",
        line=0,
        column=0,
        rule="attribute-mismatch",
        message="bad",
        suggestion='Float[nn.Parameter, "d_in d_out"]',
    )
    line = '        self.W: Float[nn.Parameter, "d_out d_in"] = nn.Parameter(torch.empty((2, 3)))'
    assert quick_fix_edits(line, diagnostic, anchor='Float[nn.Parameter, "d_out d_in"]') == [
        ('Float[nn.Parameter, "d_out d_in"]', 'Float[nn.Parameter, "d_in d_out"]')
    ]


def test_quick_fix_declines_a_tuple_return():
    """The suggestion describes one element, the diagnostic anchors the tuple.

    Searching the line for an annotation would pick the last element and
    rewrite it with element zero's suggestion, silently corrupting the code.
    Offering nothing is the only safe answer until the diagnostic can anchor
    the element itself.
    """
    diagnostic = Diagnostic(
        path="a.py",
        line=0,
        column=0,
        rule="shape-mismatch",
        message="bad",
        suggestion='Float[Tensor, "a out"]',
    )
    line = 'def f(x) -> tuple[Float[Tensor, "a b"], Int[Tensor, "c"]]:'
    assert (
        quick_fix_edits(line, diagnostic, anchor='tuple[Float[Tensor, "a b"], Int[Tensor, "c"]]')
        == []
    )


def test_quick_fix_declines_a_multiline_annotation():
    diagnostic = Diagnostic(
        path="a.py",
        line=0,
        column=0,
        rule="shape-mismatch",
        message="bad",
        end_line=1,
        end_column=4,
        suggestion='Float[Tensor, "a"]',
    )
    line = "def f(x) -> Float["
    assert quick_fix_edits(line, diagnostic) == []
