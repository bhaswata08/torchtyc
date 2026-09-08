import pytest

pytest.importorskip("pygls")

from lsprotocol import types as lsp

from torchtyc.config import Overrides
from torchtyc.diagnostics import Diagnostic, Severity
from torchtyc.lsp import TorchtycServer, _target_at, did_close, path_to_uri, to_lsp, uri_to_path


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


def inlay_positions(source: str, shapes: dict[str, dict[str, str]]) -> list[tuple[int, int]]:
    """Run the real inlay hint handler over a source string."""
    from types import SimpleNamespace

    from torchtyc.discovery import scan_source
    from torchtyc.engine import Report
    from torchtyc.lsp import inlay_hints

    uri = "file:///a.py"
    path = uri_to_path(uri)
    lines = [line + "\n" for line in source.splitlines()]
    document = SimpleNamespace(lines=lines)
    ls = SimpleNamespace(
        workspace=SimpleNamespace(get_text_document=lambda _: document),
        reports={uri: Report(hovers={path: shapes})},
        scans={uri: scan_source(source, path)},
    )
    params = lsp.InlayHintParams(
        text_document=lsp.TextDocumentIdentifier(uri=uri),
        range=lsp.Range(
            start=lsp.Position(line=0, character=0),
            end=lsp.Position(line=len(lines), character=0),
        ),
    )
    return [(h.position.line, h.position.character) for h in inlay_hints(ls, params)]


SOURCE_ONE_LINE = """from jaxtyping import Float
from torch import Tensor


def forward(x: Float[Tensor, "a"]) -> Float[Tensor, "a"]:
    return x
"""


def test_inlay_hint_lands_after_the_whole_signature():
    positions = inlay_positions(SOURCE_ONE_LINE, {"forward": {"return": "float32[(a,)]"}})
    assert len(positions) == 1
    line, character = positions[0]
    text = SOURCE_ONE_LINE.splitlines()[line]
    # The hint must not split `def forward` from its parameter list, which is
    # what anchoring at the end of the function name did.
    assert text.lstrip().startswith("def forward")
    assert character == len(text)
    assert character > text.index("(")


SOURCE_WRAPPED = """from jaxtyping import Float
from torch import Tensor


def forward(
    x: Float[Tensor, "a b"],
    y: Float[Tensor, "b c"],
) -> Float[Tensor, "a c"]:
    return x @ y
"""


def test_inlay_hint_follows_a_wrapped_signature_to_its_last_line():
    positions = inlay_positions(SOURCE_WRAPPED, {"forward": {"return": "float32[(a, c)]"}})
    assert len(positions) == 1
    line, character = positions[0]
    text = SOURCE_WRAPPED.splitlines()[line]
    assert text.strip() == ') -> Float[Tensor, "a c"]:'
    assert character == len(text)


def test_only_one_trace_runs_at_a_time_per_file():
    """A queued trace is dropped rather than starting a second worker.

    Cancelling the task that awaits a trace does not stop the subprocess
    already running in its thread, so overlapping traces would each import
    torch and run to completion.
    """
    import asyncio

    server = TorchtycServer()
    running = 0
    peak = 0
    traced = 0

    async def fake_trace_once(uri: str, generation: int | None = None) -> None:
        nonlocal running, peak, traced
        running += 1
        peak = max(peak, running)
        traced += 1
        await asyncio.sleep(0.05)
        running -= 1

    server._trace_once = fake_trace_once

    async def drive() -> None:
        await asyncio.gather(*(server.trace_now("file:///a.py") for _ in range(5)))

    asyncio.run(drive())

    assert peak == 1
    # The first runs, the last wins, and the three superseded in between are
    # skipped instead of each starting a worker.
    assert traced == 2


def close_params(uri: str) -> lsp.DidCloseTextDocumentParams:
    return lsp.DidCloseTextDocumentParams(text_document=lsp.TextDocumentIdentifier(uri=uri))


def test_close_drops_per_file_trace_state():
    """Closing a file releases its lock and generation, not just its report.

    `did_close` used to pop only pending, reports and scans, so each distinct
    URI ever opened left a lock and an int behind for the life of the server.
    """
    import asyncio

    server = TorchtycServer()
    server.publish = lambda _uri, _diagnostics: None

    async def fake_trace_once(uri: str, generation: int | None = None) -> None:
        return None

    server._trace_once = fake_trace_once

    async def drive() -> None:
        for i in range(5):
            await server.trace_now(f"file:///{i}.py")

    asyncio.run(drive())
    assert len(server.tracing) == 5
    assert len(server.wanted) == 5

    for i in range(5):
        did_close(server, close_params(f"file:///{i}.py"))

    assert server.tracing == {}
    assert server.wanted == {}
    assert server.reports == {}
    assert server.scans == {}
    assert server.pending == {}


def test_close_mid_trace_publishes_nothing_afterwards(monkeypatch):
    """A worker that finishes after close must not revive the file's state.

    Cancelling the debounce task does not stop a worker already running in
    its thread, so the trace has to notice the close when it finishes and
    drop its result instead of storing a report and publishing diagnostics.
    """
    import asyncio
    import threading

    from torchtyc import lsp as lsp_module
    from torchtyc.engine import Report

    server = TorchtycServer()
    published: list = []
    server.publish = lambda uri, diagnostics: published.append((uri, diagnostics))
    server.source_of = lambda _uri: "x = 1\n"

    entered = threading.Event()
    release = threading.Event()

    def slow_check(paths, config, sources=None, hover=False, **kwargs):
        entered.set()
        assert release.wait(timeout=10)
        return Report()

    monkeypatch.setattr(lsp_module, "check_paths", slow_check)

    uri = "file:///a.py"

    async def drive() -> None:
        task = asyncio.create_task(server.trace_now(uri))
        assert await asyncio.to_thread(entered.wait, 10)
        did_close(server, close_params(uri))
        assert uri not in server.tracing
        assert uri not in server.wanted
        # The close itself clears the editor. Nothing may follow it.
        assert published[-1] == (uri, [])
        seen = len(published)
        release.set()
        await task
        assert len(published) == seen
        assert uri not in server.reports

    asyncio.run(drive())


def test_stale_worker_does_not_overwrite_a_fresh_trace_after_reopen(monkeypatch):
    """A close must not let the old worker masquerade as the new trace.

    Generations restart from zero per file, so without a server-wide counter
    the reopened file would get the same number the stale worker holds and
    the stale result, finishing last, would overwrite the fresh one.
    """
    import asyncio
    import threading

    from torchtyc import lsp as lsp_module
    from torchtyc.engine import Report

    server = TorchtycServer()
    published: list = []
    server.publish = lambda uri, diagnostics: published.append((uri, list(diagnostics)))
    server.source_of = lambda _uri: "x = 1\n"

    entered = threading.Event()
    release_stale = threading.Event()
    calls = 0

    def flaky_check(paths, config, sources=None, hover=False, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            assert release_stale.wait(timeout=10)
            return Report(
                diagnostics=[Diagnostic(path=paths[0], line=9, column=0, rule="r", message="stale")]
            )
        return Report(
            diagnostics=[Diagnostic(path=paths[0], line=1, column=0, rule="r", message="fresh")]
        )

    monkeypatch.setattr(lsp_module, "check_paths", flaky_check)

    uri = "file:///a.py"

    async def drive() -> None:
        stale = asyncio.create_task(server.trace_now(uri))
        assert await asyncio.to_thread(entered.wait, 10)
        did_close(server, close_params(uri))
        await server.trace_now(uri)
        assert server.reports[uri].diagnostics[0].message == "fresh"
        release_stale.set()
        await stale
        assert server.reports[uri].diagnostics[0].message == "fresh"
        assert [d.message for d in published[-1][1]] == ["fresh"]

    asyncio.run(drive())


def test_reopen_traces_again_after_close():
    """Closing a file must not break tracing it again afterwards."""
    import asyncio

    server = TorchtycServer()
    server.publish = lambda _uri, _diagnostics: None

    runs = 0

    async def fake_trace_once(uri: str, generation: int | None = None) -> None:
        nonlocal runs
        runs += 1

    server._trace_once = fake_trace_once

    uri = "file:///a.py"

    async def drive() -> None:
        await server.trace_now(uri)

    asyncio.run(drive())
    assert runs == 1
    did_close(server, close_params(uri))
    assert uri not in server.tracing
    assert uri not in server.wanted
    asyncio.run(drive())
    assert runs == 2
    assert uri in server.tracing
    assert uri in server.wanted


def cancelling_check_factory(entered, fresh):
    """A stand-in worker that blocks like a torch import but honours cancel.

    It behaves the way engine.run_worker does with a cancel flag: the first
    call blocks until the flag is set, then raises instead of returning a
    result. Later calls finish at once with the `fresh` report.
    """
    import threading

    from torchtyc.engine import WorkerCancelled

    calls = 0
    lock = threading.Lock()

    def check(paths, config, sources=None, hover=False, **kwargs):
        nonlocal calls
        with lock:
            calls += 1
            first = calls == 1
        cancel = kwargs.get("cancel")
        if first:
            entered.set()
            assert cancel is not None
            assert cancel.wait(timeout=10)
            raise WorkerCancelled()
        return fresh()

    return check


def test_superseded_trace_stops_its_worker(monkeypatch):
    """A newer trace kills the running worker instead of waiting behind it.

    The per-file lock alone only serializes traces: the stale run would still
    import torch to completion while the new trace waits, then spawn a second
    import. Here the first run must die as soon as the second is asked for,
    publish nothing, and leave the final result to the second.
    """
    import asyncio
    import threading
    import time

    from torchtyc import lsp as lsp_module
    from torchtyc.engine import Report

    server = TorchtycServer()
    published: list = []
    server.publish = lambda uri, diagnostics: published.append((uri, list(diagnostics)))
    server.source_of = lambda _uri: "x = 1\n"

    entered = threading.Event()
    fresh = Report(
        diagnostics=[Diagnostic(path="a.py", line=1, column=0, rule="r", message="fresh")]
    )
    monkeypatch.setattr(lsp_module, "check_paths", cancelling_check_factory(entered, lambda: fresh))

    uri = "file:///a.py"

    async def drive() -> None:
        stale = asyncio.create_task(server.trace_now(uri))
        assert await asyncio.to_thread(entered.wait, 10)
        started = time.monotonic()
        # The stale run never finishes on its own. If the new trace does not
        # stop it, this waits out the whole block below.
        await asyncio.wait_for(server.trace_now(uri), timeout=10)
        await stale
        return time.monotonic() - started

    elapsed = asyncio.run(drive())
    assert elapsed < 10
    assert server.reports[uri] is fresh
    assert [d.message for d in published[-1][1]] == ["fresh"]
    assert all(messages != ["stale"] for _, messages in published)


def test_save_mid_trace_does_not_stack_workers(monkeypatch):
    """Saving mid-trace is the guaranteed overlap: delay=0 skips the debounce.

    The save cancels the task awaiting the running worker. That cancel used
    to free the lock while the orphaned worker ran on, so the new trace
    spawned a second torch import beside it. Now the orphan dies first, so
    only one worker runs at a time and the save's trace still publishes.
    """
    import asyncio
    import threading

    from torchtyc import lsp as lsp_module
    from torchtyc.engine import Report

    server = TorchtycServer()
    published: list = []
    server.publish = lambda uri, diagnostics: published.append((uri, list(diagnostics)))
    server.source_of = lambda _uri: "x = 1\n"

    entered = threading.Event()
    lock = threading.Lock()
    saw_cancel: list = []
    events: list = []

    def check(paths, config, sources=None, hover=False, **kwargs):
        from torchtyc.engine import WorkerCancelled

        with lock:
            first = not entered.is_set()
            events.append(("start", "stale" if first else "fresh"))
        if not first:
            with lock:
                events.append(("complete", "fresh"))
            return Report(
                diagnostics=[Diagnostic(path=paths[0], line=1, column=0, rule="r", message="saved")]
            )
        entered.set()
        saw_cancel.append(kwargs["cancel"].wait(timeout=10))
        with lock:
            events.append(("stop", "stale"))
        raise WorkerCancelled()

    monkeypatch.setattr(lsp_module, "check_paths", check)

    uri = "file:///a.py"

    async def drive() -> None:
        await server.trace_soon(uri, delay=0.0)
        assert await asyncio.to_thread(entered.wait, 10)
        first = server.pending[uri]
        # What did_save does: no debounce, straight to a new trace.
        await server.trace_soon(uri, delay=0.0)
        second = server.pending[uri]
        await asyncio.gather(first, second, return_exceptions=True)

    asyncio.run(drive())
    assert saw_cancel == [True]
    # The stale run died instead of running to term, and only the save's
    # trace ran to completion and published.
    assert ("stop", "stale") in events
    assert ("complete", "stale") not in events
    assert ("complete", "fresh") in events
    assert server.reports[uri] is not None
    assert [d.message for d in published[-1][1]] == ["saved"]


def test_close_stops_the_running_worker(monkeypatch):
    """Closing mid-trace kills the worker, not just its result.

    Dropping the result keeps a stale publish away, but the orphaned worker
    would still run to completion or the timeout. The close must stop it.
    """
    import asyncio
    import threading
    import time

    from torchtyc import lsp as lsp_module

    server = TorchtycServer()
    published: list = []
    server.publish = lambda uri, diagnostics: published.append((uri, diagnostics))
    server.source_of = lambda _uri: "x = 1\n"

    entered = threading.Event()
    stopped = threading.Event()

    def check(paths, config, sources=None, hover=False, **kwargs):
        from torchtyc.engine import WorkerCancelled

        entered.set()
        assert kwargs["cancel"].wait(timeout=10)
        stopped.set()
        raise WorkerCancelled()

    monkeypatch.setattr(lsp_module, "check_paths", check)

    uri = "file:///a.py"

    async def drive() -> None:
        task = asyncio.create_task(server.trace_now(uri))
        assert await asyncio.to_thread(entered.wait, 10)
        started = time.monotonic()
        did_close(server, close_params(uri))
        await asyncio.wait_for(task, timeout=10)
        return time.monotonic() - started

    elapsed = asyncio.run(drive())
    assert elapsed < 10
    assert stopped.is_set()
    assert uri not in server.reports
    assert uri not in server.workers
    # The close itself clears the editor. Nothing may follow it.
    assert published == [(uri, [])]


def test_concurrent_traces_are_bounded_across_files(monkeypatch):
    """Each file has its own lock, so without a cap N open files run N workers.

    A config change rechecks every open file at once. The server-wide slots
    bound that fan-out while letting a few files recheck together.
    """
    import asyncio
    import threading
    import time

    from torchtyc import lsp as lsp_module
    from torchtyc.engine import Report
    from torchtyc.lsp import MAX_CONCURRENT_WORKERS

    server = TorchtycServer()
    server.publish = lambda _uri, _diagnostics: None
    server.source_of = lambda _uri: "x = 1\n"

    running = 0
    peak = 0
    entered = 0
    lock = threading.Lock()
    release = threading.Event()

    def check(paths, config, sources=None, hover=False, **kwargs):
        nonlocal running, peak, entered
        with lock:
            running += 1
            peak = max(peak, running)
            entered += 1
        try:
            assert release.wait(timeout=10)
            return Report()
        finally:
            with lock:
                running -= 1

    monkeypatch.setattr(lsp_module, "check_paths", check)

    uris = [f"file:///{i}.py" for i in range(MAX_CONCURRENT_WORKERS + 2)]

    async def drive() -> None:
        tasks = [asyncio.create_task(server.trace_now(uri)) for uri in uris]
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            with lock:
                if entered >= MAX_CONCURRENT_WORKERS:
                    break
            await asyncio.sleep(0.01)
        # Let any extra worker start if the cap is broken.
        await asyncio.sleep(0.3)
        with lock:
            assert entered == MAX_CONCURRENT_WORKERS
            assert peak == MAX_CONCURRENT_WORKERS
        release.set()
        await asyncio.gather(*tasks)

    asyncio.run(drive())


def test_cancelled_run_worker_terminates_and_discards_output(monkeypatch):
    """A stopped worker's half-written stdout is never parsed as a result.

    Killing a child mid-write leaves half a JSON document on the pipe. The
    run must raise instead of parsing it, and the child must be reaped.
    """
    import threading

    from torchtyc import engine as engine_module
    from torchtyc.config import Config
    from torchtyc.engine import WorkerCancelled, run_worker

    calls: list = []

    class FakeProc:
        pid = None

        def communicate(self, input=None, timeout=None):
            calls.append("communicate")
            raise AssertionError("a cancelled run must not read the pipes")

        def terminate(self):
            calls.append("terminate")

        def kill(self):
            calls.append("kill")

        def wait(self, timeout=None):
            calls.append("wait")
            return 0

    spawned: list = []

    def fake_popen(*args, **kwargs):
        proc = FakeProc()
        spawned.append(proc)
        return proc

    monkeypatch.setattr(engine_module.subprocess, "Popen", fake_popen)

    cancel = threading.Event()
    cancel.set()
    config = Config(root="/tmp", python="/nonexistent/python")
    with pytest.raises(WorkerCancelled):
        run_worker(["a.py"], config, cancel=cancel)
    assert spawned != []
    assert "terminate" in calls
    assert "wait" in calls
    assert "communicate" not in calls


def test_worker_timeout_reports_as_before(monkeypatch, tmp_path):
    """The plain path keeps its timeout message and reaps the child.

    run_worker is shared with the CLI, which never cancels. A slow worker
    must still report the same message and leave no process behind.
    """
    import sys

    from torchtyc import engine as engine_module
    from torchtyc.config import Config
    from torchtyc.engine import run_worker

    monkeypatch.setattr(engine_module, "_BOOTSTRAP", "import time; time.sleep(30)")
    config = Config(root=tmp_path, python=sys.executable)
    config.timeout = 0.5
    procs: list = []
    _, _, error = run_worker([str(tmp_path / "model.py")], config, on_proc=procs.append)
    assert error == "the trace timed out after 0.5s"
    assert len(procs) == 1
    assert procs[0].poll() is not None
