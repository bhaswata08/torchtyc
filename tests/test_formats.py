"""The github style is a serialized protocol: GitHub parses these lines."""

from pathlib import Path

from torchtyc.diagnostics import Diagnostic, Severity
from torchtyc.engine import Report
from torchtyc.formats import render


def github(report: Report) -> list[str]:
    return render(report, "github", Path("/project")).splitlines()


def test_percent_and_newlines_are_escaped_in_a_message():
    report = Report(
        diagnostics=[
            Diagnostic(
                path="/project/model.py",
                line=3,
                column=0,
                rule="trace-error",
                severity=Severity.ERROR,
                message="ValueError: 50% off\r\nsecond line",
                hint="100% sure\nabout this",
            )
        ]
    )
    (line,) = github(report)
    assert line.startswith("::error file=model.py,line=4,col=1,title=torchtyc[trace-error]::")
    body = line.split("::", 2)[2]
    assert body == "ValueError: 50%25 off%0D%0Asecond line%0Ahint: 100%25 sure%0Aabout this"
    assert "\n" not in line and "\r" not in line


def test_worker_error_is_escaped_too():
    report = Report(worker_error="died at 90%\nno output")
    (line,) = github(report)
    assert line == "::error title=torchtyc::died at 90%25%0Ano output"


def test_a_comma_in_a_path_does_not_split_the_property_list():
    report = Report(
        diagnostics=[
            Diagnostic(
                path="/project/my,model.py",
                line=0,
                column=0,
                rule="shape-mismatch",
                severity=Severity.ERROR,
                message="wrong",
            )
        ]
    )
    (line,) = github(report)
    command, _, _ = line.partition("::")[2].partition("::")
    properties = dict(item.split("=", 1) for item in command.split(" ", 1)[1].split(","))
    assert properties["file"] == "my%2Cmodel.py"
    assert set(properties) == {"file", "line", "col", "title"}


def test_summary_renders_skipped_functions():
    import json

    report_clean = Report(checked_files=1, checked_functions=0, skipped_functions=2)
    full_clean = render(report_clean, "full", Path("/project"), color=False)
    assert "No problems in 0 function(s) (2 skipped) across 1 file(s)" in full_clean

    report_diags = Report(
        diagnostics=[
            Diagnostic(
                path="/project/model.py",
                line=0,
                column=0,
                rule="uninstantiable",
                severity=Severity.WARNING,
                message="cannot construct",
            )
        ],
        checked_files=1,
        checked_functions=1,
        skipped_functions=1,
    )
    full_diags = render(report_diags, "full", Path("/project"), color=False)
    assert "Found 1 warning(s) in 1 function(s) (1 skipped) across 1 file(s)" in full_diags

    # When skipped_functions is 0, (0 skipped) should not be printed
    report_no_skips = Report(checked_files=1, checked_functions=3, skipped_functions=0)
    full_no_skips = render(report_no_skips, "full", Path("/project"), color=False)
    assert "No problems in 3 function(s) across 1 file(s)" in full_no_skips
    assert "skipped" not in full_no_skips

    # JSON includes skipped_functions
    data = json.loads(render(report_clean, "json", Path("/project")))
    assert data["checked_functions"] == 0
    assert data["skipped_functions"] == 2
