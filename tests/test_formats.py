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
