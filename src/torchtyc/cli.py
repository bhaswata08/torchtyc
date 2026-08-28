"""The command line: check, trace, watch, lsp, mux, rules, version."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from . import config as config_module
from .diagnostics import RULES, Severity
from .engine import check_paths, collect_files
from .formats import render, use_color


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--python", help="interpreter that imports project code (default: the project venv)"
    )
    parser.add_argument("--variadic-rank", type=int, help="how many axes `...` stands for")
    parser.add_argument("--ignore", action="append", default=[], metavar="RULE")
    parser.add_argument(
        "--severity",
        choices=[s.label for s in Severity],
        help="drop diagnostics below this level",
    )
    parser.add_argument("--timeout", type=float, help="seconds to allow the worker")


def _overrides_from(args: argparse.Namespace) -> config_module.Overrides:
    severity = getattr(args, "severity", None)
    return config_module.Overrides(
        python=getattr(args, "python", None) or None,
        variadic_rank=getattr(args, "variadic_rank", None) or None,
        ignore=frozenset(getattr(args, "ignore", None) or ()),
        severity=Severity.parse(severity) if severity else None,
        timeout=getattr(args, "timeout", None) or None,
    )


def _config_from(args: argparse.Namespace, start: str):
    return _overrides_from(args).apply(config_module.load(start))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="torchtyc",
        description="Static array shape checking for PyTorch, powered by meta tensors.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="shape-check files or directories")
    check.add_argument("paths", nargs="+")
    check.add_argument("--format", default="full", choices=["full", "concise", "json", "github"])
    _add_common(check)

    trace = sub.add_parser("trace", help="show the shapes flowing through one function")
    trace.add_argument("target", help="file.py::function or file.py::Class.method")
    _add_common(trace)

    watch = sub.add_parser("watch", help="re-check when files change")
    watch.add_argument("paths", nargs="+")
    watch.add_argument("--format", default="full", choices=["full", "concise"])
    _add_common(watch)

    lsp = sub.add_parser("lsp", help="run the language server on stdio")
    lsp.add_argument("--tcp", type=int, metavar="PORT", help="listen on a port instead of stdio")
    _add_common(lsp)

    mux = sub.add_parser("mux", help="run another language server alongside torchtyc")
    mux.add_argument(
        "--server",
        action="append",
        default=[],
        metavar="CMD",
        help="a server to multiplex, repeatable (default: basedpyright-langserver --stdio)",
    )
    _add_common(mux)

    sub.add_parser("rules", help="list the diagnostic rules")
    sub.add_parser("version", help="print the version")
    return parser


def cmd_check(args: argparse.Namespace) -> int:
    config = _config_from(args, args.paths[0])
    files = collect_files(args.paths, config)
    if not files:
        print("no python files found", file=sys.stderr)
        return 1

    report = check_paths(files, config)
    print(render(report, args.format, config.root))
    if report.worker_error:
        return 2
    return 1 if report.errors else 0


def cmd_trace(args: argparse.Namespace) -> int:
    if "::" not in args.target:
        print("trace takes file.py::function", file=sys.stderr)
        return 2
    path, qualname = args.target.split("::", 1)
    config = _config_from(args, path)

    report = check_paths([path], config, hover=True)
    traced = report.shapes_in(path)
    shapes = traced.get(qualname)
    # An empty entry means the trace ran and failed: the worker records the
    # target with no shapes. Treat it like a target that was never traced.
    if not shapes:
        available = ", ".join(sorted(name for name, found in traced.items() if found)) or "none"
        print(f"could not trace `{qualname}` (traceable here: {available})", file=sys.stderr)
        for diagnostic in report.diagnostics:
            if diagnostic.function == qualname:
                print(f"  {diagnostic.rule}: {diagnostic.message}", file=sys.stderr)
        return 1

    color = use_color()
    bold = "\033[1m" if color else ""
    dim = "\033[2m" if color else ""
    reset = "\033[0m" if color else ""

    print(f"{bold}{qualname}{reset}")
    width = max((len(n) for n in shapes), default=0)
    for name, shape in shapes.items():
        arrow = "->" if name == "return" else " :"
        label = "return" if name == "return" else name
        print(f"  {label:<{width}} {dim}{arrow}{reset} {shape}")
    print(f"\n{dim}dimension names are bound to distinct primes starting at 101{reset}")
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    try:
        from watchfiles import watch as watch_files
    except ImportError:
        print("watch needs the `watch` extra: pip install 'torchtyc[watch]'", file=sys.stderr)
        return 2

    config = _config_from(args, args.paths[0])

    def run_once() -> None:
        files = collect_files(args.paths, config)
        report = check_paths(files, config)
        print("\033[2J\033[H", end="")
        print(render(report, args.format, config.root))

    run_once()
    for _ in watch_files(*args.paths):
        run_once()
    return 0


def cmd_lsp(args: argparse.Namespace) -> int:
    try:
        from .lsp import serve
    except ImportError as exc:
        if (exc.name or "").split(".")[0] not in ("pygls", "lsprotocol"):
            raise
        print("lsp needs the `lsp` extra: pip install 'torchtyc[lsp]'", file=sys.stderr)
        return 2

    return serve(tcp_port=args.tcp, overrides=_overrides_from(args))


def cmd_mux(args: argparse.Namespace) -> int:
    from .mux import serve_mux

    servers = args.server or ["basedpyright-langserver --stdio"]
    return serve_mux(_overrides_from(args), servers)


def cmd_rules(_: argparse.Namespace) -> int:
    width = max(len(name) for name in RULES)
    for name, rule in sorted(RULES.items(), key=lambda item: (item[1].severity, item[0])):
        print(f"{name:<{width}}  {rule.severity.label:<7}  {rule.summary}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "check": cmd_check,
        "trace": cmd_trace,
        "watch": cmd_watch,
        "lsp": cmd_lsp,
        "mux": cmd_mux,
        "rules": cmd_rules,
        "version": lambda _: (print(f"torchtyc {__version__}"), 0)[1],
    }
    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
