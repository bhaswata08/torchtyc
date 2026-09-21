"""The AST pass: find what is worth tracing, without importing anything.

This runs in the editor's process on every keystroke-ish event, so it stays
cheap and never executes user code. It answers three questions:

  * which functions carry jaxtyping annotations, and where exactly are they
  * how would you construct the class that owns a method
  * which lines carry `# torchtyc: ignore` comments
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .annotations import AnnotationError, ArraySpec, Spec, TupleSpec, parse_annotation

_IGNORE = re.compile(r"#\s*torchtyc:\s*ignore(?:\[([^\]]*)\])?")


def _byte_to_codepoint(line: str, byte_offset: int) -> int:
    """Convert a 0-based UTF-8 byte offset in `line` to a 0-based code point index."""
    if byte_offset <= 0:
        return 0
    raw = line.encode("utf-8")
    if byte_offset >= len(raw):
        return len(line)
    return len(raw[:byte_offset].decode("utf-8", errors="replace"))


@dataclass
class Position:
    line: int  # 0-based, LSP convention
    column: int
    end_line: int
    end_column: int

    @classmethod
    def of(cls, node: ast.AST, lines: Sequence[str] | str | None = None) -> Position:
        line = node.lineno - 1
        end_line = (node.end_lineno or node.lineno) - 1
        col = node.col_offset
        end_col = node.end_col_offset if node.end_col_offset is not None else col

        if lines is not None:
            if isinstance(lines, str):
                lines = lines.splitlines(keepends=True)
            if 0 <= line < len(lines):
                col = _byte_to_codepoint(lines[line], col)
            if 0 <= end_line < len(lines):
                end_col = _byte_to_codepoint(lines[end_line], end_col)

        return cls(
            line=line,
            column=col,
            end_line=end_line,
            end_column=end_col,
        )


@dataclass
class Param:
    name: str
    spec: Spec | None
    position: Position
    has_default: bool
    annotation_error: str | None = None
    # The declared type when it is a plain builtin, used to synthesise a value
    # for a non-tensor parameter.
    plain_type: str | None = None
    # Declared before `/`, so it can only be passed positionally.
    positional_only: bool = False
    # The written default evaluated without running any code, so a retry can
    # prefer it. None when there is no default or it is not a literal.
    default: Any = None


@dataclass
class Attribute:
    """An annotated assignment to `self`, as in `self.W: Float[...] = ...`.

    Python never checks these at runtime and neither does jaxtyping, which only
    looks at function signatures. Since torchtyc has a constructed instance in
    hand anyway, checking them is nearly free and covers the place where a
    weight matrix gets its axes transposed.
    """

    name: str
    spec: Spec
    position: Position


@dataclass
class InitDef:
    """One `__init__` written in a class body.

    A class can write more than one, in different branches of a guard, and only
    the branch the import ran produced the constructor that will be called. The
    scanner cannot know which that was, so it keeps them all and the tracer
    picks by the lines the live `__init__` actually came from.
    """

    params: list[Param] = field(default_factory=list)
    attributes: list[Attribute] = field(default_factory=list)
    def_line: int = 0
    end_line: int = 0
    conditional: bool = False
    position: Position | None = None


@dataclass
class ClassInfo:
    name: str
    position: Position
    # Dotted path from the module to this class, as Python's own `__qualname__`
    # spells it: `Outer.Inner`, or `factory.<locals>.Inner` inside a function.
    qualname: str
    # The lines the whole `class` statement covers, decorators included. Used
    # after import to tell this definition from another one of the same name.
    def_line: int = 0
    end_line: int = 0
    # Reached through an `if`, `try` or loop, so the import may never have run
    # it. Such a definition is only traced once it proves it is the live one.
    conditional: bool = False
    # Every `__init__` written in the body, in source order.
    inits: list[InitDef] = field(default_factory=list)
    bases: list[str] = field(default_factory=list)
    # Dimension names annotated by the class's own methods, filled in as the
    # scanner visits them.
    method_dim_names: set[str] = field(default_factory=set)

    @property
    def all_attributes(self) -> list[Attribute]:
        """Annotated attributes across every constructor written in the body.

        Which of them are reported is the tracer's call, once the import has
        said which `__init__` ran. Whether the class is worth constructing at
        all is decided before that, and deciding it from one guessed
        constructor would leave a guarded one silently unchecked.
        """
        return [attribute for init in self.inits for attribute in init.attributes]

    @property
    def all_init_params(self) -> list[Param]:
        """Constructor parameters across every `__init__` written in the body."""
        return [param for init in self.inits for param in init.params]

    @property
    def dim_names(self) -> set[str]:
        """Every dimension name the class writes, across attributes and methods.

        Constructing an instance is a class-wide act: `__init__` takes the axes
        that `forward` and `self.W` both talk about, so one method's annotation
        is what names a constructor parameter for every other. Scoping this per
        method would let the same class build under one method and not another.
        """
        names: set[str] = set(self.method_dim_names)
        for attribute in self.all_attributes:
            for array in iter_arrays(attribute.spec):
                names.update(array.named_dims)
        return names

    @property
    def literal_dims(self) -> set[int]:
        literals: set[int] = set()
        specs = [p.spec for p in self.all_init_params] + [a.spec for a in self.all_attributes]
        for spec in specs:
            for array in iter_arrays(spec):
                literals.update(array.literal_dims)
        return literals


@dataclass
class Target:
    """One function or method torchtyc will try to trace."""

    qualname: str
    name: str
    position: Position
    params: list[Param]
    returns: Spec | None
    returns_position: Position | None
    decorators: list[str]
    # First line of the whole `def` statement, decorators included. Used after
    # import to tell this definition from another one of the same name.
    def_line: int = 0
    # Last line of the function body, so a cursor below it belongs to no target.
    end_line: int = 0
    # Reached through an `if`, `try` or loop, so the import may never have run
    # it. Such a definition is only traced once it proves it is the live one.
    conditional: bool = False
    # 0-based line the `def` header finishes on, which is not `position.line`
    # once a signature wraps across several lines. `position` deliberately
    # covers the function name, because that is what a diagnostic underlines,
    # so it is the wrong anchor for anything belonging after the signature.
    signature_end_line: int = 0
    # Dimension names annotated by the function this one is nested inside. A
    # helper written inside an annotated function shares its axes by intent even
    # though it repeats none of them, so the einops near-miss rule reads both.
    enclosing_dim_names: frozenset[str] = frozenset()
    owner: ClassInfo | None = None
    annotation_error: str | None = None
    einops_calls: list[EinopsCall] = field(default_factory=list)
    # Declared `async def`, so the tracer has to await what it returns. A
    # coroutine out of a plain `def` is a forgotten await, which is a finding.
    is_async: bool = False

    @property
    def is_method(self) -> bool:
        return self.owner is not None

    @property
    def array_params(self) -> list[Param]:
        return [p for p in self.params if isinstance(p.spec, ArraySpec)]

    @property
    def has_array_annotation(self) -> bool:
        return bool(self.array_params) or isinstance(self.returns, (ArraySpec, TupleSpec))

    @property
    def dim_names(self) -> set[str]:
        names: set[str] = set()
        specs = [p.spec for p in self.params] + [self.returns]
        for spec in specs:
            for array in iter_arrays(spec):
                names.update(array.named_dims)
        return names

    @property
    def literal_dims(self) -> set[int]:
        literals: set[int] = set()
        specs = [p.spec for p in self.params] + [self.returns]
        if self.owner is not None:
            literals.update(self.owner.literal_dims)
        for spec in specs:
            for array in iter_arrays(spec):
                literals.update(array.literal_dims)
        return literals

    @property
    def visible_dim_names(self) -> set[str]:
        """Axis names in scope here, this function's own and its enclosing one's.

        Only for reading a name a human wrote nearby. The tracer keeps to
        `dim_names`, because an enclosing axis must not decide what one of this
        function's own integer parameters is bound to.
        """
        return self.dim_names | set(self.enclosing_dim_names)


@dataclass
class EinopsCall:
    func: str  # rearrange | reduce | repeat | einsum | pack | unpack
    pattern: str | None
    position: Position
    # Positional arguments that are not the pattern string. For einsum that is
    # the number of operand tensors, which the pattern must agree with.
    tensor_args: int = 0
    # `f(*operands, pattern)` hides the real operand count from the AST.
    starred_args: bool = False
    keywords: frozenset[str] = frozenset()


@dataclass
class Suppression:
    line: int  # 0-based
    rules: frozenset[str] | None  # None means "every rule on this line"
    used: bool = False


@dataclass
class FileScan:
    path: str
    targets: list[Target]
    suppressions: list[Suppression]
    classes: list[ClassInfo] = field(default_factory=list)
    syntax_error: tuple[str, Position] | None = None


def iter_arrays(spec: Spec | None):
    """Flatten a spec into its array leaves."""
    if isinstance(spec, ArraySpec):
        yield spec
    elif isinstance(spec, TupleSpec):
        for item in spec.items:
            yield from iter_arrays(item)


def _decorator_name(node: ast.expr) -> str:
    target = node.func if isinstance(node, ast.Call) else node
    parts: list[str] = []
    while isinstance(target, ast.Attribute):
        parts.append(target.attr)
        target = target.value
    if isinstance(target, ast.Name):
        parts.append(target.id)
    return ".".join(reversed(parts))


def _base_name(node: ast.expr) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


_EINOPS_FUNCS = {"rearrange", "reduce", "repeat", "einsum", "pack", "unpack"}


@dataclass
class EinopsNames:
    """Which names in this file actually refer to einops.

    The rules only hold for einops, so a call is claimed only when its callee
    was imported from einops. `torch.einsum` shares a name with `einops.einsum`
    but takes a different pattern syntax, and flagging it would be wrong.
    """

    # `import einops as E` -> {"einops", "E"}
    modules: set[str] = field(default_factory=set)
    # `from einops import rearrange as rr` -> {"rr": "rearrange"}
    functions: dict[str, str] = field(default_factory=dict)

    def func_of(self, call: ast.Call) -> str | None:
        target = call.func
        if isinstance(target, ast.Name):
            return self.functions.get(target.id)
        if (
            isinstance(target, ast.Attribute)
            and target.attr in _EINOPS_FUNCS
            and _base_name(target.value) in self.modules
        ):
            return target.attr
        return None


def _einops_names(tree: ast.Module) -> EinopsNames:
    names = EinopsNames()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "einops":
                    names.modules.add(alias.asname or "einops")
                elif alias.name.startswith("einops.") and alias.asname is None:
                    # `import einops.layers.torch` binds the top-level `einops`.
                    names.modules.add("einops")
        elif isinstance(node, ast.ImportFrom) and node.module == "einops":
            for alias in node.names:
                if alias.name in _EINOPS_FUNCS:
                    names.functions[alias.asname or alias.name] = alias.name
    return names


def _own_nodes(node: ast.AST):
    """Every node belonging to `node` itself, stopping at a nested scope.

    A nested function or class is its own target and collects its own calls, so
    walking into one here would report everything inside it twice.
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        yield child
        yield from _own_nodes(child)


def _find_einops(
    node: ast.AST, names: EinopsNames, lines: Sequence[str] | None = None
) -> list[EinopsCall]:
    calls: list[EinopsCall] = []
    for child in _own_nodes(node):
        if not isinstance(child, ast.Call):
            continue
        name = names.func_of(child)
        if name is None:
            continue
        pattern = next(
            (
                arg.value
                for arg in child.args
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            ),
            None,
        )
        tensor_args = sum(
            1
            for arg in child.args
            if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str))
        )
        calls.append(
            EinopsCall(
                func=name,
                pattern=pattern,
                position=Position.of(child, lines),
                tensor_args=tensor_args,
                starred_args=any(isinstance(arg, ast.Starred) for arg in child.args),
                keywords=frozenset(k.arg for k in child.keywords if k.arg),
            )
        )
    return calls


def _parse_param(
    arg: ast.arg,
    has_default: bool,
    positional_only: bool = False,
    lines: Sequence[str] | None = None,
    default_node: ast.expr | None = None,
) -> Param:
    position = Position.of(arg, lines)
    spec: Spec | None = None
    error: str | None = None
    plain: str | None = None
    if arg.annotation is not None:
        position = Position.of(arg.annotation, lines)
        try:
            spec = parse_annotation(arg.annotation)
        except AnnotationError as exc:
            error = str(exc)
        plain = _plain_type(arg.annotation)
    return Param(
        name=arg.arg,
        spec=spec,
        position=position,
        has_default=has_default,
        annotation_error=error,
        plain_type=plain,
        positional_only=positional_only,
        default=_literal_value(default_node),
    )


def _literal_value(node: ast.expr | None) -> Any:
    """Evaluate a written default without running any code.

    Anything that is not a literal, a name lookup away from running, reads as
    no default: the value is only ever compared by type, never called.
    """
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except Exception:  # noqa: BLE001
        return None


def _plain_type(node: ast.expr) -> str | None:
    """Reduce `int`, `int | None`, `Optional[int]` to `int` for value synthesis."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        for side in (node.left, node.right):
            name = _plain_type(side)
            if name and name != "None":
                return name
        return None
    if isinstance(node, ast.Constant) and node.value is None:
        return "None"
    if isinstance(node, ast.Subscript):
        head = _base_name(node.value).split(".")[-1]
        if head in ("Optional",):
            return _plain_type(node.slice)
        return head
    if isinstance(node, ast.Attribute):
        return _base_name(node)
    return None


def _signature_end_line(fn: ast.FunctionDef | ast.AsyncFunctionDef, lines: list[str]) -> int:
    """The 0-based line the `def` header finishes on.

    The header ends at the colon that opens the body, so a hint anchored there
    sits after the whole signature however that signature wraps. Deriving the
    line from the return annotation or the last argument instead would miss a
    wrapped signature whose `):` sits on a line of its own. The colon is the
    first one written outside brackets: the name, the parameters and any
    annotation that holds a colon of its own are all bracketed, so nothing
    earlier can be mistaken for it.
    """
    start = fn.lineno - 1
    body_start = fn.body[0].lineno if fn.body else fn.lineno
    fragment = lines[start : min(body_start, len(lines))]
    depth = 0
    try:
        for token in tokenize.generate_tokens(iter(fragment).__next__):
            if token.type != tokenize.OP:
                continue
            if token.string in "([{":
                depth += 1
            elif token.string in ")]}":
                depth -= 1
            elif token.string == ":" and depth == 0:
                return start + token.start[0] - 1
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return start


def _attributes_of(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    lines: Sequence[str] | None = None,
) -> list[Attribute]:
    found: list[Attribute] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.AnnAssign):
            continue
        target = node.target
        if not (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        ):
            continue
        try:
            spec = parse_annotation(node.annotation)
        except AnnotationError:
            continue
        if spec is None:
            continue
        found.append(
            Attribute(name=target.attr, spec=spec, position=Position.of(node.annotation, lines))
        )
    return found


def _params_of(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    lines: Sequence[str] | None = None,
) -> list[Param]:
    args = fn.args
    positional = args.posonlyargs + args.args
    defaults_start = len(positional) - len(args.defaults)
    params = [
        _parse_param(
            arg,
            has_default=index >= defaults_start,
            positional_only=index < len(args.posonlyargs),
            lines=lines,
            default_node=(
                args.defaults[index - defaults_start] if index >= defaults_start else None
            ),
        )
        for index, arg in enumerate(positional)
    ]
    params += [
        _parse_param(arg, has_default=default is not None, lines=lines, default_node=default)
        for arg, default in zip(args.kwonlyargs, args.kw_defaults or [])
    ]
    return params


# Dunder methods worth tracing. `__call__` is where a callable module or a
# functional wrapper puts its real signature, so skipping it would leave that
# code silently unchecked. Every other dunder either has a fixed signature the
# checker cannot build values for, or is not a shape-carrying entry point.
_TRACED_DUNDERS = frozenset({"__call__"})

# Statements that hold nested bodies without opening a new naming scope, so a
# class or function under one is still defined at the enclosing scope.
_NESTING = (
    ast.If,
    ast.Try,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Match,
)


def _is_type_checking(node: ast.If) -> bool:
    """Whether an `if` guards a block that never runs at import time.

    `if TYPE_CHECKING:` bodies exist only for static checkers, so a class
    defined there is genuinely absent after import. Reporting it as unreachable
    would be noise, not a finding.
    """
    test = node.test
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return False
    return _base_name(test).split(".")[-1] == "TYPE_CHECKING"


def _def_keyword(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """The keyword the header opens with, which `col_offset` points at.

    An `async def` header starts six characters earlier than a plain one, so
    measuring the span with `def ` would underline `async def fo` instead of
    the name.
    """
    return "async def " if isinstance(fn, ast.AsyncFunctionDef) else "def "


def _def_line(node: ast.stmt) -> int:
    """The 0-based first line of a `def` or `class`, decorators included."""
    lines = [node.lineno]
    lines += [d.lineno for d in getattr(node, "decorator_list", [])]
    return min(lines) - 1


def _statements(body: list[ast.stmt], conditional: bool = False):
    """Every statement in a body, with whether a branch guards it.

    A class under a module-level `if` or `try` is a module-level class, so the
    scanner has to see it. It may equally never run, so each statement carries
    whether it was reached through a branch; the tracer uses that to tell a
    definition the import produced from one it did not. Function and class
    bodies are left alone here, because they open a scope the caller handles.
    """
    for node in body:
        if isinstance(node, _NESTING):
            if isinstance(node, ast.If) and _is_type_checking(node):
                yield from _statements(node.orelse, conditional)
                continue
            for nested in _nested_bodies(node):
                yield from _statements(nested, True)
        else:
            yield node, conditional


def _nested_bodies(node: ast.stmt):
    for name in ("body", "orelse", "finalbody"):
        block = getattr(node, name, None)
        if block:
            yield block
    for handler in getattr(node, "handlers", []):
        if handler.body:
            yield handler.body
    for case in getattr(node, "cases", []):
        if case.body:
            yield case.body


def scan_source(source: str, path: str) -> FileScan:
    """Parse one file and collect targets and suppressions."""
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        line = max((exc.lineno or 1) - 1, 0)
        column = max((exc.offset or 1) - 1, 0)
        return FileScan(
            path=path,
            targets=[],
            suppressions=[],
            syntax_error=(exc.msg, Position(line, column, line, column + 1)),
        )

    lines = source.splitlines(keepends=True)
    targets: list[Target] = []
    classes: list[ClassInfo] = []
    einops_names = _einops_names(tree)

    def visit_function(
        fn: ast.FunctionDef | ast.AsyncFunctionDef,
        owner: ClassInfo | None,
        prefix: str,
        conditional: bool,
        enclosing: frozenset[str],
    ) -> None:
        qualname = f"{prefix}.{fn.name}" if prefix else fn.name
        inner = enclosing
        if owner is None or not fn.name.startswith("__") or fn.name in _TRACED_DUNDERS:
            target = add_target(fn, owner, qualname, conditional, enclosing)
            inner = frozenset(target.visible_dim_names)
        # A function body is a new scope, and anything defined in it is a local
        # of that function, exactly as `__qualname__` records it.
        visit_body(fn.body, None, f"{qualname}.<locals>", conditional, inner)

    def add_target(
        fn: ast.FunctionDef | ast.AsyncFunctionDef,
        owner: ClassInfo | None,
        qualname: str,
        conditional: bool,
        enclosing: frozenset[str],
    ) -> Target:
        returns: Spec | None = None
        error: str | None = None
        try:
            returns = parse_annotation(fn.returns)
        except AnnotationError as exc:
            error = str(exc)
        target_col = (
            _byte_to_codepoint(lines[fn.lineno - 1], fn.col_offset)
            if lines and 0 <= fn.lineno - 1 < len(lines)
            else fn.col_offset
        )
        target = Target(
            qualname=qualname,
            name=fn.name,
            position=Position(
                fn.lineno - 1,
                target_col,
                fn.lineno - 1,
                target_col + len(_def_keyword(fn)) + len(fn.name),
            ),
            params=_params_of(fn, lines),
            returns=returns,
            returns_position=Position.of(fn.returns, lines) if fn.returns else None,
            decorators=[_decorator_name(d) for d in fn.decorator_list],
            def_line=_def_line(fn),
            end_line=(fn.end_lineno or fn.lineno) - 1,
            conditional=conditional or (owner.conditional if owner else False),
            enclosing_dim_names=enclosing,
            signature_end_line=_signature_end_line(fn, lines),
            owner=owner,
            annotation_error=error,
            einops_calls=_find_einops(fn, einops_names, lines),
            is_async=isinstance(fn, ast.AsyncFunctionDef),
        )
        targets.append(target)
        if owner is not None:
            owner.method_dim_names.update(target.dim_names)
        return target

    def visit_class(
        node: ast.ClassDef, prefix: str, conditional: bool, enclosing: frozenset[str]
    ) -> None:
        qualname = f"{prefix}.{node.name}" if prefix else node.name
        class_col = (
            _byte_to_codepoint(lines[node.lineno - 1], node.col_offset)
            if lines and 0 <= node.lineno - 1 < len(lines)
            else node.col_offset
        )
        info = ClassInfo(
            name=node.name,
            qualname=qualname,
            position=Position(
                node.lineno - 1,
                class_col,
                node.lineno - 1,
                class_col + len("class ") + len(node.name),
            ),
            def_line=_def_line(node),
            end_line=(node.end_lineno or node.lineno) - 1,
            conditional=conditional,
            bases=[_base_name(base) for base in node.bases],
        )
        classes.append(info)
        # __init__ first, so every method sees the constructor parameters and
        # the annotated attributes they may refer to.
        for child, guarded in _statements(node.body):
            if (
                isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and child.name == "__init__"
            ):
                init_col = (
                    _byte_to_codepoint(lines[child.lineno - 1], child.col_offset)
                    if lines and 0 <= child.lineno - 1 < len(lines)
                    else child.col_offset
                )
                info.inits.append(
                    InitDef(
                        params=_params_of(child, lines)[1:],  # drop self
                        attributes=_attributes_of(child, lines),
                        def_line=_def_line(child),
                        end_line=(child.end_lineno or child.lineno) - 1,
                        conditional=guarded,
                        position=Position(
                            child.lineno - 1,
                            init_col,
                            child.lineno - 1,
                            init_col + len(_def_keyword(child)) + len(child.name),
                        ),
                    )
                )
        visit_body(node.body, info, qualname, conditional, enclosing)

    def visit_body(
        body: list[ast.stmt],
        owner: ClassInfo | None,
        prefix: str,
        conditional: bool,
        enclosing: frozenset[str],
    ) -> None:
        for node, guarded in _statements(body, conditional):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit_function(node, owner, prefix, guarded, enclosing)
            elif isinstance(node, ast.ClassDef):
                visit_class(node, prefix, guarded, enclosing)

    visit_body(tree.body, None, "", False, frozenset())

    return FileScan(
        path=path,
        targets=targets,
        suppressions=scan_suppressions(source),
        classes=classes,
    )


def scan_suppressions(source: str) -> list[Suppression]:
    found: list[Suppression] = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        comments = [(t.start[0], t.string) for t in tokens if t.type == tokenize.COMMENT]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        comments = [
            (index + 1, line) for index, line in enumerate(source.splitlines()) if "#" in line
        ]
    for lineno, text in comments:
        # A line carries one comment token but may hold several ignores, since
        # the editor's silence action appends to whatever is already there.
        named: set[str] = set()
        bare = False
        seen = False
        for match in _IGNORE.finditer(text):
            seen = True
            rules = match.group(1)
            if rules is None:
                bare = True
            else:
                named.update(r.strip() for r in rules.split(",") if r.strip())
        if not seen:
            continue
        found.append(
            Suppression(line=lineno - 1, rules=None if bare or not named else frozenset(named))
        )
    return found


def scan_file(path: str | Path) -> FileScan:
    text = Path(path).read_text(encoding="utf-8")
    return scan_source(text, str(path))
