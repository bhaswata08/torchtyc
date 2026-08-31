# torchtyc

Static array shape checking for PyTorch, powered by meta tensors.

torchtyc reads [jaxtyping](https://docs.kidger.site/jaxtyping/) annotations and
verifies the shapes your code actually produces, before you run it on a GPU. It
is the PyTorch counterpart to [jaxtyc](https://github.com/BeeGass/jaxtyc), which
does the same thing for JAX with `jax.eval_shape`.

```python
# model.py
import torch
from einops import einsum
from jaxtyping import Float
from torch import Tensor, nn


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.empty((out_features, in_features)))

    def forward(self, x: Float[Tensor, "... in_features"]) -> Float[Tensor, "... in_features"]:
        return einsum(x, self.W, "... in_features, out_features in_features -> ... out_features")
```

```
$ torchtyc check model.py
model.py:12:63: error[shape-mismatch]
  in the return of `Linear.forward`: annotated `in_features`, but the traced dimension is `out_features`
    Expected: (..., in_features)
    Got:      (..., out_features)
    12 | def forward(self, x: Float[Tensor, "... in_features"]) -> Float[Tensor, "... in_features"]:
  try:  Float[Tensor, "... out_features"]
  hint: this dimension is `out_features`, so the annotation likely names the wrong axis

Found 1 error(s) in 1 function(s) across 1 file(s)
```

Pyright and mypy cannot do this. To them `Float[Tensor, "... in_features"]` is
just `Tensor`, and the dim string is an opaque literal.

## How it works

torchtyc constructs each annotated function's arguments on
`torch.device("meta")`, calls the function, and compares the shape that comes
back against the annotation. Meta tensors carry shape, dtype, and stride but own
no storage, so a whole model runs for the cost of a dictionary lookup per
operator, with no allocation and no arithmetic.

Every dimension name is bound to a distinct prime number starting at 101. This
buys two things:

**Mistakes cannot hide.** If `d_in` and `d_out` were both bound to 64, a
transposed weight matrix would sail through. Distinct primes make the two
impossible to confuse.

**Products stay readable.** A traced dimension that is the product of two
bound primes factors back into the names that produced it, which tells you the
function flattened two axes together. The primes themselves never reach the
message:

```python
# flat.py
def flatten(x: Float[Tensor, "batch seq d_model"]) -> Float[Tensor, "batch seq d_model"]:
    return x.reshape(x.shape[0], -1)
```

```
flat.py:6:55: error[rank-mismatch]
  in the return of `flatten`: expected 3 dimensions, traced 2
    Expected: (batch, seq, d_model)
    Got:      (batch, d_model*seq)
    6 | def flatten(x: Float[Tensor, "batch seq d_model"]) -> Float[Tensor, "batch seq d_model"]:
  hint: d_model*seq looks like two annotated axes flattened into one
```

An axis you did not name renders as `...`, and a raw number appears only when
an axis has no name at all.

Because it runs your function rather than reasoning about it symbolically,
torchtyc also catches anything that raises on the way: a bad `einsum`, a
`matmul` between incompatible operands, an `nn.Module` that cannot be built. The
diagnostic anchors to the deepest frame inside your own file, not to a line in
torch.

## Install

```bash
uv add --dev torchtyc      # or: pip install torchtyc
```

Extras: `torchtyc[lsp]` for the language server, `[watch]` for watch mode,
`[einops]` for einops-aware hints, `[all]` for everything.

torchtyc must run under the same interpreter as your project, since it imports
your code. By default it finds `.venv/bin/python` next to your `pyproject.toml`.
Override with `--python` or `[tool.torchtyc] python = "..."`.

## Commands

```
torchtyc check <paths>...        Shape-check files or directories
torchtyc trace <file.py::func>   Show the shapes flowing through one function
torchtyc watch <paths>...        Re-check on change
torchtyc lsp                     Language server on stdio
torchtyc rules                   List the diagnostic rules
```

`check` takes `--format full` (default), `concise`, `json`, or `github`.

```
$ torchtyc trace model.py::Linear.forward
Linear.forward
  x       : (..., in_features)
  return -> float32[(..., out_features)]

dimension names are bound to distinct primes starting at 101
```

## Constructing modules

To check a method, torchtyc needs an instance, and to build an instance it needs
constructor arguments. It matches integer parameters of `__init__` against the
dimension names in the method's annotations:

```python
def __init__(self, in_features: int, out_features: int) -> None: ...
def forward(self, x: Float[Tensor, "... in_features"]) -> ...
```

`in_features` is a dimension name, so it receives that dimension's prime.
Parameters with defaults are left alone. A parameter that is neither a
dimension, a known type, nor defaulted produces an `unresolved-arg` warning and
the function is skipped rather than guessed at.

Modules are built inside `torch.device("meta")`, and the initialisers in
`torch.nn.init` are neutralised for the duration, since initial values cannot
affect a shape.

## Annotated attributes

Python does not check variable annotations at runtime, and neither does
jaxtyping, which only reads function signatures. Since torchtyc has a
constructed instance in hand anyway, it checks them too:

```python
# attr.py
class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.W: Float[nn.Parameter, "d_out d_in"] = nn.Parameter(torch.empty((d_in, d_out)))
```

```
attr.py:9:17: error[attribute-mismatch]
  `self.W`: annotated `d_out`, but the traced dimension is `d_in`
    Expected: (d_out, d_in)
    Got:      (d_in, d_out)
    9 | self.W: Float[nn.Parameter, "d_out d_in"] = nn.Parameter(torch.empty((d_in, d_out)))
  try:  Float[nn.Parameter, "d_in d_out"]
  hint: this dimension is `d_in`, so the annotation likely names the wrong axis
```

## Suppressing

```python
y = x.reshape(-1)  # torchtyc: ignore
y = x.reshape(-1)  # torchtyc: ignore[rank-mismatch]
```

A scoped ignore that never matches is itself reported, so suppressions do not
rot.

## Configuration

```toml
[tool.torchtyc]
python = ".venv/bin/python"   # interpreter that imports your code
severity = "warning"          # drop anything below this level
ignore = ["unused-dim"]
exclude = [".venv", "build", "experiments"]
variadic-rank = 2             # how many axes `...` stands for
einops = true
timeout = 60.0
```

## Editors

Any LSP client works. Neovim, without a plugin:

```lua
vim.lsp.config.torchtyc = {
  cmd = { "torchtyc", "lsp" },
  filetypes = { "python" },
  root_markers = { "pyproject.toml", ".git" },
}
vim.lsp.enable("torchtyc")
```

The server publishes lint diagnostics immediately on every change, and traces
after the buffer has been quiet for 0.7s, on open, and on save. It never imports
your code on a keystroke.

Hover over a function to see the traced shapes. Inlay hints show the traced
return next to each signature. Code actions offer to silence a rule or to adopt
the shape that was actually traced.

## CI

```yaml
- run: uv run torchtyc check src/ --format github
```

`--format github` emits workflow commands, so each finding becomes an inline
annotation on the pull request diff. `check` exits 1 when there are errors, and
2 when torchtyc itself could not do the job: the worker failed, or the paths
matched no python file.

## Rules

| Rule | Level | Meaning |
| --- | --- | --- |
| `shape-mismatch` | error | a traced shape disagrees with its annotation |
| `rank-mismatch` | error | a traced value has a different number of dimensions |
| `dtype-mismatch` | error | a traced dtype is outside the annotated dtype set |
| `dim-inconsistent` | error | one dimension name is bound to two different sizes |
| `attribute-mismatch` | error | an annotated attribute on self holds a different shape |
| `not-a-tensor` | error | an annotated tensor position received a non-tensor |
| `tuple-arity` | error | a tuple return has a different length than annotated |
| `einops-pattern` | error | an einops pattern disagrees with the tensors given to it |
| `trace-error` | error | the function raised while being traced |
| `import-error` | error | the module could not be imported |
| `device-mismatch` | warning | a traced value left the meta device |
| `einops-unknown-axis` | warning | an einops axis matches no input axis or keyword |
| `uninstantiable` | warning | a module's `__init__` could not be called automatically |
| `unresolved-arg` | warning | a parameter has no annotation and no default |
| `unsupported-annotation` | warning | an annotation could not be parsed |
| `local-definition` | info | a target inside a function body cannot be reached after import |
| `anonymous-return` | info | arguments are annotated but the return is not |
| `missing-annotation` | info | a public function has no jaxtyping annotation |
| `unused-dim` | info | a dimension name is used once, so it constrains nothing |
| `suppression-unused` | info | an ignore comment matched no diagnostic |

## Limits

torchtyc imports your module, so module-level side effects run. Keep training
loops behind `if __name__ == "__main__":`.

Classes and functions nested inside other classes, or written under a
module-level `if` or `try`, are checked like any other. One inside a *function
body* is not: after import there is no name to reach it by, so it reports
`local-definition` rather than passing silently. A `if TYPE_CHECKING:` block is
skipped without a diagnostic, because nothing in it exists at runtime.

It runs one concrete trace, not a proof. A function whose control flow depends
on tensor *values* rather than shapes takes whichever branch the primes send it
down. `...` stands for a fixed number of axes, two by default, so code that
behaves differently at other ranks needs `variadic-rank` or a second annotated
wrapper. Every bare `...` in one signature also stands for the *same* axes,
which is narrower than jaxtyping: a loss taking two `"... d"` arguments traces
with one batch shape, because that is what the annotation almost always means.

A dimension is a prime, and a prime does not divide. Code that splits an axis
- `head_dim = d_model // n_heads`, then `view(b, s, n_heads, head_dim)` - gets a
quotient that does not multiply back, so correct multi-head attention is
reported as a `trace-error`. A default on the divided parameter does not help,
because a name the annotations use is bound to its prime whatever it defaults
to. Silence the line with `# torchtyc: ignore[trace-error]` until dimensions
carry a divisible factor.

Runtime checking with `jaxtyping` and `beartype` remains worth having. torchtyc
tells you the shapes are consistent for the sizes it chose; beartype tells you
they were right for the batch you actually ran.

## Licence

MIT.
