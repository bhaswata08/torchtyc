import ast

import pytest

from torchtyc.annotations import (
    AnnotationError,
    ArraySpec,
    OpaqueSpec,
    TupleSpec,
    parse_annotation,
    parse_dim_string,
)


def annotate(text: str):
    return parse_annotation(ast.parse(text, mode="eval").body)


def test_named_dims():
    spec = annotate('Float[Tensor, "batch seq d_model"]')
    assert isinstance(spec, ArraySpec)
    assert spec.dtype == "Float"
    assert spec.array_type == "Tensor"
    assert spec.named_dims == ("batch", "seq", "d_model")


def test_dotted_array_type():
    spec = annotate('Float[nn.Parameter, "out in"]')
    assert isinstance(spec, ArraySpec) and spec.array_type == "nn.Parameter"


def test_dim_kinds():
    dims = parse_dim_string("... 3 _ *batch #chan d_in+d_out")
    assert [d.kind for d in dims] == [
        "variadic",
        "fixed",
        "anonymous",
        "variadic",
        "named",
        "symbolic",
    ]
    assert dims[1].size == 3
    assert dims[3].name == "batch"
    assert dims[4].broadcastable is True
    assert dims[5].expr == "d_in+d_out"


def test_tuple_return():
    spec = annotate('tuple[Float[Tensor, "a"], Float[Tensor, "b"]]')
    assert isinstance(spec, TupleSpec) and len(spec.items) == 2


def test_non_jaxtyping_is_opaque():
    assert isinstance(annotate("int | None"), OpaqueSpec)
    assert isinstance(annotate("list[str]"), OpaqueSpec)


def test_missing_annotation_is_none():
    assert parse_annotation(None) is None


def test_bad_dim_string_raises():
    with pytest.raises(AnnotationError):
        annotate("Float[Tensor, 3]")
    with pytest.raises(AnnotationError):
        parse_dim_string("a $b")


def test_roundtrip_str():
    spec = annotate('Float[Tensor, "... d_model"]')
    assert str(spec) == 'Float[Tensor, "... d_model"]'


def test_a_leading_space_reads_the_same_as_no_leading_space():
    # jaxtyping strips surrounding whitespace, and a leading space is how a
    # single-axis annotation is written so ruff does not read it as a forward
    # reference (UP037). The two spellings have to mean one thing here too.
    spaced = annotate('Float[nn.Parameter, " d_model"]')
    plain = annotate('Float[nn.Parameter, "d_model"]')
    assert isinstance(spaced, ArraySpec) and isinstance(plain, ArraySpec)
    assert spaced.dims == plain.dims
    assert spaced.named_dims == ("d_model",)
    # The space is kept as written, so a suggestion can write it back.
    assert spaced.dim_text == " d_model"
    assert plain.dim_text == "d_model"


@pytest.mark.parametrize(
    ("text", "names"),
    [
        (" d_out d_in", ("d_out", "d_in")),
        ("  batch  seq  ", ("batch", "seq")),
        (" ...", ()),
        (" *shape", ("shape",)),
    ],
)
def test_surrounding_whitespace_never_changes_a_dim_string(text, names):
    dims = parse_dim_string(text)
    assert tuple(d.name for d in dims if d.name is not None) == names
    assert dims == parse_dim_string(text.strip())
