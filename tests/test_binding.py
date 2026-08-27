import pytest

from torchtyc.annotations import ArraySpec, parse_dim_string
from torchtyc.binding import BindingError, DimBinder, check_shape, shape_for


def spec(dims: str) -> ArraySpec:
    return ArraySpec(dtype="Float", array_type="Tensor", dims=parse_dim_string(dims), raw=dims)


def test_distinct_names_get_distinct_primes():
    binder = DimBinder()
    shape = shape_for(spec("a b c"), binder)
    assert len(set(shape)) == 3
    assert all(size >= 101 for size in shape)


def test_repeated_name_reuses_size():
    binder = DimBinder()
    assert shape_for(spec("a a"), binder) == (binder.sizes["a"],) * 2


def test_fixed_dim_is_literal():
    assert shape_for(spec("3 a"), DimBinder())[0] == 3


def test_variadic_expands_to_rank():
    binder = DimBinder(variadic_rank=3)
    assert len(shape_for(spec("... d"), binder)) == 4


def test_named_variadic_is_consistent():
    binder = DimBinder()
    first = shape_for(spec("*batch d"), binder)
    second = shape_for(spec("*batch e"), binder)
    assert first[:-1] == second[:-1]


def test_check_accepts_matching_shape():
    binder = DimBinder()
    shape = shape_for(spec("a b"), binder)
    check_shape(spec("a b"), shape, binder)


def test_check_rejects_swapped_dims():
    binder = DimBinder()
    a, b = shape_for(spec("a b"), binder)
    with pytest.raises(BindingError) as caught:
        check_shape(spec("a b"), (b, a), binder)
    assert "`a`" in caught.value.message


def test_check_rejects_wrong_rank():
    binder = DimBinder()
    shape = shape_for(spec("a b c"), binder)
    with pytest.raises(BindingError) as caught:
        check_shape(spec("a b c"), shape[:2], binder)
    assert "dimensions" in caught.value.message


def test_anonymous_dim_matches_anything():
    binder = DimBinder()
    check_shape(spec("_ _"), (7, 9), binder)


def test_broadcastable_dim_accepts_one():
    binder = DimBinder()
    binder.bind("chan")
    check_shape(spec("#chan"), (1,), binder)


def test_symbolic_dim():
    binder = DimBinder()
    binder.sizes.update({"a": 101, "b": 103})
    check_shape(spec("a+b"), (204,), binder)
    with pytest.raises(BindingError):
        check_shape(spec("a+b"), (205,), binder)


def test_describe_factors_a_flattened_axis():
    binder = DimBinder()
    binder.sizes.update({"seq": 101, "d_model": 103})
    assert binder.describe(101 * 103) in ("seq*d_model", "d_model*seq")


def test_two_variadics_rejected():
    binder = DimBinder()
    with pytest.raises(BindingError):
        shape_for(spec("... a ..."), binder)
