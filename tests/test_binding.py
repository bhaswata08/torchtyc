import re
import threading

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


def test_describe_names_a_variadic_dimension():
    binder = DimBinder()
    binder.bind_variadic("batch")
    assert binder.describe(binder.variadics["batch"][0]) == "batch[0]"


def test_factored_axis_keeps_the_variadic_name():
    binder = DimBinder()
    batch = binder.bind_variadic("batch")
    d_model = binder.bind("d_model")
    factored = binder.describe(batch[0] * d_model)
    assert set(factored.split("*")) == {"batch[0]", "d_model"}


def test_variadic_conflict_is_reported_as_dim_inconsistent():
    binder = DimBinder()
    shape = shape_for(spec("*batch d"), binder)
    with pytest.raises(BindingError) as caught:
        check_shape(spec("*batch d"), shape[1:], binder)
    assert caught.value.rule == "dim-inconsistent"


def test_rank_error_carries_the_rank_rule():
    binder = DimBinder()
    shape = shape_for(spec("a b c"), binder)
    with pytest.raises(BindingError) as caught:
        check_shape(spec("a b c"), shape[:2], binder)
    assert caught.value.rule == "rank-mismatch"


def test_swapped_dims_stay_a_shape_mismatch():
    binder = DimBinder()
    a, b = shape_for(spec("a b"), binder)
    with pytest.raises(BindingError) as caught:
        check_shape(spec("a b"), (b, a), binder)
    assert caught.value.rule == "shape-mismatch"


def describe_within(binder: DimBinder, size: int, seconds: float = 5.0) -> str:
    """describe() in a thread, so a non-terminating loop fails fast."""
    out: list[str] = []
    worker = threading.Thread(target=lambda: out.append(binder.describe(size)), daemon=True)
    worker.start()
    worker.join(timeout=seconds)
    assert not worker.is_alive(), "describe did not return"
    return out[0]


def test_describe_returns_when_a_name_is_bound_to_one():
    binder = DimBinder()
    binder.sizes.update({"one": 1, "d_model": 101})
    assert describe_within(binder, 7) == "7"


def test_a_name_bound_to_one_does_not_pad_a_factored_axis():
    binder = DimBinder()
    binder.sizes.update({"one": 1, "seq": 101, "d_model": 103})
    assert set(describe_within(binder, 101 * 103).split("*")) == {"seq", "d_model"}


def test_describe_survives_a_name_bound_to_zero():
    binder = DimBinder()
    binder.sizes.update({"empty": 0, "d_model": 101})
    assert describe_within(binder, 7) == "7"


def test_describe_survives_a_variadic_bound_to_one():
    binder = DimBinder()
    binder.variadics["batch"] = (1, 1)
    binder.sizes["d_model"] = 101
    assert describe_within(binder, 7) == "7"


def test_anonymous_variadic_renders_as_ellipsis():
    binder = DimBinder()
    shape = shape_for(spec("... d"), binder)
    rendered = binder.render_shape(shape)
    # The primes standing in for `...` are bookkeeping, not something the user
    # wrote, so they must not surface.
    assert rendered == "(..., d)"
    assert "10" not in rendered


def test_named_variadic_keeps_its_name():
    binder = DimBinder()
    shape = shape_for(spec("*batch d"), binder)
    assert "batch[0]" in binder.render_shape(shape)


def test_mismatch_message_names_both_axes_without_primes():
    binder = DimBinder()
    binder.bind("in_features")
    binder.bind("out_features")
    with pytest.raises(BindingError) as caught:
        check_shape(spec("in_features"), (binder.sizes["out_features"],), binder)
    message = caught.value.message
    assert "`in_features`" in message and "`out_features`" in message
    assert str(binder.sizes["in_features"]) not in message


def test_mismatch_message_falls_back_to_a_number_when_unnamed():
    binder = DimBinder()
    binder.bind("d")
    with pytest.raises(BindingError) as caught:
        check_shape(spec("d"), (7,), binder)
    assert "7" in caught.value.message


def test_suggestion_is_offered_when_every_axis_has_a_name():
    binder = DimBinder()
    binder.bind("in_features")
    out = binder.bind("out_features")
    with pytest.raises(BindingError) as caught:
        check_shape(spec("... in_features"), (*binder.bind_variadic(None), out), binder)
    assert caught.value.suggestion == "... out_features"


def test_no_suggestion_when_an_axis_is_a_product():
    binder = DimBinder()
    binder.sizes.update({"seq": 101, "d_model": 103})
    with pytest.raises(BindingError) as caught:
        check_shape(spec("seq d_model"), (101 * 103,), binder)
    # Suggesting `seq*d_model` would not be a valid dim string, so nothing is
    # offered rather than something wrong.
    assert caught.value.suggestion is None


def test_no_suggestion_when_an_axis_has_no_name():
    binder = DimBinder()
    binder.bind("d")
    with pytest.raises(BindingError) as caught:
        check_shape(spec("d"), (7,), binder)
    assert caught.value.suggestion is None


def test_suggest_dims_collapses_an_anonymous_run():
    binder = DimBinder(variadic_rank=3)
    shape = shape_for(spec("... d"), binder)
    assert binder.suggest_dims(shape) == "... d"


def test_anonymous_axis_never_shows_its_prime():
    binder = DimBinder()
    built = shape_for(spec("_ d"), binder)
    with pytest.raises(BindingError) as caught:
        check_shape(spec("d _"), built, binder)
    error = caught.value
    assert error.got == "(_, d)"
    assert not re.search(r"\d", error.message)
    assert not re.search(r"\d", error.got)


def test_no_suggestion_when_the_shape_holds_an_anonymous_axis():
    binder = DimBinder()
    built = shape_for(spec("_ d"), binder)
    with pytest.raises(BindingError) as caught:
        check_shape(spec("d _"), built, binder)
    assert caught.value.suggestion is None


def test_anonymous_axes_render_one_each_but_a_variadic_run_collapses():
    binder = DimBinder(variadic_rank=3)
    assert binder.render_shape(shape_for(spec("_ _ d"), binder)) == "(_, _, d)"
    assert binder.render_shape(shape_for(spec("... e"), binder)) == "(..., e)"


def test_bare_variadic_binds_the_same_axes_across_one_trace():
    binder = DimBinder()
    first = shape_for(spec("... d"), binder)
    second = shape_for(spec("... e"), binder)
    assert first[:-1] == second[:-1]


def test_a_flattened_pair_of_anonymous_axes_never_shows_a_prime():
    binder = DimBinder()
    batch = shape_for(spec("... d"), binder)[:-1]
    flattened = batch[0] * batch[1]
    assert binder.describe(flattened) == "..."
    assert str(flattened) not in binder.rename_primes(f"shapes {flattened} and 4 do not match")


def test_a_flattened_pair_of_named_axes_renames_to_its_factors():
    binder = DimBinder()
    rows, columns = shape_for(spec("s d"), binder)
    renamed = binder.rename_primes(f"size {rows * columns}")
    assert renamed in ("size s*d", "size d*s")


def test_a_flattened_axis_is_never_suggested_as_a_dim_string():
    binder = DimBinder()
    rows, columns = shape_for(spec("s d"), binder)
    assert binder.suggest_dims((rows * columns,)) is None
