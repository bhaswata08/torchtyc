from pathlib import Path

import pytest

from torchtyc.config import Config
from torchtyc.discovery import scan_source
from torchtyc.einops_rules import PatternError, parse_pattern
from torchtyc.engine import lint_scan


def rules_for(source: str) -> list[str]:
    scan = scan_source(source, "x.py")
    return [d.rule for d in lint_scan(scan, Config(root=Path(".")))]


def test_parse_pattern():
    pattern = parse_pattern("a b, b c -> a c")
    assert pattern.left == [["a", "b"], ["b", "c"]]
    assert pattern.right == ["a", "c"]


def test_pattern_without_arrow():
    with pytest.raises(PatternError):
        parse_pattern("a b c")


def test_unknown_output_axis_flagged():
    source = """
from einops import einsum
def f(x, y):
    return einsum(x, y, "ij, jk -> ik")
"""
    assert "einops-unknown-axis" in rules_for(source)


def test_kwarg_supplied_axis_is_accepted():
    source = """
from einops import repeat
def f(x):
    return repeat(x, "a -> a b", b=4)
"""
    assert "einops-unknown-axis" not in rules_for(source)


def test_einsum_operand_count_checked():
    source = """
from einops import einsum
def f(x, y):
    return einsum(x, y, "a b, b c, c d -> a d")
"""
    assert "einops-pattern" in rules_for(source)


def test_comma_in_rearrange_flagged():
    source = """
from einops import rearrange
def f(x):
    return rearrange(x, "a b, b c -> a c")
"""
    assert "einops-pattern" in rules_for(source)


def test_valid_pattern_is_quiet():
    source = """
from einops import einsum
def f(x, y):
    return einsum(x, y, "a b, b c -> a c")
"""
    assert rules_for(source) == []
