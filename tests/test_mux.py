from torchtyc.mux import Mux, Pending, _deep_merge, _hover_text


def merge(method: str, results: list) -> object:
    pending = Pending(client_id=1, method=method, expected=len(results), results=results)
    return Mux([]).merge(pending)


def test_initialize_merges_capabilities():
    merged = merge(
        "initialize",
        [
            {"capabilities": {"hoverProvider": True, "workspace": {"a": 1}}},
            {"capabilities": {"codeLensProvider": {}, "workspace": {"b": 2}}},
        ],
    )
    caps = merged["capabilities"]
    assert caps["hoverProvider"] is True
    assert "codeLensProvider" in caps
    assert caps["workspace"] == {"a": 1, "b": 2}


def test_list_results_concatenate():
    assert merge("textDocument/codeAction", [[1, 2], [3]]) == [1, 2, 3]


def test_none_results_are_dropped():
    assert merge("textDocument/codeLens", [None, [1]]) == [1]
    assert merge("textDocument/definition", [None, None]) is None


def test_hovers_stack():
    merged = merge(
        "textDocument/hover",
        [
            {"contents": {"kind": "markdown", "value": "from pyright"}},
            {"contents": {"kind": "markdown", "value": "from torchtyc"}},
        ],
    )
    assert "from pyright" in merged["contents"]["value"]
    assert "from torchtyc" in merged["contents"]["value"]


def test_other_methods_take_the_first_answer():
    assert merge("textDocument/rename", [{"changes": {}}, {"changes": {"x": 1}}]) == {"changes": {}}


def test_ids_are_unique_per_server():
    mux = Mux([])
    assert mux.next_id() != mux.next_id()


def test_hover_text_forms():
    assert _hover_text("plain") == "plain"
    assert _hover_text({"value": "v"}) == "v"
    assert _hover_text([{"value": "a"}, "b"]) == "a\n\nb"
    assert _hover_text(None) == ""


def test_deep_merge_keeps_first_scalar():
    into = {"a": 1}
    _deep_merge(into, {"a": 2, "b": 3})
    assert into == {"a": 1, "b": 3}


def test_deep_merge_unions_lists():
    into = {"a": [1, 2]}
    _deep_merge(into, {"a": [2, 3]})
    assert into["a"] == [1, 2, 3]
