import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "tools" / "clean_notebooks.py"
SPEC = importlib.util.spec_from_file_location("clean_notebooks", SCRIPT)
assert SPEC is not None
clean_notebooks = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(clean_notebooks)


def _header_source(toc):
    cell = clean_notebooks._build_header_cell("Title", "Description", toc)
    return "".join(cell["source"])


def test_duplicate_headings_get_github_style_suffixes():
    source = _header_source([(1, "Results"), (2, "Details"), (1, "Results")])

    assert "[Results](#results)" in source
    assert "[Results](#results-1)" in source


def test_toc_first_entry_is_never_indented_when_later_heading_is_shallower():
    source = _header_source([(3, "Setup"), (1, "Results")])
    toc_lines = [line for line in source.splitlines() if "](#" in line]

    assert toc_lines == ["- [Setup](#setup)", "- [Results](#results)"]


def test_extract_toc_ignores_headings_inside_fenced_code_blocks():
    cells = [
        {
            "cell_type": "markdown",
            "source": [
                "# Visible heading\n",
                "```python\n",
                "# Python comment, not a heading\n",
                "```\n",
                "~~~text\n",
                "## Also not a heading\n",
                "~~~~\n",
                "## Visible subheading\n",
            ],
        }
    ]

    assert clean_notebooks._extract_existing_toc_entries(cells) == [
        (1, "Visible heading"),
        (2, "Visible subheading"),
    ]


def test_update_notebook_reads_and_writes_utf8(tmp_path, monkeypatch):
    path = tmp_path / "notebook.ipynb"
    path.write_text(json.dumps({"cells": []}), encoding="utf-8")
    calls = []
    original_read_text = Path.read_text
    original_write_text = Path.write_text

    def read_text(self, *args, **kwargs):
        calls.append(("read", kwargs.get("encoding")))
        return original_read_text(self, *args, **kwargs)

    def write_text(self, data, *args, **kwargs):
        calls.append(("write", kwargs.get("encoding")))
        return original_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(Path, "write_text", write_text)

    clean_notebooks.update_notebook(path, "Title", "Description")

    assert ("read", "utf-8") in calls
    assert ("write", "utf-8") in calls


def test_external_cli_path_is_skipped_instead_of_raising(tmp_path, capsys):
    external = tmp_path / "external.ipynb"

    assert clean_notebooks.main([str(external)]) == 0
    assert "[skip]" in capsys.readouterr().err
