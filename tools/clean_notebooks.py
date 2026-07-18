"""Inject a clean header (title + description + auto-TOC) at the top of
each scPRINT-2 example notebook.

The header is delimited by HTML comment markers so this script is
idempotent: re-running it replaces the existing header instead of
stacking new ones.

Usage:
    python tools/clean_notebooks.py            # update all configured notebooks
    python tools/clean_notebooks.py path/to/nb.ipynb [more.ipynb ...]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
HEADER_BEGIN = "<!-- scprint2-nb-header:begin -->"
HEADER_END = "<!-- scprint2-nb-header:end -->"

# (path, title, description) — one row per public example notebook.
NOTEBOOKS: list[tuple[str, str, str]] = [
    (
        "notebooks/prepare_scprint2.ipynb",
        "Get started: prepare scPRINT-2",
        "One-off setup: download the ontologies into lamindb and load a "
        "scPRINT-2 checkpoint. Run this once per environment before any of "
        "the other example notebooks.",
    ),
    (
        "notebooks/scPRINT-2-repro-notebooks/unknown_species_classification.ipynb",
        "Run scPRINT-2 on a new species",
        "How to apply scPRINT-2 to an organism the model has never seen, "
        "including UMAP visualisation and extracting cell-type labels at "
        "different resolutions based on the model's prediction certainties.",
    ),
    (
        "notebooks/scPRINT-2-repro-notebooks/gene_networks.ipynb",
        "Gene regulatory network inference with scPRINT-2",
        "End-to-end GRN inference: load reference networks (human "
        "interactome, CellMap), run scPRINT-2 on a dataset, benchmark "
        "against multiple ground truths with BenGRN, and plot the scores.",
    ),
    (
        "notebooks/scPRINT-2-repro-notebooks/batch_corr_op.ipynb",
        "Cell embeddings + label projection (Open Problems)",
        "How to generate cell embeddings with scPRINT-2 and use them for "
        "the Open Problems batch-correction / label-projection task.",
    ),
    (
        "notebooks/scPRINT-2-repro-notebooks/output_embeddings.ipynb",
        "Gene output embeddings",
        "How to extract gene-level output embeddings from your scRNAseq "
        "data. Compares a CLS-pooling model with one trained without "
        "XPressor layers.",
    ),
    (
        "notebooks/scPRINT-2-repro-notebooks/generative_modelling.ipynb",
        "Counterfactual gene expression prediction",
        "Use scPRINT-2 to re-generate cell types across conditions "
        "(species, sex, ...): move embeddings between organisms / "
        "conditions, predict expression, and run downstream GSEA-style "
        "benchmarks on the predicted profiles.",
    ),
    (
        "notebooks/scPRINT-2-repro-notebooks/denoising_V3.ipynb",
        "Denoising with scPRINT-2",
        "Denoise scRNAseq counts and benchmark against MAGIC and DCA. "
        "Includes the optional isolated `.dca-env/` setup needed for DCA "
        "(TF 2.12) and the full impact-vs-depth analysis.",
    ),
    (
        "notebooks/scPRINT-2-repro-notebooks/xenium_imputation.ipynb",
        "Spatial transcriptomics: imputation on Xenium panel data",
        "Run scPRINT-2 on Xenium spatial transcriptomics: imputation of "
        "the missing genes outside the panel, denoising, label transfer, "
        "and benchmarking against Tangram.",
    ),
    (
        "notebooks/scPRINT-2-repro-notebooks/fine_tuning_cross_species_emb_mmd.ipynb",
        "Fine-tuning scPRINT-2 (cell type / batch correction / cross-species)",
        "Full fine-tuning loop using the `tasks/finetune.py` helper class: "
        "dataloaders, additional modules, what to freeze/train, MMD loss "
        "for cross-species alignment, saving the fine-tuned checkpoint, "
        "and evaluating the result.",
    ),
]


def _slugify(text: str) -> str:
    """Approximation of the slug GitHub's notebook viewer uses for headings."""
    slug = text.strip().lower()
    slug = re.sub(r"[`*_~]+", "", slug)
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug)
    return slug


def _extract_existing_toc_entries(cells: Iterable[dict]) -> list[tuple[int, str]]:
    """Return a list of (level, text) heading entries from markdown cells.

    The injected header cell itself is skipped via its delimiter markers.
    """
    entries: list[tuple[int, str]] = []
    for cell in cells:
        if cell.get("cell_type") != "markdown":
            continue
        src = "".join(cell.get("source", []))
        if HEADER_BEGIN in src:
            continue
        fence_char: str | None = None
        fence_length = 0
        for line in src.splitlines():
            if fence_char is not None:
                closing_fence = re.match(
                    rf"^ {{0,3}}({re.escape(fence_char)}{{{fence_length},}})\s*$",
                    line,
                )
                if closing_fence:
                    fence_char = None
                    fence_length = 0
                continue
            opening_fence = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
            if opening_fence:
                marker = opening_fence.group(1)
                fence_char = marker[0]
                fence_length = len(marker)
                continue
            m = re.match(r"^(#{1,4})\s+(.+?)\s*#*\s*$", line)
            if not m:
                continue
            level = len(m.group(1))
            text = m.group(2).strip()
            # skip empty or anchor-only lines
            if not text:
                continue
            entries.append((level, text))
    return entries


def _build_header_cell(
    title: str, description: str, toc: list[tuple[int, str]]
) -> dict:
    lines = [
        HEADER_BEGIN,
        f"# {title}",
        "",
        description,
        "",
    ]
    if toc:
        lines.append("## Contents")
        lines.append("")
        # Start at the first heading's level so the TOC cannot begin with an
        # indented bullet when a shallower heading appears later.
        base_level = toc[0][0]
        slug_counts: dict[str, int] = {}
        for level, text in toc:
            indent = "  " * max(0, level - base_level)
            slug = _slugify(text)
            occurrence = slug_counts.get(slug, 0)
            slug_counts[slug] = occurrence + 1
            if occurrence:
                slug = f"{slug}-{occurrence}"
            lines.append(f"{indent}- [{text}](#{slug})")
        lines.append("")
    lines.append(
        "> _Run `notebooks/prepare_scprint2.ipynb` once before any other "
        "example notebook._"
    )
    lines.append("")
    lines.append(HEADER_END)
    source = [s + "\n" for s in lines[:-1]] + [lines[-1]]
    return {
        "cell_type": "markdown",
        "metadata": {"scprint2_header": True},
        "source": source,
    }


def _strip_existing_header(cells: list[dict]) -> list[dict]:
    out = []
    for cell in cells:
        if cell.get("cell_type") == "markdown":
            src = "".join(cell.get("source", []))
            if HEADER_BEGIN in src and HEADER_END in src:
                continue
            if cell.get("metadata", {}).get("scprint2_header"):
                continue
        out.append(cell)
    return out


def _has_literal_nonascii(text: str) -> bool:
    return any(ord(ch) > 127 for ch in text)


def update_notebook(path: Path, title: str, description: str) -> bool:
    """Update the header of a single notebook. Returns True if changed."""
    raw_text = path.read_text(encoding="utf-8")
    nb = json.loads(raw_text)
    cells = nb.get("cells", [])
    stripped = _strip_existing_header(cells)
    toc = _extract_existing_toc_entries(stripped)
    header = _build_header_cell(title, description, toc)
    new_cells = [header] + stripped
    if new_cells == cells:
        return False
    nb["cells"] = new_cells
    # Preserve whichever unicode-escaping convention the original file used
    # (nbformat is inconsistent across writers: nbconvert escapes non-ASCII,
    # papermill writes literal unicode). Keeping the original convention
    # avoids massive cosmetic diffs in execution output cells.
    if _has_literal_nonascii(raw_text):
        ensure_ascii = False
    else:
        ensure_ascii = True
    path.write_text(
        json.dumps(nb, indent=1, ensure_ascii=ensure_ascii) + "\n",
        encoding="utf-8",
    )
    return True


def main(argv: list[str]) -> int:
    if argv:
        targets = []
        for raw in argv:
            p = Path(raw)
            if not p.is_absolute():
                p = REPO_ROOT / p
            # find the configured title/description if available
            try:
                rel = p.relative_to(REPO_ROOT).as_posix()
            except ValueError:
                print(f"[skip] outside repository: {p}", file=sys.stderr)
                continue
            match = next(((t, d) for (q, t, d) in NOTEBOOKS if q == rel), None)
            if match is None:
                print(f"[skip] no header config for {rel}", file=sys.stderr)
                continue
            targets.append((p, *match))
    else:
        targets = [(REPO_ROOT / q, t, d) for (q, t, d) in NOTEBOOKS]

    changed = 0
    for path, title, description in targets:
        if not path.exists():
            print(f"[miss] {path} not found", file=sys.stderr)
            continue
        if update_notebook(path, title, description):
            print(f"[updt] {path.relative_to(REPO_ROOT)}")
            changed += 1
        else:
            print(f"[noop] {path.relative_to(REPO_ROOT)}")
    print(f"\n{changed}/{len(targets)} notebooks updated.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
