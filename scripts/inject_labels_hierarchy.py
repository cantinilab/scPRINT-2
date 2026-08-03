#!/usr/bin/env python3
"""
inject_labels_hierarchy.py  (v2 - handles dir or file gene embeddings)
Recover and inject `labels_hierarchy` + `genes` into a scPRINT-2 checkpoint.

Usage:
    python inject_labels_hierarchy.py <ckpt_path> <config_yaml> <gene_emb_path>
    gene_emb_path: parquet file OR directory of per-species parquets.
"""

import sys, os, argparse, yaml

parser = argparse.ArgumentParser()
parser.add_argument("ckpt_path")
parser.add_argument("config_yaml")
parser.add_argument("gene_emb_path", help="Gene embeddings parquet file or directory")
args = parser.parse_args()

ckpt_path = args.ckpt_path
config_yaml = args.config_yaml
gene_emb_path = args.gene_emb_path
out_path = ckpt_path + ".patched.ckpt"
is_dir = os.path.isdir(gene_emb_path)

print("=== inject_labels_hierarchy v2 ===")
print(f"Input checkpoint : {ckpt_path}")
print(f"Config YAML      : {config_yaml}")
print(f"Gene emb path    : {gene_emb_path} ({'directory' if is_dir else 'file'})")
print(f"Output checkpoint: {out_path}")
print()

import torch
import pandas as pd
import pyarrow.parquet as pq

print("Loading checkpoint ...")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
hp = ckpt["hyper_parameters"]

ckpt_label_decoders = hp["label_decoders"]
print("label_decoders summary:")
for clss, d in ckpt_label_decoders.items():
    print(f"  {clss}: {len(d)} entries")
print()

ckpt_encoder = {
    clss: {str_val: int_key for int_key, str_val in d.items()}
    for clss, d in ckpt_label_decoders.items()
}
n_pred = hp.get("classes", {})
print("N_predictable per class:", n_pred)

organisms = hp.get("organisms", ["NCBITaxon:9606", "NCBITaxon:10090"])
print("Organisms (ckpt order):", organisms)
print()

sd = ckpt["state_dict"]
expected_n_genes = None
for k, v in sd.items():
    if "gene_encoder" in k and "weight" in k and len(v.shape) == 2:
        expected_n_genes = v.shape[0]
        print(f"Expected n_genes (from state_dict {k}): {expected_n_genes}")
        break
print()

with open(config_yaml) as f:
    config = yaml.safe_load(f)
hierarchical_clss = config.get("data", {}).get("hierarchical_clss", [])
print("hierarchical_clss:", hierarchical_clss)
print()

print("Connecting to lamindb ...")
import lamindb as ln
import bionty as bt
from scdataloader.utils import load_genes, get_ancestry_mapping
print("  connected")
print()

# ── Recover gene set ─────────────────────────────────────────────────────────
print(f"Building gene embedding index from: {gene_emb_path}")
emb_gene_set = set()
if is_dir:
    parquet_files = [f for f in os.listdir(gene_emb_path) if f.endswith(".parquet")]
    print(f"  Reading {len(parquet_files)} parquet files (index column only) ...")
    for fname in parquet_files:
        fpath = os.path.join(gene_emb_path, fname)
        try:
            table = pq.read_table(fpath, columns=["__index_level_0__"])
            emb_gene_set.update(table["__index_level_0__"].to_pylist())
        except Exception as e:
            # fallback: read first row only to detect schema
            try:
                df_idx = pd.read_parquet(fpath)
                emb_gene_set.update(df_idx.index.tolist())
            except Exception as e2:
                print(f"  Warning: could not read {fname}: {e2}")
    print(f"  Total genes in embedding dir: {len(emb_gene_set)}")
else:
    try:
        table = pq.read_table(gene_emb_path, columns=["__index_level_0__"])
        emb_gene_set = set(table["__index_level_0__"].to_pylist())
    except Exception:
        df = pd.read_parquet(gene_emb_path)
        emb_gene_set = set(df.index.tolist())
    print(f"  Gene embedding size: {len(emb_gene_set)}")

print("Loading organism gene tables from lamindb ...")
genedf = load_genes(organisms)
print(f"  genedf shape: {genedf.shape}")

genes_dict = {}
total_genes = 0
for org in organisms:
    org_genes = genedf.index[genedf.organism == org].tolist()
    filtered = [g for g in org_genes if g in emb_gene_set]
    genes_dict[org] = filtered
    total_genes += len(filtered)
    print(f"  {org}: {len(filtered)} genes in intersection")

print(f"Total genes: {total_genes}")
if expected_n_genes is not None:
    if total_genes == expected_n_genes:
        print(f"  Gene count MATCHES state_dict: {total_genes} == {expected_n_genes} CHECK")
    else:
        print(f"  WARNING: Gene count mismatch! {total_genes} != {expected_n_genes}")
print()

# ── Build labels_hierarchy ───────────────────────────────────────────────────
bionty_query_map = {
    "cell_type_ontology_term_id": lambda: (
        bt.CellType.filter()
        .to_dataframe(limit=None, include=["parents__ontology_id", "ontology_id"])
        .set_index("ontology_id")
    ),
    "tissue_ontology_term_id": lambda: (
        bt.Tissue.filter()
        .to_dataframe(limit=None, include=["parents__ontology_id", "ontology_id"])
        .set_index("ontology_id")
    ),
    "disease_ontology_term_id": lambda: (
        bt.Disease.filter()
        .to_dataframe(limit=None, include=["parents__ontology_id", "ontology_id"])
        .set_index("ontology_id")
    ),
    "assay_ontology_term_id": lambda: (
        bt.ExperimentalFactor.filter()
        .to_dataframe(limit=None, include=["parents__ontology_id", "ontology_id"])
        .set_index("ontology_id")
    ),
    "self_reported_ethnicity_ontology_term_id": lambda: (
        bt.Ethnicity.filter()
        .to_dataframe(limit=None, include=["parents__ontology_id", "ontology_id"])
        .set_index("ontology_id")
    ),
}

labels_hierarchy = {}
for clss in hierarchical_clss:
    if clss not in ckpt_label_decoders or clss not in bionty_query_map:
        print(f"  SKIP {clss}")
        continue
    print(f"Processing {clss} ...")
    n_p = n_pred.get(clss)
    if n_p is None:
        leaf_cats = {s for k, s in ckpt_label_decoders[clss].items() if k >= 0}
    else:
        leaf_cats = {s for k, s in ckpt_label_decoders[clss].items() if 0 <= k < n_p}
    print(f"  Leaf categories: {len(leaf_cats)}")

    parentdf = bionty_query_map[clss]()
    groupings, _, _ = get_ancestry_mapping(leaf_cats, parentdf)
    groupings.pop(None, None)

    enc = ckpt_encoder[clss]
    int_groupings = {}
    skipped_p = 0
    for parent_str, children in groupings.items():
        if parent_str not in enc:
            skipped_p += 1
            continue
        child_ints = [enc[cs] for cs in children if cs in enc]
        if child_ints:
            int_groupings[enc[parent_str]] = child_ints

    print(f"  Int-based groupings: {len(int_groupings)} parent nodes (skipped_p={skipped_p})")
    if n_p:
        bad_p = [k for k in int_groupings if 0 <= k < n_p]
        bad_c = [v for vl in int_groupings.values() for v in vl if v >= n_p]
        if bad_p or bad_c:
            print(f"  WARNING: bad_parents={bad_p[:5]}, bad_children={bad_c[:5]}")
        else:
            print(f"  Sanity check PASSED: all parents >= {n_p}, all children < {n_p}")
    labels_hierarchy[clss] = int_groupings
    print()

print("=== labels_hierarchy summary ===")
for clss, h in labels_hierarchy.items():
    print(f"  {clss}: {len(h)} parent nodes")
print()

print("Injecting labels_hierarchy and genes ...")
hp["labels_hierarchy"] = labels_hierarchy
hp["genes"] = genes_dict
print(f"  Injected labels_hierarchy ({len(labels_hierarchy)} classes)")
print(f"  Injected genes ({total_genes} genes across {len(genes_dict)} organisms)")

print(f"Saving to: {out_path}")
torch.save(ckpt, out_path)
print("Done!")

orig_size = os.path.getsize(ckpt_path)
new_size = os.path.getsize(out_path)
ratio = new_size / orig_size
print(f"\nOriginal: {orig_size/1e6:.1f} MB  Patched: {new_size/1e6:.1f} MB  Ratio: {ratio:.3f}")
if not (0.95 <= ratio <= 1.10):
    print("WARNING: Size ratio out of expected range!")
else:
    print("Size check PASSED.")
