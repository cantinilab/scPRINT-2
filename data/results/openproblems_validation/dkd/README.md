# DKD OpenProblems `no_integration` validation

This validation reconstructs the OpenProblems batch-integration reference from
the public common dataset's raw `counts` layer. It does not use the published
task solution as the source of normalized expression.

## Inputs and procedure

- Common dataset: `s3://openproblems-data/resources/datasets/cellxgene_census/dkd/log_cp10k/dataset.h5ad`
- Cells × genes: 39,176 × 27,980
- Normalization: `normalize_total(target_sum=10000)` followed by `log1p`
- Task preprocessing: 2,000 batch-aware HVGs, 50 PCs, 30-neighbour dataset graph
- `no_integration`: the reconstructed task PCA used as `X_emb`
- Metric graph: 15 neighbours; LISI uses 90 neighbours
- Jean Zay job: `821130` (H100), completed in 7 minutes 17 seconds

The reconstructed normalized matrix differs from the normalized layer shipped
in the public common dataset by at most `9.5367431640625e-07`. This confirms the
normalization pipeline to floating-point precision.

## Result

The following values match the published values after rounding to four decimal
places: ASW batch, ASW label, cell-cycle conservation, cLISI and PCR. The three
expected missing metrics also remain missing.

The graph- and clustering-sensitive values are close but not bit-identical:

- graph connectivity: 0.968334 vs 0.9701
- iLISI: 0.076802 vs 0.0754
- ARI: 0.580009 vs 0.5999
- NMI: 0.766383 vs 0.7735
- kBET: 0.148140 vs 0.1529

kBET is stochastic in this implementation: an immediately preceding run on the
same reconstructed reference returned 0.151715. The remaining differences are
consistent with graph construction and Leiden-version sensitivity. In contrast,
the distance-based silhouette scores match to better than `3.3e-05`, supporting
that the reconstructed PCA representation itself is equivalent for evaluation.

See `no_integration_comparison.csv` for the complete metric-by-metric table.

## Numerical environment

- Python 3.12.11
- anndata 0.11.4
- igraph 0.11.9
- leidenalg 0.10.2
- numpy 1.26.4
- pandas 2.3.1
- pynndescent 0.5.13
- scanpy 1.11.4
- scib 1.1.7
- scib-metrics 0.5.6
- scikit-learn 1.6.0
- scipy 1.14.1
- umap-learn 0.5.9.post2
- R 4.4.1
- kBET 0.99.6
