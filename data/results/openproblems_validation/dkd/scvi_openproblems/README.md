# DKD scVI OpenProblems recipe validation

Fresh run on 2026-08-15 using the current OpenProblems recipe:

- counts layer, top 2,000 genes by `hvg_score`
- `SCVI.setup_anndata(..., batch_key="batch")`
- 30 latent dimensions, 128 hidden units, 2 layers
- 400 epochs, `train_size=1.0`
- `scvi-tools==1.5.0.post1`, `anndata==0.11.4`, `torch==2.8.0`

The OpenProblems dependency is specified as `scvi-tools>=1.1.0` and the recipe
does not set a random seed, so this validates the current recipe rather than
reproducing a historical embedding bit-for-bit.

| metric | fresh | published OP | delta |
|---|---:|---:|---:|
| ari | 0.748776 | 0.8284 | -0.079624 |
| asw_batch | 0.900266 | 0.9166 | -0.016334 |
| asw_label | 0.590758 | 0.5754 | +0.015358 |
| cell_cycle_conservation | 0.612722 | 0.5349 | +0.077822 |
| clisi | 0.999396 | 0.9992 | +0.000196 |
| graph_connectivity | 0.981763 | 0.9812 | +0.000563 |
| ilisi | 0.330931 | 0.3030 | +0.027931 |
| kbet | 0.325159 | 0.2538 | +0.071359 |
| nmi | 0.820326 | 0.8495 | -0.029174 |
| pcr | 0.670997 | 0.6671 | +0.003897 |

HVG overlap and isolated-label metrics are unavailable for this DKD setup.
