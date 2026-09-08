# scPRINT-2 FT MMD weight grid on DKD

All runs use `small-v2.ckpt`, seed 0, two fine-tuning epochs, the raw 64D
cell-type embedding, held-out donor `control_3`, model-context KNN cells, and
the repository OpenProblems scIB evaluator. Only the differentiable MMD weight
changes.

The batch mean below averages ASW batch, graph connectivity, iLISI, kBET, and
PCR. The biology mean averages ARI, ASW label, cell-cycle conservation, cLISI,
and NMI. The displayed combined mean is `0.4 * batch + 0.6 * biology`; it is a
within-grid summary, not an official published OpenProblems aggregate.

| MMD weight | Batch mean | Biology mean | Combined | ARI | NMI | Held-out accuracy | Macro-F1 | Weighted-F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 0.541607 | 0.859234 | 0.732183 | 0.865406 | 0.856765 | 0.956802 | 0.862397 | 0.936933 |
| 0.003 | 0.582205 | 0.850014 | 0.742890 | 0.830574 | 0.838959 | 0.956612 | 0.866163 | 0.936595 |
| 0.01 | 0.590117 | 0.872367 | 0.759467 | 0.895101 | 0.870150 | 0.937476 | 0.846628 | 0.909629 |
| 0.03 | 0.602372 | 0.873140 | 0.764833 | 0.913979 | 0.866255 | 0.938613 | 0.857304 | 0.910927 |
| 0.1 | 0.616877 | 0.868590 | 0.767905 | 0.876540 | 0.862392 | 0.921372 | 0.798896 | 0.891465 |
| 3 | 0.570189 | 0.225569 | 0.363417 | 0.049656 | 0.179655 | 0.344638 | 0.204308 | 0.343459 |

`0.003` is the conservative choice when preserving classification is required.
Weights `0.01-0.1` improve the available scIB batch/biology aggregate but cost
roughly 2-4 percentage points of held-out accuracy. The old weight `3` is far
outside the usable regime.
