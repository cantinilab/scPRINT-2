# scPRINT-2 FT MMD ablation on DKD

Both runs use `small-v2.ckpt`, seed 0, two fine-tuning epochs, the raw 64D
cell-type embedding, the same held-out donor (`control_3`), model-context KNN
cells, and the repository OpenProblems scIB evaluator. The only intended
difference is whether the differentiable donor-alignment MMD is enabled.

The historical implementation converted each MMD tensor to a Python float with
`.item()` before adding it to the loss, so it contributed no gradient. The
`off` condition therefore represents the effective optimization used to produce
the existing scPRINT-2 FT benchmark scores.

| Metric | MMD fixed | MMD off | Delta |
| --- | ---: | ---: | ---: |
| ARI | 0.049656 | 0.865406 | -0.815749 |
| ASW batch | 0.606666 | 0.870223 | -0.263557 |
| ASW label | 0.059533 | 0.736217 | -0.676683 |
| Cell-cycle conservation | 0.105912 | 0.837781 | -0.731870 |
| cLISI | 0.733087 | 1.000000 | -0.266913 |
| Graph connectivity | 0.410332 | 0.966048 | -0.555716 |
| iLISI | 0.559137 | 0.244867 | +0.314270 |
| kBET | 0.352303 | 0.260700 | +0.091603 |
| NMI | 0.179655 | 0.856765 | -0.677111 |
| PCR | 0.922507 | 0.366196 | +0.556310 |

Held-out direct cell-type accuracy is 0.344638 with the differentiable MMD and
0.956802 without it. The current differentiable formulation therefore does not
provide a usable batch/biology trade-off without redesign or substantial
retuning.
