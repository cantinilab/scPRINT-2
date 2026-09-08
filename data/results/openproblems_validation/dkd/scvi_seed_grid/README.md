# OpenProblems scVI seed grid on DKD

These runs use the same current data, GPU type, evaluator, and scVI recipe. The
only intended change is `scvi.settings.seed` (`0`, `1`, or `2`). The current
environment uses scvi-tools 1.5.0.post1; the published OpenProblems recipe only
specified `scvi-tools>=1.1.0` and did not record a seed.

| Metric | Seed 0 | Seed 1 | Seed 2 | Mean | Sample SD | Published OP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ARI | 0.823036 | 0.824407 | 0.741823 | 0.796422 | 0.047289 | 0.8284 |
| ASW batch | 0.902910 | 0.897712 | 0.901729 | 0.900784 | 0.002725 | 0.9166 |
| ASW label | 0.590538 | 0.593559 | 0.592712 | 0.592270 | 0.001558 | 0.5754 |
| Cell-cycle conservation | 0.567501 | 0.582232 | 0.583425 | 0.577719 | 0.008869 | 0.5349 |
| cLISI | 0.999468 | 0.999591 | 0.999418 | 0.999492 | 0.000089 | 0.9992 |
| Graph connectivity | 0.980264 | 0.981020 | 0.985988 | 0.982424 | 0.003110 | 0.9812 |
| iLISI | 0.336174 | 0.331780 | 0.332231 | 0.333395 | 0.002417 | 0.3030 |
| kBET | 0.312050 | 0.319844 | 0.318642 | 0.316845 | 0.004196 | 0.2538 |
| NMI | 0.841705 | 0.846000 | 0.819084 | 0.835596 | 0.014461 | 0.8495 |
| PCR | 0.686912 | 0.680645 | 0.681388 | 0.682982 | 0.003424 | 0.6671 |

The seed can explain most of the observed ARI/NMI discrepancy: seeds 0 and 1
nearly recover the published clustering scores, while seed 2 resembles the
lower unseeded rerun. It does not explain persistent shifts in cell-cycle,
iLISI, kBET, or PCR. Those residuals remain compatible with scVI/software
version drift and evaluator stochasticity.
