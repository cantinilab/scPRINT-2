# Audit de l'integration cross-species oeil (4 especes)

> **REJETE — NE PAS UTILISER CES SCORES.** Ce bundle repose sur une intersection de seulement 1 768 genes et sur des cibles FT absentes du decodeur small-v2. Il est conserve uniquement pour la provenance. Le remplacement valide est `eye4_task3_corrected_20260823`.

Date de l'audit : 2026-08-23

## Conclusion

Le notebook fusionne dans le PR d'Ines ne constituait pas, en l'etat, un benchmark comparable aux analyses cross-species precedentes. Le workflow livre ici corrige les points bloquants : comptages bruts officiels, orthologie stricte, alignement sur la table de genes du checkpoint, FT scPRINT-2 differentiable sur les quatre especes, controle aleatoire, TranscriptFormer Metazoa, scIB historique 1.1.3 et UMAP issus des graphes effectivement scores.

## Problemes identifies dans le notebook fusionne

- chemins Pasteur et connexion LaminDB executes dans le notebook ;
- fonctions, boucles de production et commandes shell locales dans les cellules ;
- reconstruction de pseudo-comptages a partir d'une matrice log-normalisee ;
- emploi de `scib-metrics` moderne au lieu de l'evaluateur historique du papier ;
- MMD detache du graphe autograd par `mmd.item()` ;
- comparaison MMD limitee aux groupes 0 et 1 alors que quatre especes sont presentes ;
- `requires_grad` applique au module `cross_attn`, pas a ses parametres ;
- absence des controles Random et TranscriptFormer.

## Corrections appliquees

- les H5AD d'origine servent uniquement aux annotations expertes ; les matrices d'expression proviennent des comptages entiers GEO GSE148371, GSE148373, GSE146186 et GSE146187 ;
- chaque table d'orthologie est filtree globalement en one-to-one strict vers les symboles souris ;
- les cellules sont filtrees a au moins 200 genes non nuls, puis les 1 762 symboles souris non ambigus sont places dans l'ordre exact des 57 066 genes du checkpoint ;
- le FT utilise une MMD differentiable et equilibree sur toutes les paires uniques de groupes `species`, avec restauration du meilleur etat de validation ;
- les parametres des blocs `cross_attn` sont correctement degeles ;
- les cinq methodes sont evaluees par la meme fonction et le meme environnement legacy : PCA, Random seed 42, scPRINT-2 zero-shot, scPRINT-2 FT MMD 0.03 et TranscriptFormer Metazoa ;
- les notebooks sont des consommateurs expliques des artefacts, sans fonction locale ni cellule shell/CLI.

## Cohorte finale

| Espece | Cellules |
| --- | ---: |
| humain | 9 706 |
| macaque crabier | 3 494 |
| souris | 1 264 |
| porc | 2 160 |
| **Total** | **16 624** |

La matrice finale contient 57 066 features dans l'ordre du checkpoint et 16 978 727 comptages entiers. Les 13 labels biologiques utilises par scIB sont les annotations expertes `celltype`. `CL:7770002` (JCT) et `CL:7770003` (Beam A) sont des termes officiels actuels de la Cell Ontology, mais absents du vocabulaire du checkpoint small-v2 ; ils sont donc convertis en `unknown` uniquement dans la cible de classification interne au FT. Ils restent distincts dans l'evaluation scIB.

## Fine-tuning scPRINT-2

- objectif : reconstruction + MMD ;
- cible MMD : `species` ;
- token MMD : `cell_type_ontology_term_id` ;
- poids MMD : 0,03 ;
- 8 epochs, batch size 32, seed 42 ;
- toutes les 16 624 cellules sont incluses ;
- meilleur epoch de validation restaure : 6 ;
- meilleure loss de validation : 0,22918879584624216 ;
- graphe kNN reconstruit avant le FT et utilise dans le forward multi-cellules.

## Scores scIB 1.1.3

| Methode | Batch correction | Bio conservation | Total |
| --- | ---: | ---: | ---: |
| scPRINT-2 FT cell-type token, MMD 0.03 | 0.587641 | 0.825087 | **0.730109** |
| TranscriptFormer Metazoa | 0.450855 | **0.864107** | 0.698806 |
| scPRINT-2 zero-shot cell-type token | 0.427854 | 0.795647 | 0.648530 |
| Expression PCA50 | 0.335367 | 0.741126 | 0.578822 |
| Random seed 42 | **0.731136** | 0.468313 | 0.573442 |

Chaque ligne contient les cinq metriques de batch et les six metriques biologiques attendues. Le score total a ete reverifie numeriquement a partir des deux composantes. Les hashes des connectivites sont identiques avant et apres l'appel a scIB pour les cinq methodes.

## Provenance d'execution

- racine distante : `/lustre/fswork/projects/rech/xeg/uat95fg/data/eye_cross_species/eye4_task3_matched_20260823` ;
- preparation : Slurm 1293429, `COMPLETED` ;
- TranscriptFormer : Slurm 1293471, `COMPLETED` ;
- scPRINT-2 zero-shot et FT : Slurm 1293546, `COMPLETED` ;
- scoring scIB 1.1.3 : Slurm 1293776, `COMPLETED 0:0` en 01:02:09 ;
- UMAP dependants : Slurm 1293777, `COMPLETED 0:0` en 00:02:23.

Les artefacts locaux sont controles par SHA-256 contre les manifestes generes sur Jean Zay.

## Limites d'interpretation

Ce benchmark mesure l'integration d'un seul tissu et d'un seul jeu de quatre especes. Le score total est derive comme `0.4 * Batch correction + 0.6 * Bio conservation`. Un score plus eleve ici ne demontre pas une superiorite universelle du modele ; il depend notamment des annotations expertes, du mapping one-to-one strict, de la seed 42 et de l'objectif FT choisi.
