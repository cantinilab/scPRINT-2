# Audit corrigé de l'intégration cross-species oeil (4 espèces)

Date : 2026-08-23

## Conclusion

Le notebook du PR d'Inès ne permettait pas, dans son état fusionné, une comparaison strictement homogène avec les analyses cross-species précédentes. Le benchmark a été reconstruit à partir des comptages entiers GEO et réévalué avec les mêmes briques réutilisables du package, les cinq méthodes attendues et l'environnement historique `scib==1.1.3`.

Le bundle précédent `eye4_task3_matched_20260823` est explicitement rejeté : l'intersection de 1 768 gènes provenait de tables d'orthologie incomplètes et les cibles FT `CL:7770002` et `CL:7770003` n'étaient pas connues du décodeur small-v2. Ses embeddings, scores et UMAPs ne doivent pas être utilisés.

## Problèmes constatés dans l'analyse initiale

- chemins Pasteur et connexion LaminDB exécutés directement dans le notebook ;
- logique de production et fonctions locales dans les cellules au lieu des fonctions partagées du package ;
- reconstruction de pseudo-comptages à partir d'une matrice log-normalisée ;
- emploi d'un évaluateur moderne au lieu de l'environnement historique `scib==1.1.3` ;
- MMD détachée du graphe autograd avec `mmd.item()` et comparaison limitée à deux groupes malgré quatre espèces ;
- paramètres de `cross_attn` non dégelés correctement ;
- absence des contrôles Random et TranscriptFormer dans la comparaison finale.

## Cohorte corrigée

| Espèce | Cellules |
| --- | ---: |
| humain | 16 177 |
| macaque crabier | 8 682 |
| souris | 1 613 |
| porc | 3 096 |
| **Total** | **29 568** |

Les 13 annotations expertes `celltype` sont conservées pour l'évaluation scIB. Les comptages d'entrée sont entiers et l'ordre des cellules est identique entre les cinq méthodes.

## Audit des gènes

Le chiffre de 1 768 ne correspondait pas au contenu des jeux GEO. Il résultait d'une intersection à quatre espèces de petits fichiers d'orthologie. La préparation corrigée distingue désormais explicitement les features de la matrice source, le vocabulaire small-v2, le chevauchement source/checkpoint, les gènes du checkpoint réellement exprimés et les orthologues souris réellement mappés.

| Espèce | Features source | Vocabulaire small-v2 | Chevauchement source/checkpoint | Gènes checkpoint exprimés | Orthologues souris mappés |
| --- | ---: | ---: | ---: | ---: | ---: |
| humain | 33 660 | 20 004 | 16 662 | 15 132 | 15 406 |
| macaque crabier | 36 162 | 21 591 | 14 244 | 13 237 | 10 555 |
| souris | 31 053 | 21 550 | 18 925 | 15 280 | 31 053 |
| porc | 25 880 | 22 002 | 14 984 | 13 545 | 13 039 |

Pour scPRINT-2, la matrice combinée est un bloc diagonal de 85 147 colonnes couvrant les quatre vocabulaires du checkpoint. Le collateur sélectionne uniquement le bloc de l'organisme de chaque cellule avant l'appel au collateur scPRINT. Les colonnes absentes sont remplies par zéro pour respecter la forme du modèle ; elles ne sont pas comptées comme des gènes observés.

Pour TranscriptFormer, PCA et Random, les mappings Ensembl BioMart à haute confiance sont résolus de façon déterministe dans une union de 31 619 symboles orthologues souris, dont 9 408 sont communs aux quatre espèces. Cette union est ensuite alignée sur les 57 066 colonnes de la table de référence scPRINT utilisée par le benchmark historique.

### Réserve sur le seuil de 15 000 gènes

Il n'est pas scientifiquement correct d'affirmer que chaque espèce possède au moins 15 000 orthologues observés dans cette source. L'humain et la souris dépassent ce seuil selon au moins une définition mesurée, mais le macaque et le porc restent en dessous pour les gènes exprimés ou les orthologues explicitement mappés. Une analyse séparée du macaque atteint au mieux 14 748 identifiants rhesus small-v2 avec un mapping explicite ; une inférence par alias/symbole dépasserait artificiellement 15 000 mais a été rejetée car trop ambiguë.

Le fichier GEO macaque contient bien 36 162 features uniques : la limite vient de nombreux identifiants `LOC`, `MSTRG` et suffixes non reliés sans ambiguïté à une référence Ensembl, pas d'une troncature du téléchargement ou du lecteur. Pour obtenir honnêtement au moins 15 000 gènes observés et mappés, il faudrait re-quantifier les lectures de `SRP255873` / `PRJNA624096` contre une annotation Ensembl Macaca mulatta, ou récupérer le GTF/crosswalk original de l'étude.

## Labels utilisés pour le fine-tuning

Les deux termes absents du checkpoint ont été remplacés par des termes biologiquement compatibles présents dans small-v2 :

| Annotation experte | Ancienne cible absente | Cible FT small-v2 |
| --- | --- | --- |
| Beam A | `CL:7770003` | `CL:0000327` |
| JCT | `CL:7770002` | `CL:0002320` |
| SChlemm's Canal | terme oeil trop spécifique | `CL:0000115` |

`CL:0002367` a également été contrôlé mais n'est pas présent dans ce checkpoint. Après remplacement, les 13 labels sont connus : `checkpoint_unknown_labels=[]`, aucune cellule n'est exclue et les 29 568 cellules participent au FT.

## Modèles et protocole

- `PCA` : PCA50 de l'expression normalisée dans l'espace orthologue souris ;
- `Random-seed42` : contrôle négatif uniforme à 50 dimensions, seed 42 ;
- `scPRINT-2-ZS-native-cell-token` : small-v2 zero-shot, token cell type, kNN multi-cellules ;
- `scPRINT-2-FT-native-cell-token-MMD003` : small-v2 FT reconstruction + MMD inter-espèces 0,03, token cell type, kNN multi-cellules ;
- `TranscriptFormer-Metazoa-mouse-ortholog` : checkpoint Metazoa dans l'espace orthologue souris, embedding 2 048 dimensions.

Le FT utilise les quatre groupes `species`, une MMD différentiable équilibrée, huit époques au maximum et la restauration du meilleur checkpoint de validation. Le meilleur état restauré est celui de l'époque 5. La loss exacte n'est pas rapportée ici car le log d'entraînement et le champ récapitulatif du metadata divergent, bien que l'époque restaurée et le checkpoint soient cohérents.

## Scores scIB 1.1.3

| Méthode | Batch correction | Bio conservation | Total |
| --- | ---: | ---: | ---: |
| scPRINT-2 FT, cell-type token, MMD 0.03 | 0,595199 | **0,845159** | **0,745175** |
| TranscriptFormer Metazoa | 0,485493 | 0,767265 | 0,654556 |
| Random seed 42 | **0,730124** | 0,457305 | 0,566433 |
| scPRINT-2 zero-shot, cell-type token | 0,337092 | 0,640396 | 0,519074 |
| Expression PCA50 | 0,249832 | 0,605130 | 0,463011 |

Chaque ligne possède cinq métriques de batch et six métriques biologiques. Le total a été recalculé comme `0.4 * Batch correction + 0.6 * Bio conservation`. Les hashes des connectivités sont identiques avant et après l'appel scIB pour les cinq graphes, et les copies finales des graphes correspondent aux hashes consignés dans le metadata.

L'environnement exact est : `scib==1.1.3`, `scanpy==1.9.1`, `anndata==0.8.0`, `numpy==1.21.6`, `pandas==1.3.5` et `scikit-learn==1.0.2`.

Pour la trajectoire DPT de référence, les cellules appartenant à des composantes d'expression déconnectées restent `NaN`, conformément à l'API trajectoire de scIB 1.1.3 : 25 cellules souris et 34 cellules porc. Toutes les cellules humaines et macaques ont un pseudotime fini. Une espèce ne serait rejetée que si aucune cellule n'avait de pseudotime fini.

## Provenance d'exécution

- racine distante : `/lustre/fswork/projects/rech/xeg/uat95fg/data/eye_cross_species/eye4_task3_corrected_20260823` ;
- accès Ensembl BioMart : depuis le submit node Jean Zay avec le proxy du submit node ;
- préparation native : Slurm `1298857`, `COMPLETED 0:0`, 5 min 20 s ;
- préparation orthologue : Slurm `1298858`, `COMPLETED 0:0`, 4 min 19 s ;
- TranscriptFormer, PCA et Random : Slurm `1298962`, `COMPLETED 0:0`, 8 min 50 s ;
- scPRINT-2 zero-shot et FT : Slurm `1300384`, `COMPLETED 0:0`, 17 min 52 s ;
- scIB par méthode : array Slurm `1303721`, cinq tâches `COMPLETED 0:0` ;
- validation et collecte scIB : Slurm `1303738`, `COMPLETED 0:0`, 16 s ;
- UMAPs des graphes réellement scorés : Slurm `1303750`, `COMPLETED 0:0`, 3 min 58 s.

## Vérifications locales

- suite hors smoke test lourd : 170 tests réussis ;
- Ruff : aucune erreur sur les fichiers modifiés ;
- `compileall` : réussi ;
- `git diff --check` : réussi ;
- notebooks explicatifs : exécution locale complète, aucune définition de fonction locale et aucune cellule shell/runner ;
- cinq méthodes uniques, cinq marqueurs par méthode et marqueur final scIB présents.
- 19 artefacts locaux vérifiés contre les SHA-256 générés sur Jean Zay, y compris les quatre archives GEO matérialisées localement ;
- planche UMAP 2 x 5 contrôlée visuellement en 6 211 x 2 357 px, sans panneau vide, légende tronquée ni méthode manquante.

## Limites d'interprétation

Ce benchmark porte sur un seul tissu et quatre espèces. Le score Random illustre pourquoi la composante batch ne doit pas être interprétée seule : elle est élevée alors que la conservation biologique est faible. La comparaison dépend des annotations expertes, de la seed 42, de l'orthologie retenue et du choix de la MMD. Le macaque crabier (`NCBITaxon:9541`) utilise le vocabulaire rhesus (`NCBITaxon:9544`) comme proxy de features parce que small-v2 ne contient pas de vocabulaire cynomolgus natif ; ce proxy de forme modèle ne transforme pas les gènes absents en observations.
