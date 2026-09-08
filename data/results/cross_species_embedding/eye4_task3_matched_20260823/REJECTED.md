# Rejected benchmark bundle

This bundle is retained only for provenance and must not be used for model comparison.

It was rejected on 2026-08-23 because:

- the feature preparation reduced the four species to a 1,768-gene four-way intersection built from incomplete orthology tables;
- the scPRINT-2 fine-tuning targets included `CL:7770002` and `CL:7770003`, which are absent from the small-v2 decoder and were silently treated as unknown;
- consequently, its embeddings, scIB scores and UMAPs do not satisfy the corrected analysis contract.

The replacement analysis is `eye4_task3_corrected_20260823`, which uses native per-species small-v2 vocabularies for scPRINT-2, supported CL labels, official Ensembl BioMart mappings for the mouse-ortholog baseline space, all five methods, and a fresh joint `scib==1.1.3` run.
