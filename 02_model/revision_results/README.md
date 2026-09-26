# Executed experiment summaries

These tables are generated from the final full-precision runs, including production-batch reconciliation for numerically sensitive cases. They contain greenhouse-level aggregate results, not weights or individual prediction files. Main benchmark data remain in the raw-data snapshot; run the documented pipeline to regenerate case-level outputs.

The main display contains eight representative configurations. All 21 settings remain in all_summary.csv, including all SAITS inputs and both MOMENT adaptation settings. The display choice does not alter model predictions.


The current primary configuration uses five synchronous covariate-refinement passes selected exclusively on training-site validation. The `refinement_validation/` directory contains the pre-inference criterion, manifest, greenhouse scores, selected minimum, A6000 timing scope and selected-depth invariance checks. Unchanged baselines retain their existing weights and cases. The test depth table is a sensitivity analysis; it does not select the default. Manuscript PDF hashes are maintained only in the local publication record.
