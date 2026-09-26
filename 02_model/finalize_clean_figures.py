"""Validate complete clean reevaluation, render staged PDFs and refresh captions.
Promotion to 04_figure is a separate explicit step after visual review.
"""
import os
os.environ['BITFI_CLEAN']='1'
from pathlib import Path
import sys,json,hashlib,shutil,ast
import numpy as np,pandas as pd
sys.path[:0]=[str(Path(__file__).resolve().parent/'figures')]
import clean_protocol as cp
import build_prism_figures as b
from FigureS7_univariate_backbones import main as s7
from FigureS8_moment_tuning import main as s8

def main():
 expected=list({**b.SRC,**b.NEWSRC});manifest=pd.read_csv(cp.OUT/'mask_manifest.csv');checks={};reference_cells=None;oldroot=cp.ROOT/'04_figure/archive/pre_leakage_fix_20260911'
 for name in expected:
  folder=cp.OUT/'evaluation'/name
  assert (folder/'complete.json').exists(),name
  d=pd.read_csv(folder/'results.csv');assert set(d.case_id)==set(manifest.case_id),name
  assert not d.duplicated(b.KEY).any(),name
  assert np.isfinite(d.NMAE).all(),name
  cells=d.set_index(b.KEY)['n_eval'].sort_index()
  if reference_cells is None:reference_cells=cells
  else:pd.testing.assert_series_equal(cells,reference_cells,check_names=False)
  inv=json.loads((cp.OUT/'smoke'/name/'invariance.json').read_text());assert inv['ABA_order_invariance'] and len(inv['raw_poison_checks'])==15
  assert all(x['raw_hidden_truth_poison_invariant'] for x in inv['raw_poison_checks'])
  checks[name]=json.loads((folder/'complete.json').read_text())
 for family in ['tfm3','chronos']:assert (cp.OUT/f'{family}_auxiliary_complete.json').exists(),family
 for name in ['Chronos2','TimesFM2.5','TimesFM3.0']:assert (cp.OUT/'univariate_backbone_comparison'/f'{name}_complete.json').exists(),name
 for name in ['SAITS','MOMENT-FT','Spatial-Ridge','BiTFI']:assert (cp.OUT/'figure6_common_window'/f'{name}.csv').exists(),name
 out=b.OUT;out.mkdir(exist_ok=True);b.DATA.mkdir(exist_ok=True)
 for name in ['Figure1_Dataset_overview.pdf','Figure1_Dataset_profile.pdf','Figure1_Dataset_profile_caption.md','Figure2_BiTFI_framework.pdf','Figure2_BiTFI_caption.md']:
  if (oldroot/name).exists():shutil.copy2(oldroot/name,out/name)
 if (oldroot/'source_data/Figure1_profile').exists():shutil.copytree(oldroot/'source_data/Figure1_profile',b.DATA/'Figure1_profile',dirs_exist_ok=True)
 for node in ast.walk(ast.parse((cp.ROOT/'02_model/figures/Figure1_dataset_profile.py').read_text())):
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='caption' for t in node.targets):
   (out/'Figure1_Dataset_profile_caption.md').write_text(ast.literal_eval(node.value))
 framework_caption=out/'Figure2_BiTFI_caption.md'
 framework_caption.write_text(framework_caption.read_text()+'\n\nClean evaluation clarification (2026-09-11): Artificial targets are masked before any inference-time initialization. Fixed scaling is learned only from training prefixes. Neighbor selection is recomputed independently for each mask. Unavailable model backends or unexpected inference errors abort the corrected evaluation; no pre-mask reconstruction cache is consumed. The conceptual figure is unchanged.\n')
 # Dataset description and conceptual framework are unaffected; all performance figures rebuilt.
 b.ps.setup();full,main=b.load_results();b.overall(full);b.gap_robustness(full);b.variables(full);metrics,limits=b.examples();b.supplementary(full,main);s7();s8()
 full.groupby('model').agg(rows=('NMAE','size'),source=('source_directory','first'),context_len=('context_len','first')).to_csv(b.DATA/'result_sources.csv')
 before=pd.read_csv(oldroot/'source_data/all_model_summary.csv');after=pd.read_csv(b.DATA/'all_model_summary.csv');change=before[['model','NMAE']].merge(after[['model','NMAE']],on='model',suffixes=('_previous','_clean'));change['relative_change_percent']=100*(change.NMAE_clean/change.NMAE_previous-1);change.to_csv(b.DATA/'pre_post_audit_summary.csv',index=False)
 a=manifest.merge(pd.read_csv(cp.ROOT/'03_result/comparison_saits/mask_manifest.csv'),on=['greenhouse','scenario','masked_vars','gap_length_h','repeat'],suffixes=('_new','_old'))
 overlap=int(((a.start_idx_new==a.start_idx_old)&(a.end_idx_new==a.end_idx_old)).sum())
 protocol=json.loads((cp.OUT/'protocol.json').read_text());train=json.loads((cp.OUT/'models/SAITS/training_protocol.json').read_text());mt=json.loads((cp.OUT/'models/MOMENT/complete.json').read_text());mst=json.loads((b.DATA/'moment_tuning_statistics.json').read_text())
 captions=f'''# Figure captions — clean reevaluation, 11 September 2026

## Shared evaluation protocol

All performance results in this package were regenerated under clean-20260911. The same ten held-out greenhouses, scenarios A/B/C, gap durations 6/12/24/72/168 h, and up to ten repetitions yield {len(manifest):,} mask jobs and {checks['BiTFI-TimesFM3']['all_cells']:,} variable-level cases. Only genuine observations after physical-range filtering are eligible as hidden truth. No short-gap interpolation is performed before selecting or applying masks. One fixed MinMax scaler per variable is fitted exclusively to the first 80% of the training-greenhouse timelines; validation and test values are excluded. Original missingness is retained, and any inference-time filling is computed from the current masked input. BiTFI never consumes a pre-mask reconstruction cache; neighbor selection is recomputed per mask. Backends must be available and inference errors abort evaluation.

All main configurations use the same mask manifest. The 1,440 h minimum left-context requirement concerns placement; SAITS and MOMENT use a total 512 h gap-centered window, classical forecast baselines use 720 h context, and TimesFM3/BiTFI configurations use a 1,440 h context limit; the existing 1,080 h limits for Chronos 2 and TimesFM2.5 are retained unless explicitly varied in S2. Retrospective methods may use post-gap observations. Cross-greenhouse inputs come only from the training pool; those series can include contemporaneous observations under the retrospective availability assumption. Uni­variate comparisons in S6 are strictly past-only. Model input configurations are intentionally distinct and should not be presented as identical-information algorithms.

Physical-unit MAE is divided by a fixed full raw greenhouse-variable range for scoring only. Scores are n_eval-weighted within greenhouse and then equally averaged across the ten greenhouses. Percentile 95% intervals resample greenhouses 10,000 times (seed 42). Grouped season/time-of-day records are not counted twice. These intervals do not capture training-seed variability. The evaluation greenhouses were previously consulted during exploratory model/context and illustration selection; correcting input leakage does not make this a newly untouched external test set. New masks differ from previous masks: {overlap:,} jobs have the same keyed window. Changes versus the archived results therefore combine leakage-control changes, retraining and changed eligible windows; they are not a pure estimate of leakage inflation.

Figure 1 describes the same underlying dataset and Figure 2 the same conceptual algorithm. Both are retained. Figures 3–6 and S1–S7 below use the corrected evaluations. Raw result IDs beginning BiTFI remain for traceability; the displayed method is BiTFI. MOMENT denotes the newly fine-tuned reconstruction head with frozen encoder.

## Figure 1. Dataset profile.

The underlying greenhouse dataset and its descriptive display are unchanged. The eligibility mask for the corrected performance evaluation uses original observations, not any descriptive preprocessing used in the dataset overview. See the accompanying Figure1_Dataset_profile_caption.md for the retained descriptive sampling and preprocessing.

## Figure 2. BiTFI framework.

Bidirectional univariate initialization (R0) is followed by one bidirectional covariate refinement (R1). R1 uses a fixed R0 snapshot, observed local sensors, calendar features and training-greenhouse same-variable series. The target's own hidden truth is never supplied. Forward and reversed-backward outputs are aligned and linearly fused by distance to the gap boundaries. The backbone remains frozen. See the accompanying framework caption for the panel-level description.

## Figure 3. Overall reconstruction performance.

(A) Nine model/configuration means, individual greenhouse points and bootstrap intervals. (B) Paired relative NMAE reduction with BiTFI against eight alternatives; intervals use paired greenhouse resampling. Two-sided paired Wilcoxon tests are Holm-adjusted across the eight contrasts. Printed values are the mean NMAE in A and relative reduction in B. Positive reduction favors BiTFI; negative reduction favors the comparator. TimesFM3 configurations distinguish univariate, local covariates, and local plus cross-greenhouse covariates. MOMENT is the validation-selected fine-tuned model.

## Figure 4. Robustness to gap duration and sensor-failure scenario.

Mean greenhouse NMAE across five gap durations under (A) single-variable gaps, (B) Tin/RH/CO2 gaps and (C) all-five-variable gaps. Eight methods are linear interpolation, seasonal naive, LightGBM, Spatial ridge, SAITS, MOMENT, TimesFM3 and BiTFI. In Figures 4–6, TimesFM3 means the local plus cross-greenhouse covariate configuration, and BiTFI uses the TimesFM3 backbone. Spatial ridge is same-variable regression on training-greenhouse neighbors selected from available target observations. Horizon positions are categorical; lines guide the eye. Intervals are omitted for readability and site-level source scores are supplied. Scenario B aggregates only its masked indoor variables.

## Figure 5. Variable-specific performance.

The same eight configurations as Figure 4 are compared for indoor temperature, outdoor temperature, humidity, CO2 and radiation. Diamonds and intervals show greenhouse means and bootstrap uncertainty. Variables use their own physical ranges for NMAE; axis scales differ between panels. Model labels and color identities follow Figure 4.

## Figure 6. Reconstruction at the fixed 72 h illustrative window.

Rows A/B/C and five variable columns show PF_0025101_01 from 8 March 2025 19:00 to 11 March 2025 18:00, with 72 h preceding and 24 h following display context. This window was selected post hoc for strong BiTFI performance in the PREVIOUS evaluation and is deliberately not reselected here. It is not representative evidence of average superiority. All four models (BiTFI, SAITS, fine-tuned MOMENT and Spatial ridge) were rerun under the corrected protocol. The 360 hidden variable-hour truth entries are genuine measurements. Original physical axis limits are retained; some Spatial ridge RH predictions fall outside them, while metrics use their full values.

Gray context, pale blue gap shading, black dashed withheld truth and muted baselines follow the established style; the magenta BiTFI line is thicker. Truth is drawn above predictions. Each compact table reports BiTFI and the lowest-MAE baseline among the three alternatives, with R² and MAE calculated only over the hidden observations. Baseline choice is based on MAE, not R². Observed outdoor covariates in scenario B are muted and have no performance box. Full values, including out-of-axis predictions, are supplied in the source tables. A near-perfect illustrative curve does not supersede the aggregate comparisons.

## Figure S1. Extended model comparison.

Eighteen configurations are ordered by mean NMAE on the common corrected masks. All CAFI variants and zero-shot MOMENT are omitted from this panel. Individual greenhouse points and bootstrap intervals are shown. Model configurations and actual source directories are recorded in source_data/result_sources.csv; there are no imported pre-audit scores.

## Figure S2. Context-length sensitivity.

Matched-input context sensitivity for Chronos 2, TimesFM2.5 and TimesFM3. (A) Pre-gap univariate target input. (B) Identical four local-variable and four calendar covariates across context and gap. No post-gap target or cross-greenhouse records. TimesFM2.5 uses official XReg + TimesFM with case-independent OLS regression; the other models use direct covariate inputs. Forty-two configurations share 299 Scenario A cases (24/168 h, first three repeats), seven context limits (96–1900 h), preprocessing and masks. Hour-weighted greenhouse NMAEs are averaged equally across sites. Panels use different y-axis scales; panel A is autoscaled to its data. Input availability is matched, but covariate-integration mechanisms differ. This test-set sensitivity analysis does not retune main context choices; S6 uses the full univariate benchmark instead of this subset.

## Figure S3. SAITS training diagnostics.

Training objective and fixed-validation masked MAE are shown for a newly trained official SAITS network. There are {train['train_windows']} training and {train['validation_windows']} validation windows, length 512, from training greenhouses only. Windows do not cross each site's 80/20 temporal boundary; raw natural missingness is retained. Scalers use training prefixes only. Training corruption mixes MCAR and A/B/C contiguous gaps; validation corruption is fixed. Checkpoint selection uses validation only, with a 60-epoch cap, patience 10 and seed 42. The dashed line indicates the selected epoch.

## Figure S4. Seasonal performance.

Corrected NMAE is aggregated within greenhouse and season, then equally across available greenhouses. Only masked target observations are scored. Diamonds and intervals represent means and greenhouse bootstrap uncertainty. The model set follows the original eight-method main set used for this supplementary comparison; explicit configuration labels avoid confusion with the shortened Figure 4–6 labels.

## Figure S5. Information-source configurations.

(A) TimesFM3 univariate, local covariates, local plus cross-greenhouse covariates, and BiTFI. (B) Spatial ridge, covariate TimesFM3 and BiTFI for indoor versus outdoor variables. All use the common corrected masks; the aggregation in B averages variable-specific greenhouse NMAE within each indoor/outdoor category. This provides an information-source comparison rather than an exhaustive factorial causal ablation.

## Figure S6. Univariate backbone and BiTFI-backbone comparisons.

(A–C) Chronos 2, TimesFM2.5 and TimesFM3 use identical 1,440 h past-only target contexts and the corrected cases, without other sensors, other greenhouses or post-gap observations. Natural missingness is filled strictly inside the past slice. Each backbone predicts the complete gap in one call; this differs from chunked historical baseline adapters. Scaling is fixed from training prefixes. Models remain frozen. Means and bootstrap intervals are greenhouse-level. (D) A separate paired comparison evaluates full BiTFI with Chronos 2 versus TimesFM3 under the same corrected masks and bidirectional/covariate availability. D is not a univariate experiment. Source tables provide the numerical comparisons and multiplicity-adjusted tests; no superiority claim is assumed in advance.

## Figure S7. MOMENT fine-tuning diagnostics.

Only the 8,200-parameter reconstruction head is trained; the encoder remains frozen in evaluation mode. Raw-observation training/validation windows and training-prefix-only scaling match S3. Fixed contiguous masks cover 6/12/24/72/168 h, with independent-channel inference. Three learning rates (1e-4, 3e-4, 1e-3), AdamW, weight decay 1e-4, dropout 0.1, a 30-epoch cap and patience 5 are retained; selection uses validation only. Frozen-feature equivalence is checked before head optimization. (A) Validation curves and selected checkpoint. (B) Paired test-greenhouse results before and after tuning. (C–D) Results by gap duration and variable. Zero-shot MOMENT is retained here solely as an explicit training-effect control. Test NMAE changes from {mst['zero_shot_NMAE']:.6f} to {mst['head_tuned_NMAE']:.6f}; relative reduction is {mst['relative_reduction_percent']:.2f}%, with paired bootstrap interval [{mst['CI_low']:.2f}, {mst['CI_high']:.2f}]%. Single-seed training does not establish seed robustness.
'''
 (out/'Figure_captions.md').write_text(captions)
 for n,label in [(7,'Univariate_backbone_comparison'),(8,'MOMENT_head_tuning')]:
  section=captions.split(f'## Figure S{n}. ')[1].split('\n## Figure ')[0]
  (out/'Supplementary'/f'FigureS{n}_caption.md').write_text(f'# Figure S{n}. '+section)
 (out/'model_palette.json').write_text(json.dumps({m:dict(label=b.ps.LABELS[m],hex=b.ps.COLORS[m]) for m in b.ps.COLORS},indent=2))
 validation=dict(protocol=protocol,models=checks,raw_truth_only=True,main_masks_matched=True,univariate_contexts_past_only=True,completed_figures=14,export_formats=['Vector PDF'],previous_same_key_window_count=overlap,batch_equivalence=json.loads((cp.OUT/'batch_equivalence.json').read_text()),protocol_validation=json.loads((cp.OUT/'protocol_validation.json').read_text()),raw_hidden_poison_checks=15*len(expected),ABA_order_checks=len(expected),all_stratified_cells_and_counts_matched=True)
 validation['regression_records']={name:json.loads((cp.OUT/name).read_text()) for name in ['regression_checks.json','public_api_regression.json','dispatcher_equivalence.json','spatial_dispatcher_equivalence.json','worker_count_equivalence.json','metric_equivalence.json','rf_thread_equivalence.json']}
 (out/'validation_report.json').write_text(json.dumps(validation,indent=2))
 files=[*Path(cp.ROOT/'02_model').glob('*clean*.py'),cp.ROOT/'02_model/bitfi.py',cp.ROOT/'02_model/spatial.py',cp.ROOT/'02_model/train_saits.py',cp.ROOT/'02_model/train_moment_head.py',cp.ROOT/'02_model/figures/build_prism_figures.py',cp.ROOT/'02_model/figures/prism_style.py']
 (out/'reproducibility_manifest.json').write_text(json.dumps(dict(protocol=protocol,code_sha256={str(p.relative_to(cp.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},checkpoints={name:hashlib.sha256((cp.OUT/f'models/{name}/best.pt').read_bytes()).hexdigest() for name in ['SAITS','MOMENT']}),indent=2))
 manifest_path=out/'reproducibility_manifest.json';repro=json.loads(manifest_path.read_text())
 for relative in ['02_model/models/foundation_model.py','02_model/models/autogluon_model.py','02_model/models/imputation_models.py','02_model/clean_metrics.py','02_model/final_table.py','02_model/run_univariate_backbone_comparison.py','02_model/figures/FigureS6_univariate_backbones.py','02_model/figures/FigureS7_moment_tuning.py']:
  path=cp.ROOT/relative;repro['code_sha256'][relative]=hashlib.sha256(path.read_bytes()).hexdigest()
 repro['training_checkpoint_files_sha256']={str(path.relative_to(cp.ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in (cp.OUT/'models').rglob('*') if path.is_file() and path.suffix in ['.pkl','.pt','.ckpt'] and path.name!='windows.npz'}
 repro['protocol_validation']=json.loads((cp.OUT/'protocol_validation.json').read_text())
 manifest_path.write_text(json.dumps(repro,indent=2))
 print(change.to_string(index=False));print('STAGED FIGURES COMPLETE',out,flush=True)
if __name__=='__main__':main()
