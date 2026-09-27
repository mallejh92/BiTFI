"""Build supplementary protocols and tables without carrying old numerical text."""
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
import clean_protocol as cp
from build_hourly_manuscript import table,figure,n,ci,pct,pp,SHORT
from revision_config import MAIN_MODELS
sys.path.insert(0,str(cp.ROOT/'02_model/figures'))
import prism_style as ps
R=cp.OUT;A=R/'analysis';P=R/'publication'
VL={'Tin':r'$T_{\mathrm{in}}$','Tout':r'$T_{\mathrm{out}}$','RH':'RH','CO2':r'CO$_2$','Rad':'Rad'}
STRATA={'change_low':'Lower change','change_high':'Higher change','radiation_zero':'Zero irradiance','radiation_positive':'Positive irradiance'}

def main():
 s=pd.read_csv(A/'all_summary.csv').set_index('model');main=MAIN_MODELS;cols=cp.COLS;context=json.loads((R/'context_validation/protocol.json').read_text());depth=json.loads((R/'refinement_validation/selection.json').read_text())['selected_refinements'];vp=json.loads((R/'refinement_validation/protocol.json').read_text());q=json.loads((R/'quality/protocol.json').read_text());sa=json.loads((R/'models/SAITS/training_protocol.json').read_text());sc=json.loads((R/'models/SAITS/complete.json').read_text());mh=json.loads((R/'models/MOMENT/training_protocol.json').read_text());timing=json.loads((A/'a6000_timing.json').read_text());sel=json.loads((R/'figure6_common_window/selection.json').read_text())
 head=r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=2cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{graphicx,caption,booktabs,tabularx,amsmath,hyperref,placeins}
\usepackage{txfonts}
\graphicspath{{./}}
\hypersetup{hidelinks}
\renewcommand{\thefigure}{S\arabic{figure}}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\figurename}{Fig.}
\setlength{\emergencystretch}{2em}
\begin{document}
\begin{center}\Large\textbf{Supplementary Material}\par\medskip
\normalsize Bidirectional imputation of missing greenhouse sensor data using frozen time-series foundation models\end{center}
'''
 protocol=rf'''\section*{{Data and evaluation protocol}}
Each greenhouse is represented on a complete hourly grid from its first to last recorded timestamp. Missing timestamp rows remain NaN, as do observations excluded by physical screening. Duplicate or off-hour timestamps are rejected. All {q['sites']} main-cohort time indexes, {q['mask_cases']:,} artificial gap durations and their input-context intervals passed elapsed-time checks. The main evaluation contains {q['variable_cases']:,} scored variable-cases; overlapping windows are not independent statistical units. A single min--max scaler per variable is fitted to the pooled first 80\% of each training-site hourly recording span. Validation and test values never refit these scalers. Physical-unit MAE is divided by the full physically valid greenhouse-variable range for test NMAE; this range is used only for scoring.

Fig.~1 uses the same screened hourly data without interpolation. It samples at most 3,000 observed values per site and variable (seed~0) for distributions. Missingness and correlations weight sites equally; missingness includes absent hourly timestamp rows. Constant-value flags are defined by at least 48 consecutive finite hourly values with exact equality. Missing positions interrupt a run. Flags are not confirmed faults and are retained in the primary experiment. Supplementary Table~S7 excludes affected variable-cases for sensitivity analysis.

Inputs are constructed after masking. Reference greenhouses come only from the training pool, with observed reference values for at least 80\% of target hourly positions and 500 jointly observed pairs; the three highest absolute correlations are selected. Target hidden values cannot enter ranking. Natural gaps in selected reference series are interpolated within those series. Context-side lengths are hours on the complete grid. Forecasting adapters with no usable history retain a masked-input interpolation estimate; unexpected backend failures abort evaluation.

\section*{{Selection and model training}}
Univariate context selection uses {context['cases']:,} gaps from the final chronological 20\% of {len(context['sites'])} training greenhouses. Candidates are 96, 168, 336, 720, 1,080, 1,440 and 1,900~h. Each candidate uses the same gaps, a target-only pre-gap context and one complete-gap prediction call. The shortest context must contain at least one observed target value; every gap has the longest requested preceding span. Filling is restricted to that supplied past slice. Validation errors use training-prefix ranges and equal greenhouse weights after within-site hour weighting. The minimum-error length is selected separately per backbone, with exact ties choosing the shorter candidate. All three selected 1,900~h, which is fixed for each backbone's input configurations. The {context['feature_poison_checks']:,} context-input perturbation checks passed. This does not optimize each multivariate or covariate setting independently.

Refinement validation uses {vp['cases']:,} A/B/C gaps from {len(vp['sites'])} training greenhouses. Candidate counts are 1, 2, 3 and 5, with Step~1 alone as a zero-pass control. Fixed scalers, training-prefix scoring ranges and the exclusion of the validation target site from its reference bank are retained. Minimum mean greenhouse NMAE selects {depth} passes; exact ties favor fewer passes. The same depth is used for TimesFM3 BiTFI, its forward-only and Chronos~2 controls, and additional-site evaluation. Test depth sensitivity does not select the production depth. Backend batches group identical context lengths to avoid padding-dependent differences.

The five-channel SAITS network has two groups with one inner layer each, width 256, inner width 128, four attention heads, key/value width 64 and dropout 0.1. Adam uses learning rate $10^{{-3}}$ and batch size 16. Training is capped at 60 epochs, with patience 10. The {sa['train_windows']} training and {sa['validation_windows']} validation windows have length 512, strides 128 and 256 respectively, and do not cross the per-site split. Eligibility requires 80\% finite entries and at least 48 observed values per channel. Training corruption draws uniformly among MCAR masking at rate 20\% and contiguous A/B/C gaps. Observed-reconstruction and masked-imputation losses have equal weights; fixed validation masked MAE selects the checkpoint. Training ran for {sc['epochs']} epochs.

The matched SAITS controls have 24 channels: five local targets, four calendar features and three reference slots per target. Both have identical initialization, windows and fixed corruption realizations (four per training window and one per validation window), with the same optimizer and stopping rule. The local control sets reference values and masks to unavailable. Training references use only training prefixes and exclude the target greenhouse; validation references also exclude that greenhouse. Both losses score only the five local targets. Thus the information-control contrast holds channel capacity constant.

MOMENT retains a frozen encoder and tunes only its 1,024-to-8 linear reconstruction head (8,200 parameters). Channels are reconstructed independently. AdamW uses weight decay $10^{{-4}}$, dropout 0.1, 30 epochs, patience 5 and a batch of 512 masked patches. Learning rates are $10^{{-4}}$, $3\times10^{{-4}}$ and $10^{{-3}}$. Each training window supplies two fixed gap corruptions per channel; validation supplies five, covering all gap lengths. Frozen features are checked against direct reconstruction before caching. Validation masked MAE selects the head; training-search results and the unadapted comparison are shown in Fig.~S7.

AutoGluon-TimeSeries trains LightGBM, random forest, DeepAR and PatchTST on training prefixes, with separate final-20\% tuning slices. Each slice is forward-filled independently and leading missing values become zero. Each model has a 600-s training budget and a 720-h past-only context; forecasts roll forward when the gap exceeds the prediction horizon. Seasonal naive searches same-hour references at 24-h offsets on the complete hourly grid within its stated 720-h boundaries, with masked-input interpolation as fallback. Spatial ridge fits each masked job on available target observations, using the selected same-variable reference series and ridge penalty 1.

\section*{{Integrity checks, additional analyses and timing}}
Hidden-target perturbation and execution-order checks cover all 15 scenario--duration combinations for the 19 settings outside the two matched SAITS controls. The matched SAITS feature construction also checks that poisoned unavailable targets do not alter supplied features. Refinement validation excludes self-reference and checks hidden-target invariance. These checks support the implemented input pathways rather than unknown pretraining-data membership. The test sites were consulted in exploratory development, so they are not an untouched external validation. Source data, input rules and execution scripts are available at \url{{https://github.com/mallejh92/BiTFI}}.

The matched single-call univariate comparison uses one common pre-gap input per scored variable-case across all three backbones. The common context is the largest validation-selected length, 1,900~h. This distinguishes a backbone-only comparison from the main multivariate and rolling adapters. The B/C control uses every Scenario~C interval and compares the same indoor targets with outdoor measurements either retained or hidden. The nine additional sites use fixed training-derived settings and three indoor variables; they are from the same provider rather than an independent external source.

Residual strata use training-prefix 75th percentiles of absolute one-hour changes: 1~$^\circ$C for each temperature, 2 percentage points for RH, 35~ppm for CO$_2$ and 85~W\,m$^{{-2}}$ for radiation. Missing adjacent values are excluded from this stratification. Zero and positive radiation are also summarized separately; these measured-irradiance states are not a validated astronomical day/night classification. Greenhouse means and bootstrap intervals are computed separately for each stratum. These descriptive associations do not identify heating, ventilation, enrichment or crop-physiological events.

The Fig.~6 interval starts at {sel['start']} and ends at {sel['end']}. It is the middle entry after sorting {sel['eligible_cases']} Scenario~C 72-h cases by greenhouse identifier and start time, with at least 90\% genuine observations in each variable across the display span. Selection used no model outputs. All models and scenarios use this common interval; every displayed prediction contributes to the axis range and metric. The lowest-MAE displayed baseline is selected separately per panel. The example is illustrative rather than a sample of typical performance.

RTX A6000 (48~GB) synchronized backbone inference, fusion and refinement-covariate construction is timed over {timing['cases']:,} test masks. Reported cumulative time excludes loading, initial masking and scoring. TimesFM3 also ran on an RTX PRO 6000 Blackwell GPU for designated computation shards; A6000 timing summaries exclude that device. Training histories and run metadata identify the device for each experiment. Two pinned Python dependency manifests and the SAITS source revision accompany the public code. No fresh-machine reproduction is claimed merely from these environment snapshots.

\clearpage
'''
 chunks=[head,protocol]
 labels=ps.LABELS
 rows=[[labels[m],n(r['mean']),n(r.sd),ci(r)] for m,r in s.sort_values('mean').iterrows()]
 chunks.append(table(r'All 21 configurations: mean greenhouse NMAE, between-greenhouse SD and 95\% bootstrap CI.','tab:all',['Configuration','NMAE','SD',r'95\% CI'],rows,'lrrr'))
 paired=pd.read_csv(A/'main_paired.csv');rows=[[SHORT[r.reference],pct(r.reduction),f'[{r.low:.2f}, {r.high:.2f}]',n(r.holm_p)] for r in paired.itertuples()]
 chunks.append(table(r'Paired greenhouse comparisons of BiTFI with the main references. Positive reduction favors BiTFI; $p$ values use Holm correction across seven contrasts.','tab:paired',['Reference',r'Reduction (\%)',r'95\% CI',r'Holm $p$'],rows,'lrrr'))
 seasons=pd.read_csv(A/'all_seasons.csv').set_index(['model','group_value']);rows=[[SHORT[m],*[n(seasons.loc[(m,k),'mean']) for k in ['spring','summer','fall','winter']]] for m in main]
 chunks.append(table('Season-specific mean greenhouse NMAE.','tab:seasons',['Model','Spring','Summer','Autumn','Winter'],rows,'lrrrr'))
 va=pd.read_csv(R/'refinement_validation/summary.csv').set_index('refinements');te=pd.read_csv(A/'refinement_summary.csv').set_index('model');by_scenario=pd.read_csv(A/'refinement_scenarios.csv').set_index(['model','scenario']);tests=pd.read_csv(A/'refinement_paired.csv').set_index('reference');rows=[]
 for k in [0,1,2,3,5]:
  model=f'BiTFI-refine{k}';pvals=['---','---'] if k==depth else [n(tests.loc[model,'p']),n(tests.loc[model,'holm_p'])]
  rows.append([str(k),n(va.loc[k,'mean']),n(te.loc[model,'mean']),*[n(by_scenario.loc[(model,sc),'mean']) for sc in ['A','B','C']],f"{timing['seconds_per_mask_by_refinements'][str(k)]:.2f}",*pvals])
 chunks.append(table(r'Refinement depth: validation, overall and scenario-specific test NMAE, and cumulative A6000 time. Zero passes denotes Step~1 alone. The selected five-pass setting is paired with each other depth across ten test greenhouses; Holm correction covers four contrasts.','tab:refine',['Passes','Validation','Overall','A','B','C','s/mask',r'$p$',r'Holm $p$'],rows,'rrrrrrrrr'))
 rows=[]
 for m in ['SAITS','SAITS-matched-local','SAITS-spatial']:
  folder=R/'models/SAITS' if m=='SAITS' else R/'revision/saits_information'/m;history=pd.read_csv(folder/'history.csv');best=history.loc[history.validation_mae.idxmin()];rows.append([labels[m],str(int(best.epoch)),n(best.validation_mae),n(s.loc[m,'mean'])])
 chunks.append(table('SAITS input configurations, validation checkpoint selection and test NMAE. Validation MAE is computed on training-scaled targets.','tab:saits',['Configuration','Epoch','Validation MAE','Test NMAE'],rows,'lrrr'))
 a=pd.read_csv(A/'additional_summary.csv').set_index('model');rows=[[{'LI':'Linear interpolation','Spatial-Ridge':'Spatial ridge','TimesFM3-univariate':'TimesFM3 (univariate)','BiTFI':'BiTFI'}[m],n(r['mean']),n(r.sd),ci(r)] for m,r in a.sort_values('mean',ascending=False).iterrows()]
 chunks.append(table(r'Common indoor-variable reconstruction at nine additional facilities, ordered by decreasing NMAE. SD and CI denote between-site standard deviation and the 95\% bootstrap interval.','tab:additional',['Configuration','NMAE','SD',r'95\% CI'],rows,'lrrr'))
 qc=pd.read_csv(A/'constant_exclusion_summary.csv').set_index('model');rows=[[labels[m],n(s.loc[m,'mean']),n(qc.loc[m,'mean']),str(int(s['mean'].rank().loc[m])),str(int(qc['mean'].rank().loc[m]))] for m in s.sort_values('mean').index]
 chunks.append(table('Sensitivity to excluding variable-cases overlapping constant-value flags. Primary and sensitivity ranks refer to all 21 configurations.','tab:quality',['Configuration','Primary NMAE','Excluding flags','Primary rank','Sensitivity rank'],rows,'lrrrr'))
 metrics=pd.read_csv(R/'figures/source_data/figure6_panel_metrics.csv');rows=[]
 for panel,d in metrics.groupby('panel',sort=True):
  bit=d[d.model=='BiTFI-TimesFM3'].iloc[0];base=d[d.model!='BiTFI-TimesFM3'].sort_values('MAE').iloc[0];rows.append([panel,n(bit.R2),n(bit.MAE),SHORT[base.model],n(base.R2),n(base.MAE)])
 chunks.append(table(r'Gap-only $R^2$ and physical-unit MAE for Fig.~6: BiTFI and the lowest-MAE displayed baseline. Panels G and J are unmasked.','tab:example',['Panel',r'BiTFI $R^2$','BiTFI MAE','Best baseline',r'Baseline $R^2$','Baseline MAE'],rows,'lrrlrr'))
 v=pd.read_csv(A/'all_variables.csv').set_index(['model','variable']);rows=[[SHORT[m],*[n(v.loc[(m,c),'mean']) for c in cols]] for m in main]
 chunks.append(table('Variable-specific mean greenhouse NMAE for the representative configurations.','tab:variables',['Model',*[VL[c] for c in cols]],rows,'lrrrrr'))
 w=pd.read_csv(A/'scenario_duration_weights.csv');rows=[[r.scenario,str(r.gap_length_h),str(r.mask_cases),str(r.variable_cases),f'{r.evaluated_hours:,}',f'{100*r.overall_weight:.2f}'] for r in w.itertuples()]
 chunks.append(table(r'Evaluation composition by scenario and duration. Weight is the share of the overall averaging weight after within-site observation weighting and equal greenhouse weighting; it is not the share of observed error.','tab:weights',['Scenario','Hours','Masks','Variable-cases','Scored hours',r'Weight (\%)'],rows,'lrrrrr'))
 rs=pd.read_csv(A/'residual_strata_MAE.csv');rows=[]
 for r in rs[rs.model=='BiTFI-TimesFM3'].itertuples():rows.append([VL[r.variable],STRATA[r.stratum],f'{r.mean:.2f}',f'[{r.low:.2f}, {r.high:.2f}]',str(r.n_sites)])
 chunks.append(table(r'BiTFI physical-unit MAE by environmental-change or measured-irradiance stratum. Units are those of main-text Table~4; RH uses percentage points.','tab:strata',['Variable','Stratum','MAE',r'95\% CI','Sites'],rows,'llrrr'))
 matched=pd.read_csv(A/'matched_availability_summary.csv');mv=pd.read_csv(A/'matched_availability_variables.csv');rows=[]
 for r in matched.itertuples():rows.append(['B' if r.model.endswith('-B') else 'C','All indoor',n(r.mean),ci(r)])
 for r in mv.itertuples():rows.append(['B' if r.model.endswith('-B') else 'C',VL[r.variable],n(r.mean),ci(r)])
 chunks.append(table(r'Matched-window indoor-target NMAE with outdoor measurements retained (B) or hidden (C). Sites, intervals and durations are identical.','tab:matched',['Scenario','Target','NMAE',r'95\% CI'],rows,'llrr'))
 rows=[]
 contrasts={
 ('BiTFI-TimesFM3','BiTFI-TimesFM3-fwd'):'BiTFI: full vs. forward-only',
 ('BiTFI-TimesFM3','BiTFI-Chronos2'):'BiTFI: TimesFM3 vs. Chronos 2',
 ('SAITS-spatial','SAITS-matched-local'):'SAITS: local + cross vs. local',
 ('MOMENT-FT','MOMENT'):'MOMENT: head-tuned vs. zero-shot',
 ('TimesFM3.0-COV','TimesFM3.0'):'TimesFM3: local vs. univariate',
 ('TimesFM3.0-COV-SPA','TimesFM3.0-COV'):'TimesFM3: local + cross vs. local',
 ('BiTFI','LI'):'BiTFI vs. linear interpolation',
 ('BiTFI','Spatial-Ridge'):'BiTFI vs. Spatial ridge',
 ('BiTFI','TimesFM3-univariate'):'BiTFI vs. univariate TimesFM3',
 ('TimesFM3.0-UNI','Chronos2-UNI'):'TimesFM3 vs. Chronos 2',
 ('TimesFM3.0-UNI','TimesFM2.5-UNI'):'TimesFM3 vs. TimesFM2.5'}
 for filename,family in [('component_paired.csv','Component contrasts (6 tests)'),('additional_paired.csv','Additional facilities (3 tests)'),('univariate_paired.csv','Matched univariate backbones (2 tests)')]:
  rows.append([r'\multicolumn{6}{l}{\textit{'+family+'}}'])
  for r in pd.read_csv(A/filename).itertuples():rows.append([contrasts[(r.model,r.reference)],str(r.n_sites),pct(r.reduction),f'[{r.low:.2f}, {r.high:.2f}]',n(r.p),n(r.holm_p)])
 chunks.append(table(r'Paired greenhouse comparisons for the remaining three test families. Positive reductions favor the first configuration; intervals are paired bootstrap 95\% CIs. Two-sided Wilcoxon $p$ values are Holm-adjusted separately within each family.','tab:contrasts',['Comparison','Sites',r'Reduction (\%)',r'95\% CI',r'$p$',r'Holm $p$'],rows,'lrrrrr'))
 equal=pd.read_csv(A/'equal_case_summary.csv').set_index('model');rows=[]
 for m,r in equal.sort_values('mean').iterrows():rows.append([labels[m],n(s.loc[m,'mean']),n(r['mean']),ci(r),str(int(s['mean'].rank().loc[m])),str(int(equal['mean'].rank().loc[m]))])
 chunks.append(table(r'Equal-case sensitivity across all 21 configurations. Variable NMAEs are averaged within each masking case, cases equally within each greenhouse, and greenhouses equally. Primary scores weight evaluated observations within greenhouse; ranks refer to the same 21 configurations.','tab:equalcase',['Configuration','Primary','Equal-case',r'95\% CI','Primary rank','Case rank'],rows,'lrrrrr'))
 chunks.append(r'\clearpage')
 figs=[('FigureS1_Extended_comparison.pdf','Extended comparison of all 21 configurations. Points and intervals show greenhouse scores, means and bootstrap confidence intervals.'),('FigureS2_Context_sensitivity.pdf','Context selection. (A) Univariate training-site validation NMAE. (B) Error relative to each backbone\'s selected minimum.'),('FigureS3_SAITS_training.pdf','SAITS sensors-only training. (A) Training objective. (B) Fixed validation masked MAE; the selected epoch is marked.'),('FigureS4_Seasonal_performance.pdf','Seasonal reconstruction performance. (A) Spring. (B) Summer. (C) Autumn. (D) Winter. Intervals show greenhouse bootstrap uncertainty.'),('FigureS5_Information_sources.pdf','Information-source comparisons. (A) TimesFM3 input configurations and BiTFI. (B) Indoor and outdoor variable groups for Spatial ridge, TimesFM3 and BiTFI.'),('FigureS6_Univariate_backbone_comparison.pdf','Backbone comparisons. (A) Matched single-call univariate accuracy. (B) Gap duration. (C) Sensor variables. (D) Full BiTFI with Chronos 2 or TimesFM3.'),('FigureS7_MOMENT_head_tuning.pdf','MOMENT reconstruction-head adaptation. (A) Validation learning-rate search. (B) Paired test-greenhouse scores. (C) Gap duration. (D) Sensor variables.'),('FigureS8_Refinement_ablation.pdf','Refinement depth. (A) Training-site validation. (B) Test sensitivity. (C) Step 1 and selected-depth errors by variable. (D) A6000 cumulative time per mask.'),('FigureS9_Information_and_additional_sites.pdf','Additional comparisons. (A) SAITS input configurations. (B) Common indoor-variable reconstruction at nine additional sites.'),('FigureS10_Environmental_transitions.pdf','Physical-unit error by environmental stratum. (A--E) Lower and higher one-hour changes for indoor temperature, outdoor temperature, RH, CO$_2$ and radiation. (F) Zero and positive measured irradiance. Error bars show greenhouse bootstrap confidence intervals.')]
 for i,(file,caption) in enumerate(figs,1):chunks.append(figure(file,caption,'fig:s'+str(i)).replace('figure*','figure').replace('.78\\textheight','.83\\textheight')+'\n\\clearpage')
 chunks.append(r'\end{document}');text='\n\n'.join(chunks);(P/'supplementary.tex').write_text(text);print('Wrote',P/'supplementary.tex')
if __name__=='__main__':main()
