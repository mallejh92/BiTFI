"""Generate the revised main text from this run's verified case statistics."""
import json,re,shutil
from pathlib import Path
import numpy as np,pandas as pd
import clean_protocol as cp
from revision_config import MAIN_MODELS,LABELS
R=cp.OUT;A=R/'analysis';P=R/'publication';P.mkdir(exist_ok=True)
BASE=cp.ROOT/'05_thesis/archive/before_hourly_revision_20260927/TFM.tex'
VARS=cp.COLS;SHORT={'LI':'Linear interpolation','SeasonalNaive':'Seasonal naive','AG-LightGBM':'LightGBM','Spatial-Ridge':'Spatial ridge','SAITS-spatial':'SAITS','MOMENT-FT':'MOMENT','TimesFM3.0-COV-SPA':'TimesFM3','BiTFI-TimesFM3':'BiTFI'}
def between(text,start,end,replacement):
 a=text.index(start);b=text.index(end,a);return text[:a]+replacement+'\n\n'+text[b:]
def table(caption,label,headers,rows,align):
 return '\n'.join([r'\begin{table}[p]',r'\centering\small',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\setlength{\tabcolsep}{5pt}',r'\begin{tabular}{@{}'+align+r'@{}}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule',*[' & '.join(row)+r' \\' for row in rows],r'\bottomrule',r'\end{tabular}',r'\end{table}'])
def figure(file,caption,label):
 return '\n'.join([r'\begin{figure*}[p]',r'\centering',r'\includegraphics[width=\textwidth,height=.78\textheight,keepaspectratio]{'+file+'}',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\end{figure*}'])
def n(x):return f'{x:.4f}'
def ci(r):return f"[{r.low:.4f}, {r.high:.4f}]"
def pct(x):return f'{x:.2f}'
def pp(x):return '<0.0001' if x<.0001 else f'={x:.4f}'

def main():
 s=pd.read_csv(A/'all_summary.csv').set_index('model');v=pd.read_csv(A/'all_variables.csv').set_index(['model','variable']);ma=pd.read_csv(A/'all_physical_MAE.csv').set_index(['model','variable']);g=pd.read_csv(A/'all_gaps.csv').set_index(['model','scenario','gap_length_h']);sea=pd.read_csv(A/'all_seasons.csv').set_index(['model','group_value']);paired=pd.read_csv(A/'main_paired.csv').set_index('reference');comp=pd.read_csv(A/'component_paired.csv').set_index(['model','reference']);step=pd.read_csv(A/'refinement_summary.csv').set_index('model');sp=pd.read_csv(A/'refinement_paired.csv').set_index('reference');add=pd.read_csv(A/'additional_summary.csv').set_index('model');ap=pd.read_csv(A/'additional_paired.csv').set_index('reference');matched=pd.read_csv(A/'matched_availability_summary.csv').set_index('model');mp=pd.read_csv(A/'matched_availability_paired.csv').iloc[0];rs=pd.read_csv(A/'residual_strata_MAE.csv').set_index(['model','variable','stratum']);qc=pd.read_csv(A/'constant_exclusion_summary.csv').set_index('model');flags=pd.read_csv(A/'constant_case_flags.csv')
 sensitivity=json.loads((A/'reporting_sensitivity.json').read_text());equal=pd.read_csv(A/'equal_case_summary.csv').set_index('model');equal_pair=pd.read_csv(A/'equal_case_paired.csv').iloc[0];scenario_depth=pd.read_csv(A/'refinement_scenarios.csv').set_index(['model','scenario'])
 bm='BiTFI-TimesFM3';b=s.loc[bm];gain=paired.loc['TimesFM3.0-COV-SPA'];depth=json.loads((R/'refinement_validation/selection.json').read_text())['selected_refinements'];vp=json.loads((R/'refinement_validation/protocol.json').read_text());cx=json.loads((R/'context_validation/protocol.json').read_text());manifest=pd.read_csv(R/'mask_manifest.csv');q=json.loads((R/'quality/protocol.json').read_text());timing=json.loads((A/'a6000_timing.json').read_text());times=timing['seconds_per_mask_by_refinements'];selected_contexts=json.loads((R/'context_validation/selected_contexts.json').read_text());assert set(selected_contexts.values())=={1900}
 bf=lambda var:n(v.loc[(bm,var),'mean']);mf=lambda var:f"{ma.loc[(bm,var),'mean']:.2f}"
 train=json.loads((R/'models/SAITS/training_protocol.json').read_text());additional_cases=pd.read_csv(R/'additional_sites/mask_manifest.csv');uni=pd.concat([pd.read_csv(R/f'univariate_backbone_comparison/{m}_results.csv') for m in ['Chronos2','TimesFM2.5','TimesFM3.0']]);from analyze_revision_experiments import summary
 us=summary(uni).set_index('model');c=comp.loc[(bm,'BiTFI-TimesFM3-fwd')];saits=comp.loc[('SAITS-spatial','SAITS-matched-local')];moment=comp.loc[('MOMENT-FT','MOMENT')];backbone=comp.loc[(bm,'BiTFI-Chronos2')]
 assert b['mean']==s['mean'].min(),'Rewrite outcome phrasing if BiTFI is not the lowest-error tested setting'
 doc=BASE.read_text();doc=re.sub(r'% BEGIN embedded tables/numbers.*?% END embedded tables/numbers','',doc,flags=re.S);doc=re.sub(r'\\newcommand\{\\Validation\w+\}\{[^}]+\}\n','',doc)
 abstract=rf'''Missing greenhouse sensor observations interrupt records used for crop-climate analysis. We present bidirectional time-series foundation-model imputation (BiTFI), which initializes a gap from its two observed boundaries and refines the reconstruction using local sensors, calendar features and reference greenhouses while keeping a forecasting backbone frozen. Hourly records from 34 commercial Korean greenhouses were divided into 24 training and 10 evaluation sites. Across 21 configurations and gaps of 6--168~h, BiTFI with TimesFM3 achieved mean greenhouse normalized mean absolute error (NMAE) {n(b['mean'])} (95\% confidence interval {n(b.low)}--{n(b.high)}), {pct(gain.reduction)}\% below local-plus-cross-greenhouse TimesFM3. Covariate refinement reduced the error of bidirectional initialization from {n(step.loc['BiTFI-refine0','mean'])} to {n(b['mean'])}; {depth} passes were selected using training-site validation. RH MAE increased from {rs.loc[(bm,'RH','change_low'),'mean']:.2f} to {rs.loc[(bm,'RH','change_high'),'mean']:.2f} percentage points between lower- and higher-change hours. At nine additional sites, BiTFI achieved NMAE {n(add.loc['BiTFI','mean'])} across indoor temperature, relative humidity and CO$_2$, compared with {n(add.loc['TimesFM3-univariate','mean'])} for univariate TimesFM3. These results support retrospective climate-record reconstruction under the evaluated information-availability conditions without site-specific backbone fine-tuning.'''
 doc=between(doc,r'\begin{abstract}',r'\end{abstract}',r'\begin{abstract}'+'\n'+abstract)
 highlights=['BiTFI combines bidirectional initialization with covariate refinement.',f'BiTFI reduces NMAE by {pct(gain.reduction)}% versus covariate-enabled TimesFM3.','Context length and refinement depth are selected using training-site validation.','Variable-specific errors relate to measured environmental transitions.','The method reconstructs historical gaps using available post-gap observations.'];assert all(len(x)<=85 for x in highlights)
 doc=between(doc,'% BEGIN embedded sections/highlights','% END embedded sections/highlights','\n'.join(r'\item '+x.replace('%',r'\%') for x in highlights))
 doc=between(doc,r'\section{Introduction}',r'\section{Materials and methods}',(R/'publication_sections/introduction.tex').read_text())
 start='Hourly environmental records from commercial greenhouses';end='The 5 targets were'
 population=r'''Hourly environmental records from commercial greenhouses in South Korea were obtained from Smart Farm Korea (\url{https://www.smartfarmkorea.net/}). Recording spans varied among sites and covered August 2024 to August 2025. Among 43 candidate sites, 20 met the test-site eligibility criteria: all 5 target variables, some observations in both June--August and December--February, and columns for outdoor wind speed and indoor radiation. Ten were selected using seed~42. The two auxiliary columns were eligibility criteria and were not predictors in the five-variable benchmark. Their presence and the seasonal criterion do not imply complete observations in every season. Of the 33 remaining sites, 24 contained all 5 targets and formed the training pool; the other 9 were evaluated separately on common indoor variables. The main cohort comprised 13 tomato, 10 strawberry, 9 sweet-pepper and 2 cucumber greenhouses; cucumber occurred only in training. Table~\ref{tab:greenhouse_list} lists crops, areas and recording spans in the order used in Fig.~\ref{fig:data_overview}.'''
 doc=between(doc,start,end,population)
 doc=doc.replace('Naturally missing entries and values excluded by physical screening remain missing when evaluation intervals are selected.','Timestamps are sorted and reindexed to a complete hourly grid between each site\'s first and last records, retaining their reported local time. Absent timestamps are inserted as missing rows; duplicate or off-hour timestamps are rejected. Every adjacent position therefore represents one elapsed hour. Naturally missing entries and values excluded by physical screening remain missing when evaluation intervals are selected.')
 doc=doc.replace("first 80\\% of each training greenhouse's timeline","first 80\\% of each training greenhouse's hourly recording span")
 doc=between(doc,r'Fig.~\ref{fig:data_overview} provides descriptive summaries','% BEGIN embedded sections/figure1',r'''Fig.~\ref{fig:data_overview} uses the same hourly, physically screened data as evaluation, without short-gap filling. Distributions use at most 3,000 genuine observations per greenhouse and variable; correlations and missingness summaries weight greenhouses equally. Missingness is the fraction of hourly positions without a valid observation between a site's first and last timestamps.''')
 old=re.search(r'The resulting common manifest contains .*?Original natural gaps are never used as scored truth\.',doc).group();hours=int(pd.read_csv(A/'scenario_duration_weights.csv').evaluated_hours.sum())
 doc=doc.replace(old,rf'The common manifest contains {len(manifest):,} imputation jobs and {q["variable_cases"]:,} variable-level cases, comprising {hours:,} evaluated variable-hour occurrences. Counts and within-greenhouse weighting by scenario and duration are reported in Supplementary Table~S10. Repeated intervals across settings are not independent observations, and natural gaps are never scored as truth.')
 doc=doc.replace('Calendar features are the sine and cosine of hour of day divided by 24, and of day of year divided by 365.25, each multiplied by $2\\pi$.',r'For hour of day $u$ and day of year $d$, calendar features are $\sin(2\pi u/24)$, $\cos(2\pi u/24)$, $\sin(2\pi d/365.25)$ and $\cos(2\pi d/365.25)$.')
 doc=doc.replace('Candidates must cover at least 80\\% of the target timeline;', 'Candidate reference series must contain observed values for at least 80\\% of the target hourly positions;')
 doc=doc.replace('Step~2 uses 5 refinement passes',f'Step~2 uses {depth} refinement passes').replace('Step~2 applies 5 covariate-assisted refinement passes',f'Step~2 applies {depth} covariate-assisted refinement passes')
 doc=between(doc,'Computational cost follows directly','% BEGIN embedded Figure2_BiTFI_insert',rf'''Each masked target issues up to 2 directional backbone calls for initialization and 2 per refinement pass, giving at most {2*(1+depth)} calls per variable per gap. Synchronized backbone inference, fusion and refinement-covariate construction averaged {times[str(depth)]:.2f}~s per mask across {timing['cases']:,} test masks on NVIDIA RTX A6000 GPUs (48~GB), using independent-example batches. This measurement excludes model loading, initial mask preparation and scoring. Timing by refinement depth and the scope of each hardware measurement are reported in the Supplementary Material.''')
 doc=between(doc,r'\paragraph{SAITS}',r'\paragraph{MOMENT}',rf'''\paragraph{{SAITS}}
The sensors-only SAITS configuration uses the official architecture~\citep{{du2023saits}}, explicit missingness masks and 512-h gap-centered windows. Training and validation use {train['train_windows']} and {train['validation_windows']} windows, respectively, without crossing a site's chronological 80/20 boundary. Corruption includes pointwise missingness and contiguous A/B/C gaps; fixed validation masked MAE selects the checkpoint. Architecture, optimization and window-eligibility details are provided in the Supplementary Material (Supplementary Fig.~S3; Table~S5).

Two matched controls, SAITS (local) and SAITS (local + cross), use the same 24-channel architecture, initialization, windows and artificial corruptions. Inputs comprise 5 local targets, 4 calendar features and 15 reference slots; the local control leaves reference slots unavailable. Both losses are scored only on the 5 targets. Target sites are excluded from their reference pools, and training references are restricted to training prefixes. This comparison isolates reference availability within SAITS; its comparison with BiTFI also involves context and architectural differences.''')
 doc=between(doc,r'\paragraph{MOMENT}',r'\paragraph{Foundation models}',r'''\paragraph{MOMENT}
MOMENT-large~\citep{goswami2024moment} reconstructs each sensor variable independently in our implementation. We adapt only its 8,200-parameter reconstruction head while retaining this channel-independent input representation and keeping the pretrained encoder frozen in evaluation mode. The same training/validation windows and scalers are used as for SAITS. Validation masked MAE selects the learning rate and epoch; details are given in the Supplementary Material. Inference uses 512-h gap-centered windows. The head-tuned model represents MOMENT in the main comparisons, while the unadapted checkpoint provides the adaptation control (Supplementary Fig.~S7).''')
 doc=doc.replace('1,393 single-variable validation gaps',f'{cx["cases"]:,} single-variable validation gaps').replace('21 of the 24 training greenhouses',f'{len(cx["sites"])} of the 24 training greenhouses')
 doc=doc.replace('When less history is available in a test case, only the available positions are used.','When less history is available in a test case, only the available hourly positions are used. For retrospective forecasting adapters, a missing block without usable history retains the estimate obtained by interpolation from the masked input.')
 doc=between(doc,r'\subsection{Component and additional-site experiments}',r'\subsection{Metrics, aggregation and statistical analysis}',rf'''\subsection{{Component and additional-site experiments}}
Refinement depth was selected using {vp['cases']:,} gaps in the final chronological 20\% of {len(vp['sites'])} training-site records, covering all 3 scenarios, 5 variables and 5 durations. Candidates were 1, 2, 3 and 5 passes; Step~1 alone was a control. Selection minimized mean greenhouse NMAE with evaluated-hour weighting within greenhouse, with ties favoring fewer passes. Scalers remained fixed, scoring used training-prefix ranges, and the target greenhouse was excluded from its reference bank. The selected {depth} passes were applied to BiTFI, its forward-only and Chronos~2 controls, and additional-site evaluation. Test depth sensitivity used the main mask manifest (Supplementary Fig.~S8).

To isolate local sensor availability, Scenario~B was also evaluated on every Scenario~C interval with the same frozen BiTFI settings. The paired analysis scores only indoor temperature, RH and CO$_2$ at identical sites, times and durations (Supplementary Table~S12). This experiment varies whether outdoor temperature and radiation remain observable, while retaining the same reference-selection rules.

Nine additional sites lack at least one outdoor variable but contain the 3 indoor targets. They contribute no data to scaler fitting, global training, or context and depth selection. Single-sensor and simultaneous indoor-sensor gaps of 6--168~h, with up to 3 repeats per condition, produced {len(additional_cases):,} masking cases. BiTFI, univariate TimesFM3, Spatial ridge and linear interpolation use fixed settings. This is an additional cohort from the same provider, not independently collected external validation (Supplementary Table~S6).

OpenAI Codex (GPT-6, OpenAI) assisted development and checking of evaluation, analysis and plotting code. Figures reporting data are generated from case-level outputs by reproducible scripts. Verification includes elapsed-time assertions, hidden-target perturbations, execution-order checks and recomputation of published statistics.''')
 # State each multiplicity family once, matching the saved paired comparisons.
 doc=doc.replace('Longer evaluated gaps contribute more within a greenhouse, but sites with longer records do not receive greater between-site weight.',rf"Longer evaluated gaps contribute more within a greenhouse, but sites with longer records do not receive greater between-site weight. Gaps of 72 and 168~h account for {100*sensitivity['duration_weights']['72']:.1f}\% and {100*sensitivity['duration_weights']['168']:.1f}\% of the overall averaging weight, respectively ({100*sensitivity['long_gap_weight']:.1f}\% combined; Supplementary Table~S10), so the headline score primarily reflects long outages. In a sensitivity analysis, we first average variable-specific NMAEs within each masking case, then give cases equal weight within each greenhouse and greenhouses equal weight overall (Supplementary Table~S14).")
 doc=between(doc,'Percentile 95\\% confidence intervals','Masks are checked',r'''Percentile 95\% confidence intervals use 10,000 greenhouse bootstrap resamples (seed~42). Relative reduction against a reference is $100\times\left(1-\frac{\bar s_{\mathrm{BiTFI}}}{\bar s_{\mathrm{reference}}}\right)$, with paired greenhouse resampling. Two-sided Wilcoxon signed-rank tests use paired greenhouse scores. Holm correction is applied separately across seven main comparisons, six component contrasts, four test-depth contrasts, three additional-site contrasts and two matched-univariate contrasts (Supplementary Tables~S2, S4 and S13). The matched-window B/C analysis is one paired comparison. Individual hours and overlapping gaps are not independent statistical replicates. Intervals condition on fitted checkpoints and do not quantify training-seed uncertainty.''')
 doc=between(doc,'Masks are checked','% END embedded sections/methods',r'''Masks are checked against original observations and scaler extrema are recomputed from training prefixes. Hidden-target perturbation and execution-order checks cover all 15 scenario--duration combinations for the 19 configurations outside the two matched SAITS controls. Those controls additionally check feature invariance after hidden-target perturbation; refinement checks cover each tested pass count. These checks assess implemented access to withheld targets.''')
 insert=rf'''\paragraph{{Descriptive residual and quality analyses}}
Absolute one-hour changes were divided at the variable-specific 75th percentile of observed changes in training prefixes: 1~$^\circ$C for each temperature, 2 percentage points for RH, 35~ppm for CO$_2$ and 85~W\,m$^{{-2}}$ for radiation. Residuals were summarized separately above and at/below these fixed thresholds, using only adjacent genuinely observed hours. Radiation was additionally separated into zero and positive measured irradiance. These are descriptive strata, not identified management events. At least 48 consecutive hourly positions with an identical finite value defined a constant-value flag, interrupted by any missing hour. Primary results retain flagged values; sensitivity analysis excludes affected variable-cases. Reference-availability summaries use the actual 0--3 series selected for each masked target. These analyses were specified without using prediction errors to set their thresholds.

Descriptive residual strata use greenhouse bootstrap intervals without hour-level hypothesis tests.

'''
 doc=doc.replace('% END embedded sections/methods',insert+'% END embedded sections/methods')
 doc=between(doc,'% BEGIN embedded sections/results','\\section*{CRediT authorship contribution statement}','@RESULTS@\n\\FloatBarrier\n@DISCUSSION@\n@CONCLUSION@\n')
 # Save inputs for the final results/prose assembly below.
 context=locals();assemble(doc,context,highlights)


def assemble(doc,z,highlights):
 from types import SimpleNamespace
 d=SimpleNamespace(**z);bm=d.bm
 mu=lambda model:n(d.s.loc[model,'mean'])
 vnum=lambda var:n(d.v.loc[(bm,var),'mean'])
 mnum=lambda var:f"{d.ma.loc[(bm,var),'mean']:.2f}"
 gap=lambda sc,h:n(d.g.loc[(bm,sc,h),'mean'])
 rel=lambda a,b:100*(1-d.s.loc[a,'mean']/d.s.loc[b,'mean'])
 mainrows=[]
 for model,r in d.s.loc[MAIN_MODELS].sort_values('mean',ascending=False).iterrows():
  row=[LABELS.get(model,SHORT[model]),n(r['mean']),n(r.sd),ci(r)]
  if model==bm:row=[r'\textbf{'+x+'}' for x in row]
  mainrows.append(row)
 t3=table(r'Overall reconstruction error for the 8 representative configurations, ordered by decreasing NMAE. SD is between-greenhouse standard deviation; CI is the 95\% bootstrap confidence interval.','tab:overall',['Configuration','NMAE','SD',r'95\% CI'],mainrows,'lrrr')
 minima=d.ma.loc[MAIN_MODELS]['mean'].unstack('variable').min()
 rows=[]
 for model in MAIN_MODELS:
  vals=[]
  for var in VARS:
   value=d.ma.loc[(model,var),'mean'];cell=f'{value:.2f}';vals.append(r'\textbf{'+cell+'}' if value==minima[var] else cell)
  rows.append([SHORT[model],*vals])
 t4=table(r'Physical-unit mean absolute error for the representative configurations. RH error is in percentage points; bold values indicate the lowest mean error in each column.','tab:variable',['Model',r'$T_{\mathrm{in}}$ ($^\circ$C)',r'$T_{\mathrm{out}}$ ($^\circ$C)','RH',r'CO$_2$ (ppm)',r'Rad (W\,m$^{-2}$)'],rows,'lrrrrr')
 f3=figure('Figure3_Overall_performance.pdf',r'Overall reconstruction performance. (A) Mean NMAE; open points represent individual greenhouses. (B) Relative NMAE reduction with BiTFI against each reference; positive values favor BiTFI. Error bars show 95\% bootstrap confidence intervals.','fig:overall')
 f4=figure('Figure4_Gap_robustness.pdf',r'Reconstruction error by gap duration. (A) Single-sensor gaps. (B) Simultaneous gaps in indoor temperature, RH and CO$_2$. (C) Gaps in all 5 variables. Values are mean greenhouse NMAEs.','fig:gap')
 f5=figure('Figure5_Variable_performance.pdf',r'Reconstruction error by variable. (A) Indoor temperature. (B) Outdoor temperature. (C) RH. (D) CO$_2$. (E) Solar radiation. Diamonds and error bars show mean greenhouse NMAE and 95\% bootstrap confidence intervals.','fig:variables')
 sel=json.loads((R/'figure6_common_window/selection.json').read_text());coverage=pd.read_csv(R/'figures/source_data/Figure1_profile/coverage.csv');test=coverage[coverage.role=='test'].reset_index(drop=True);site_number=int(test.index[test.name==sel['greenhouse']][0])+1
 f6=figure('Figure6_Reconstruction_example.pdf',rf'Reconstruction of a 72-h gap at Test~{site_number:02d}. Panels A--E, F--J and K--O show scenarios A, B and C; columns show indoor temperature, outdoor temperature, RH, CO$_2$ and radiation. Panels G and J are unmasked because outdoor temperature and radiation remain observed in Scenario~B. Boxes report gap-only $R^2$ and MAE for BiTFI and the lowest-MAE displayed baseline, selected from Spatial ridge, MOMENT, SAITS and TimesFM3.','fig:examples')
 gap_table=pd.read_csv(A/'all_gaps.csv');winners=gap_table[gap_table.model.isin(MAIN_MODELS)].sort_values('mean').groupby(['scenario','gap_length_h']).first();wins=int((winners.model==bm).sum())
 var_table=pd.read_csv(A/'all_variables.csv');vw=var_table[var_table.model.isin(MAIN_MODELS)].sort_values('mean').groupby('variable').first();varwins=int((vw.model==bm).sum())
 sval=d.step;valdepth=pd.read_csv(R/'refinement_validation/summary.csv').set_index('refinements')
 depth_text=', '.join(f"{k}: {n(sval.loc['BiTFI-refine'+str(k),'mean'])}" for k in [1,2,3,5]);step_gain=100*(1-d.b['mean']/sval.loc['BiTFI-refine0','mean'])
 main=r'\section{Results}'+'\n'+rf'''\subsection{{Overall reconstruction performance}}
BiTFI with TimesFM3 achieved the lowest mean NMAE among the 21 settings: {n(d.b['mean'])} (between-greenhouse SD {n(d.b.sd)}; 95\% CI {n(d.b.low)}--{n(d.b.high)}). The eight representative configurations are summarized in Fig.~\ref{{fig:overall}} and Table~\ref{{tab:overall}}. Relative to local-plus-cross-greenhouse TimesFM3 ({mu('TimesFM3.0-COV-SPA')}), BiTFI reduced NMAE by {pct(d.gain.reduction)}\% (paired bootstrap 95\% CI {pct(d.gain.low)}--{pct(d.gain.high)}\%; Holm $p{pp(d.gain.holm_p)}$ across seven main comparisons).

SAITS (local + cross), LightGBM and head-tuned MOMENT obtained NMAEs of {mu('SAITS-spatial')}, {mu('AG-LightGBM')} and {mu('MOMENT-FT')}, respectively. MOMENT head tuning reduced error from {mu('MOMENT')} to {mu('MOMENT-FT')} ({pct(d.moment.reduction)}\%; Supplementary Fig.~S7). All 21 input and adaptation configurations are retained in Supplementary Table~S1 and Fig.~S1; these rankings do not hold information availability or context constant across architectures. With equal weighting of masking cases, BiTFI remained the lowest-error setting (NMAE {n(d.equal.loc[bm,'mean'])} versus {n(d.equal.loc['TimesFM3.0-COV-SPA','mean'])} for local-plus-cross-greenhouse TimesFM3; {pct(d.equal_pair.reduction)}\% reduction; Supplementary Table~S14), although some other rankings changed.

{t3}
{f3}

\subsection{{Information sources, refinement and backbone controls}}
TimesFM3 NMAE changed from {mu('TimesFM3.0')} with univariate inputs to {mu('TimesFM3.0-COV')} with local covariates and {mu('TimesFM3.0-COV-SPA')} with local plus cross-greenhouse covariates (Supplementary Fig.~S5). Within the matched SAITS architecture, reference channels reduced NMAE from {mu('SAITS-matched-local')} to {mu('SAITS-spatial')}, a {pct(d.saits.reduction)}\% reduction (paired 95\% CI {pct(d.saits.low)}--{pct(d.saits.high)}\%; Supplementary Table~S5).

Training-site validation selected {d.depth} refinement passes. On test sites, Step~1 alone obtained NMAE {n(sval.loc['BiTFI-refine0','mean'])}; the selected reconstruction obtained {n(d.b['mean'])}, a {pct(step_gain)}\% reduction. Test NMAEs by pass count were {depth_text}. Scenario-specific errors and the four depth contrasts are reported in Supplementary Table~S4; five versus three passes did not show a statistically significant difference (Holm $p{pp(d.sp.loc['BiTFI-refine3','holm_p'])}$). These test sensitivity scores were not used for selection (Supplementary Fig.~S8). The forward-only setting obtained {mu('BiTFI-TimesFM3-fwd')}, compared with {n(d.b['mean'])} for full BiTFI (paired reduction {pct(d.c.reduction)}\%, 95\% CI {pct(d.c.low)}--{pct(d.c.high)}\%).

Under matched past-only univariate inputs and single-call complete-gap prediction, Chronos~2, TimesFM2.5 and TimesFM3 obtained NMAEs of {n(d.us.loc['Chronos2-UNI','mean'])}, {n(d.us.loc['TimesFM2.5-UNI','mean'])} and {n(d.us.loc['TimesFM3.0-UNI','mean'])}, respectively (Supplementary Fig.~S6A--C). The component and matched-univariate tests are reported in Supplementary Table~S13. This single-call control differs from the main rolling or multivariate adapters. Within full BiTFI, TimesFM3 obtained {n(d.b['mean'])} and Chronos~2 {mu('BiTFI-Chronos2')} (Supplementary Fig.~S6D).

\subsection{{Gap duration and sensor availability}}
BiTFI had the lowest mean NMAE in {wins} of the 15 scenario--duration combinations among the eight representative configurations (Fig.~\ref{{fig:gap}}). Its NMAE changed from {gap('A',6)} at 6~h to {gap('A',168)} at 168~h in Scenario~A, from {gap('B',6)} to {gap('B',168)} in B, and from {gap('C',6)} to {gap('C',168)} in C. Scenario summaries have different target compositions and eligible time intervals.

When B and C were evaluated on identical Scenario~C intervals and scored only for the three indoor variables, mean NMAE was {n(d.matched.loc['BiTFI-matched-B','mean'])} for B and {n(d.matched.loc['BiTFI-matched-C','mean'])} for C. Retaining the outdoor measurements changed indoor-target error by a relative reduction of {pct(d.mp.reduction)}\% (paired 95\% CI {pct(d.mp.low)}--{pct(d.mp.high)}\%; $p{pp(d.mp.p)}$; Supplementary Table~S12). This paired contrast separates local sensor availability from the different intervals and target sets in the original scenario summaries.

{f4}

\subsection{{Variable-specific, seasonal and illustrative results}}
BiTFI had the lowest mean NMAE for {varwins} of the 5 variables among the representative configurations (Fig.~\ref{{fig:variables}}). Its NMAEs were {vnum('Tin')} for indoor temperature, {vnum('Tout')} for outdoor temperature, {vnum('RH')} for RH, {vnum('CO2')} for CO$_2$ and {vnum('Rad')} for radiation. Physical-unit MAEs were {mnum('Tin')}~$^\circ$C, {mnum('Tout')}~$^\circ$C, {mnum('RH')} percentage points, {mnum('CO2')}~ppm and {mnum('Rad')}~W\,m$^{{-2}}$, respectively (Table~\ref{{tab:variable}}). Variable-specific NMAEs are also tabulated in Supplementary Table~S9.

Seasonal BiTFI NMAE was {n(d.sea.loc[(bm,'spring'),'mean'])} in spring, {n(d.sea.loc[(bm,'summer'),'mean'])} in summer, {n(d.sea.loc[(bm,'fall'),'mean'])} in autumn and {n(d.sea.loc[(bm,'winter'),'mean'])} in winter. These summaries describe the observations available within each season (Supplementary Fig.~S4; Table~S3). Error strata defined by training-derived change thresholds and by zero/positive irradiance are shown in Supplementary Fig.~S10 and Table~S11.

{t4}
{f5}

Fig.~\ref{{fig:examples}} shows the same 72-h interval under all three scenarios, including local-plus-cross-greenhouse TimesFM3. The interval was selected from {sel['eligible_cases']} observation-coverage-qualified cases using the middle entry after sorting by greenhouse identifier and start time, without consulting predictions. Each masked-panel box identifies the lowest-MAE displayed baseline and reports its gap-only $R^2$ and MAE alongside BiTFI; all displayed predictions are included in the axis ranges. Supplementary Table~S8 provides the numerical values. This example illustrates individual trajectories rather than estimating typical performance.

{f6}

\subsection{{Additional facilities and data-quality sensitivity}}
Across nine additional facilities with incomplete outdoor instrumentation, BiTFI achieved NMAE {n(d.add.loc['BiTFI','mean'])} (95\% CI {n(d.add.loc['BiTFI','low'])}--{n(d.add.loc['BiTFI','high'])}), compared with {n(d.add.loc['TimesFM3-univariate','mean'])} for univariate TimesFM3, {n(d.add.loc['Spatial-Ridge','mean'])} for Spatial ridge and {n(d.add.loc['LI','mean'])} for linear interpolation. The reduction relative to univariate TimesFM3 was {pct(d.ap.loc['TimesFM3-univariate','reduction'])}\% (Supplementary Fig.~S9B; Tables~S6 and S13). Only the three indoor targets were evaluated, so these absolute scores are not directly comparable with the five-variable main benchmark.

Excluding the {int((d.flags.constant_hours>0).sum())} variable-cases that overlapped constant-value flags changed BiTFI NMAE from {n(d.b['mean'])} to {n(d.qc.loc[bm,'mean'])}. Sensitivity rankings are provided in Supplementary Table~S7.
'''
 # Discussion statements use physical errors, not assertions of crop tolerance.
 change=lambda var,level:float(d.rs.loc[(bm,var,'change_'+level),'mean'])
 refcounts=pd.read_csv(A/'reference_availability_counts.csv');nz=refcounts.groupby('n_references').variable_cases.sum();reftotal=int(nz.sum());reftext='; '.join(f'{int(k)} references in {int(value):,} variable-cases' for k,value in nz.items())
 rank_before=d.s['mean'].rank();rank_after=d.qc['mean'].reindex(d.s.index).rank();same=bool((rank_before==rank_after).all())
 replacements={
 'MAIN_CONTRAST':rf'The paired main comparison gave a {pct(d.gain.reduction)}\% error reduction, and the direct full-versus-forward-only contrast gave {pct(d.c.reduction)}\%.',
 'STEP_CONTRAST':rf'Bidirectional initialization alone had NMAE {n(d.step.loc["BiTFI-refine0","mean"])}; covariate refinement reduced it to {n(d.b["mean"])}.',
 'COST_CONTRAST':rf'One and {d.depth} passes had test NMAEs of {n(d.step.loc["BiTFI-refine1","mean"])} and {n(d.b["mean"])}, with A6000 inference-and-refinement times of {d.times["1"]:.2f} and {d.times[str(d.depth)]:.2f}~s per mask, respectively.',
 'SCENARIO_DEPTH_CONTRAST':rf"From one to five passes, NMAE stayed at {n(d.scenario_depth.loc[('BiTFI-refine1','A'),'mean'])} in Scenario~A, changed from {n(d.scenario_depth.loc[('BiTFI-refine1','B'),'mean'])} to {n(d.scenario_depth.loc[('BiTFI-refine5','B'),'mean'])} in B, and decreased from {n(d.scenario_depth.loc[('BiTFI-refine1','C'),'mean'])} to {n(d.scenario_depth.loc[('BiTFI-refine5','C'),'mean'])} in C (Supplementary Table~S4), consistent with cross-pass updates among simultaneously missing variables while a single missing target has no changing target-gap covariates. Thus, the additional computation benefits the all-sensor outage most and adds no measured benefit after one pass in A; the five-pass setting is the common validation-selected depth, not an established minimum requirement for every scenario.",
 'TRAINED_CONTRAST':rf'Adding reference channels reduced matched SAITS error by {pct(d.saits.reduction)}\%, while head adaptation reduced MOMENT error by {pct(d.moment.reduction)}\%.',
 'BACKBONE_CONTRAST':rf'The matched univariate test gave NMAEs of {n(d.us.loc["Chronos2-UNI","mean"])} for Chronos~2 and {n(d.us.loc["TimesFM3.0-UNI","mean"])} for TimesFM3; within BiTFI the corresponding values were {mu("BiTFI-Chronos2")} and {n(d.b["mean"])}. These comparisons hold different input rules constant and need not give the same ranking.',
 'GAP_CONTRAST':rf'BiTFI retained the lowest mean error in {wins} of 15 scenario--duration combinations, but its long-gap errors remained variable- and scenario-dependent.',
 'MATCHED_CONTRAST':rf'On the same intervals and indoor targets, retaining outdoor observations produced NMAE {n(d.matched.loc["BiTFI-matched-B","mean"])} versus {n(d.matched.loc["BiTFI-matched-C","mean"])} when they were also hidden.',
 'REFERENCE_CONTRAST':f'All {reftotal:,} scored variable-cases had three qualifying reference series, so performance with fewer references remains untested.' if set(nz.index)=={3} else f'Reference counts varied across {reftotal:,} scored variable-cases ({reftext}).',
 'TEMPERATURE_VALUES':rf'Indoor and outdoor temperature MAEs were {mnum("Tin")} and {mnum("Tout")}~$^\circ$C, respectively.',
 'TEMPERATURE_CHANGES':rf'For indoor temperature, lower- and higher-change hours had MAEs of {change("Tin","low"):.2f} and {change("Tin","high"):.2f}~$^\circ$C; the corresponding outdoor errors were {change("Tout","low"):.2f} and {change("Tout","high"):.2f}~$^\circ$C (Supplementary Fig.~S10).',
 'HUMIDITY_VALUES':rf'RH MAE was {mnum("RH")} percentage points overall, with {change("RH","low"):.2f} and {change("RH","high"):.2f} in lower- and higher-change hours.',
 'CARBON_VALUES':rf'CO$_2$ MAE was {mnum("CO2")}~ppm overall and {change("CO2","low"):.2f} versus {change("CO2","high"):.2f}~ppm in the two change strata.',
 'RADIATION_VALUES':rf'Radiation MAE was {mnum("Rad")}~W\,m$^{{-2}}$ overall. Zero- and positive-irradiance hours had MAEs of {d.rs.loc[(bm,"Rad","radiation_zero"),"mean"]:.2f} and {d.rs.loc[(bm,"Rad","radiation_positive"),"mean"]:.2f}~W\,m$^{{-2}}$, respectively.',
 'SEASON_INTERPRETATION':r'Seasonal error differences may also involve day length, heating--ventilation regimes and crop stage, but recording coverage and site composition differ between seasons. Their aggregate differences cannot identify a seasonal management mechanism.',
 'QUALITY_CONTRAST':rf'Removing constant-flag-overlapping cases changed BiTFI NMAE from {n(d.b["mean"])} to {n(d.qc.loc[bm,"mean"])}; '+('all 21 configuration ranks were retained.' if same else 'the resulting configuration ranks are reported in Supplementary Table~S7.'),
 }
 discussion=(R/'publication_sections/discussion_template.tex').read_text().replace('tab:variable_scores','tab:variable')
 for key,value in replacements.items():discussion=discussion.replace('@'+key+'@',value)
 conclusion=rf'''\section{{Conclusions}}
BiTFI combines bidirectional initialization and validation-selected covariate refinement with a frozen forecasting foundation model for retrospective greenhouse sensor reconstruction. Across {len(d.manifest):,} masking cases at ten test facilities, its TimesFM3 configuration achieved NMAE {n(d.b['mean'])}, {pct(d.gain.reduction)}\% below local-plus-cross-greenhouse TimesFM3. Matched configuration and availability contrasts describe the roles of temporal boundaries and remaining observations, while physical-unit and transition-specific errors distinguish the five environmental variables. Evaluation at nine additional facilities supports application to records with incomplete outdoor instrumentation under the same protocol. Reconstructed values should remain identifiable within climate records, and these sensor-level results require independent-cohort and downstream crop-response validation before broader agricultural benefits can be claimed.'''
 doc=doc.replace('@RESULTS@',main).replace('@DISCUSSION@',discussion).replace('@CONCLUSION@',conclusion)
 doc=re.sub(r'^%.*(?:clean-|Generated by|Standalone Overleaf|embedded).*$','',doc,flags=re.M);doc=re.sub(r'\n{4,}','\n\n\n',doc)
 assert not re.search(r'@[A-Z_]+@',doc)
 commit_file=R/'public_code_commit.json'
 if commit_file.exists():
  commit=json.loads(commit_file.read_text())['commit'][:7];doc=doc.replace('838e275',commit)
 (P/'TFM.tex').write_text(doc);(P/'Highlights.txt').write_text('\n'.join(highlights)+'\n')
 try:
  from docx import Document
  word=Document()
  for text in highlights:word.add_paragraph(text,style='List Bullet')
  word.save(P/'Highlights.docx')
 except ImportError:
  # Minimal editable Word document; no optional authoring dependency required.
  import zipfile
  from xml.sax.saxutils import escape
  paragraphs='<w:p><w:r><w:rPr><w:b/></w:rPr><w:t>Highlights</w:t></w:r></w:p>'
  paragraphs+=''.join('<w:p><w:r><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="22"/></w:rPr><w:t>'+escape('• '+text)+'</w:t></w:r></w:p>' for text in highlights)
  document='<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'+paragraphs+'<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr></w:body></w:document>'
  with zipfile.ZipFile(P/'Highlights.docx','w',zipfile.ZIP_DEFLATED) as archive:
   archive.writestr('[Content_Types].xml','<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
   archive.writestr('_rels/.rels','<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
   archive.writestr('word/document.xml',document)
 (P/'numeric_claims.json').write_text(json.dumps(dict(result_root=str(R.relative_to(cp.ROOT)),mask_cases=len(d.manifest),variable_cases=d.q['variable_cases'],selected_refinements=d.depth,mean_NMAE=float(d.b['mean']),reduction=float(d.gain.reduction),main_models=MAIN_MODELS,fig6_selection=sel),indent=2))
 print('Wrote',P/'TFM.tex',flush=True)

if __name__=='__main__':main()
