"""Insert executed revision results; never generate experimental numbers in prose."""
from pathlib import Path
import re,json
import numpy as np,pandas as pd
from revision_config import MAIN_MODELS,LABELS
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'03_result/revision_experiments_20260926';A=R/'analysis';T=ROOT/'05_thesis';D=ROOT/'04_figure/source_data'
from figures.prism_style import LABELS as ALL_LABELS
labels=ALL_LABELS|LABELS

def table(caption,label,heads,rows,cols=None):
 cols=cols or ('l'+'r'*(len(heads)-1));body='\n'.join(' & '.join(row)+r' \\' for row in rows)
 return '\n'.join([r'\begin{table}[p]',r'\centering\fontfamily{ptm}\selectfont\small',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\setlength{\tabcolsep}{6pt}',r'\begin{tabular}{@{}'+cols+'@{}}',r'\toprule',' & '.join(heads)+r' \\',r'\midrule',body,r'\bottomrule',r'\end{tabular}',r'\end{table}'])
def replace_table(s,label,new):
 pattern=r'\\begin\{table\*?\}(?:(?!\\end\{table).)*?\\label\{'+re.escape(label)+r'\}.*?\\end\{table\*?\}'
 s,n=re.subn(pattern,lambda m:new,s,flags=re.S);assert n==1,(label,n);return s
def replace_subsection(s,title,new):
 a=s.index('\\subsection{'+title+'}');b=s.find('\\subsection{',a+1)
 if b<0:b=s.index('\\section{Conclusions}',a)
 return s[:a]+new+'\n\n'+s[b:]
def f(x):return f'{x:.4f}'
def ci(r):return '['+f(r.low)+', '+f(r.high)+']'
def read(n):return pd.read_csv(A/(n+'.csv'))
ref=read('refinement_summary').set_index('model');rp=read('refinement_paired').set_index('reference');sa=read('saits_summary').set_index('model');sap=read('saits_paired').iloc[0];ad=read('additional_summary').set_index('model');addp=read('additional_paired').set_index('reference');ag=read('agronomic_summary').set_index(['model','metric']);ind=read('indoor_scenarios_common_sites').set_index('scenario');allr=read('all_summary').set_index('model');var=read('all_variables');seas=read('all_seasons');tests=pd.read_csv(D/'paired_greenhouse_tests.csv');ms=pd.read_csv(D/'main_summary.csv').set_index('model')
s=(T/'TFM.tex').read_text();su=(T/'supplementary.tex').read_text()
assert r'\label{tab:refinement}' not in su, 'Result insert already present; use the pre-insertion source for regeneration.'
# Updated Holm family; point estimates and bootstrap intervals of existing models are unchanged.
pv=tests.set_index('reference').loc['TimesFM3.0-COV-SPA','holm_p'];s=re.sub(r'\\newcommand\{\\MainHolmP\}\{[^}]*\}',lambda m:r'\newcommand{\MainHolmP}{'+f(pv)+'}',s);su=re.sub(r'\\newcommand\{\\MainHolmP\}\{[^}]*\}',lambda m:r'\newcommand{\MainHolmP}{'+f(pv)+'}',su)
# Concise, outcome-focused abstract.
abstract=r'''Missing greenhouse sensor observations interrupt the environmental records used for crop-climate analysis. We present bidirectional time-series foundation-model imputation (BiTFI), which initializes a gap from its two observed boundaries and refines the reconstruction using local sensors, calendar features and correlated training-greenhouse records, with a frozen forecasting backbone. Hourly records from 34 commercial Korean greenhouses were divided into 24 training and 10 evaluation sites. Across 21 configurations and 3,357 artificial masking cases covering 5 variables, 3 sensor-failure patterns and durations of 6--168~h, BiTFI with TimesFM3 achieved mean greenhouse NMAE \BiTFINMAE{} (95\% CI \BiTFICILow{}--\BiTFICIHigh{}), \CovReduction{}\% below covariate-enabled TimesFM3. Covariate refinement reduced the Step~1-only error from @@STEP@@ to \BiTFINMAE{}; additional passes yielded smaller further improvements. Adding reference-greenhouse channels to a matched SAITS control reduced NMAE from @@LOCAL@@ to @@CROSS@@. At 9 additional sites with incomplete outdoor instrumentation, BiTFI achieved NMAE @@ADD@@ across indoor temperature, RH and CO$_2$, compared with @@ADUNI@@ for univariate TimesFM3. BiTFI reconstructed hourly air VPD with MAE @@VPD@@~kPa during joint temperature--humidity outages. These results support retrospective recovery of greenhouse climate records and derived environmental indicators without site-specific backbone training.'''
for key,value in {'STEP':f(ref.loc['BiTFI-refine0','mean']),'LOCAL':f(sa.loc['SAITS-matched-local','mean']),'CROSS':f(sa.loc['SAITS-spatial','mean']),'ADD':f(ad.loc['BiTFI','mean']),'ADUNI':f(ad.loc['TimesFM3-univariate','mean']),'VPD':f(ag.loc[('BiTFI-refine1','VPD'),'mean'])}.items():abstract=abstract.replace('@@'+key+'@@',value)
s=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda m:'\\begin{abstract}\n'+abstract+'\n\\end{abstract}',s,flags=re.S)
s=s.replace('including the zero-shot MOMENT adaptation control omitted from that plot,', 'including the zero-shot MOMENT adaptation control,')
s=s.replace('among the 19 evaluated settings','among the 21 evaluated settings').replace('show the 9 main configurations','show the 12 main configurations').replace('across the 8 main comparisons','across the 11 main comparisons').replace('all 19 settings','all 21 settings').replace('Among the 8 configurations','Among the 12 configurations').replace('within the 8-model comparison','within the 12-setting comparison').replace('among the 8 settings','among the 12 settings')
rows=[]
for m in sorted(MAIN_MODELS,key=lambda m:allr.loc[m,'mean']):
 r=allr.loc[m];rows.append([labels[m],f(r['mean']),f(r.sd),ci(r)])
s=replace_table(s,'tab:overall',table('Overall reconstruction error for the common 12-setting comparison across 10 test greenhouses. SD is between-greenhouse standard deviation; CI is the 95\\% bootstrap confidence interval.','tab:overall',['Configuration','NMAE','SD','95\\% CI'],rows))
rows=[]
for m in MAIN_MODELS:
 values=[]
 for v in ['Tin','Tout','RH','CO2','Rad']:
  r=var[(var.model==m)&(var.variable==v)].iloc[0];value=f(r['mean']);value=(r'\textbf{'+value+'}') if r['mean']==var[(var.model.isin(MAIN_MODELS))&(var.variable==v)]['mean'].min() else value;values.append(value)
 rows.append([labels[m],*values])
s=replace_table(s,'tab:variable',table('Variable-specific mean greenhouse NMAE for the common 12-setting comparison. Bold values indicate the lowest error in each column; cross denotes cross-greenhouse covariates.','tab:variable',['Model',r'$T_{\mathrm{in}}$',r'$T_{\mathrm{out}}$','RH',r'CO$_2$','Rad'],rows))
# Explicit matched information result, both controls retained in main plots.
needle='These comparisons include differences in available information and context, and do not isolate architecture alone.'
addition=f" The matched SAITS controls obtained NMAEs of {f(sa.loc['SAITS-matched-local','mean'])} (SAITS-local) and {f(sa.loc['SAITS-spatial','mean'])} (SAITS-cross). Reference channels reduced error by {sap.reduction:.2f}\\% (95\\% paired CI {sap.low:.2f}--{sap.high:.2f}\\%; $p={f(sap.p)}$), showing that a trained imputer also benefits from contemporaneous reference-site information (Supplementary Table~S5)."
s=s.replace(needle,needle+addition)
# Common indoor-target and common-site scenario comparison, not a causal treatment effect.
needle='In Scenario~C, neighboring-greenhouse observations and calendar covariates remained available despite the absence of all local target observations inside the gap.'
addition=f" Restricting both scenarios to indoor temperature, RH and CO$_2$ and the same {int(ind.loc['C','n_sites'])} eligible greenhouses gave NMAEs of {f(ind.loc['B','mean'])} for B and {f(ind.loc['C','mean'])} for C. The reversal of the aggregate ordering indicates that target composition and site coverage contribute to the apparent B--C difference. Masked intervals still differ, so this is not an equal-window sensor-removal experiment."
s=s.replace(needle,needle+addition)
needle='\\subsection{Information sources and backbone choice}'
new=r'''\subsection{Step contribution and additional-site performance}
With all other inputs fixed, Step~1-only bidirectional initialization achieved NMAE @@STEP@@. One covariate-refinement pass reduced this to @@ONE@@, a @@GAIN@@\% improvement (95\% paired CI @@LOW@@--@@HIGH@@\%; Holm $p=@@P@@$). Two, 3 and 5 passes achieved @@TWO@@, @@THREE@@ and @@FIVE@@, respectively (Supplementary Fig.~S8; Supplementary Table~S4). Additional passes improved mean error but with diminishing gains. The specified one-pass method is retained; the diagnostic test results are not used to choose a new iteration count.

Across the 9 additional greenhouses, BiTFI achieved NMAE @@ADD@@ (95\% CI @@ADDLOW@@--@@ADDHIGH@@), versus @@UNI@@ for univariate TimesFM3, @@RIDGE@@ for Spatial ridge and @@LI@@ for linear interpolation. Relative to univariate TimesFM3, the reduction was @@ADDGAIN@@\% (95\% paired CI @@AGLOW@@--@@AGHIGH@@\%; Holm $p=@@AP@@$; Supplementary Fig.~S9B; Supplementary Table~S6). These results concern only the 3 indoor variables and a different site set, so their absolute NMAE is not directly comparable with the 5-variable main benchmark.

'''
pr=rp.loc['BiTFI-refine0'];ar=addp.loc['TimesFM3-univariate'];vals={'STEP':f(ref.loc['BiTFI-refine0','mean']),'ONE':f(ref.loc['BiTFI-refine1','mean']),'GAIN':f'{pr.reduction:.2f}','LOW':f'{pr.low:.2f}','HIGH':f'{pr.high:.2f}','P':f(pr.holm_p),'TWO':f(ref.loc['BiTFI-refine2','mean']),'THREE':f(ref.loc['BiTFI-refine3','mean']),'FIVE':f(ref.loc['BiTFI-refine5','mean']),'ADD':f(ad.loc['BiTFI','mean']),'ADDLOW':f(ad.loc['BiTFI','low']),'ADDHIGH':f(ad.loc['BiTFI','high']),'UNI':f(ad.loc['TimesFM3-univariate','mean']),'RIDGE':f(ad.loc['Spatial-Ridge','mean']),'LI':f(ad.loc['LI','mean']),'ADDGAIN':f'{ar.reduction:.2f}','AGLOW':f'{ar.low:.2f}','AGHIGH':f'{ar.high:.2f}','AP':f(ar.holm_p)}
for k,v in vals.items():new=new.replace('@@'+k+'@@',v)
s=s.replace(needle,new+needle)
needle='% END embedded sections/results'
r=ag.loc[('BiTFI-refine1','VPD')];v0=ag.loc[('BiTFI-refine0','VPD'),'mean'];vc=ag.loc[('SAITS-spatial','VPD'),'mean'];new=f"\\subsection{{Agronomic diagnostic performance}}\nBiTFI's physical-unit MAEs were 1.0061~$^{{\\circ}}$C for indoor temperature, 1.0180~$^{{\\circ}}$C for outdoor temperature, 3.3115 percentage points for RH, 39.7386~ppm for CO$_2$ and 26.6159~W\\,m$^{{-2}}$ for radiation (Supplementary Table~S7). The two temperature errors were similar in physical units despite their different NMAEs.\n\nDuring joint temperature--humidity outages, air-VPD MAE was {f(r['mean'])}~kPa (95\\% CI {f(r.low)}--{f(r.high)}), compared with {f(v0)}~kPa for Step~1 alone and {f(vc)}~kPa for SAITS-cross. Fig.~\\ref{{fig:agronomic}} shows VPD and radiation-integral errors by gap duration. These derived quantities evaluate whether reconstruction preserves environmental combinations and accumulated exposure, beyond individual normalized sensor errors. Radiation-defined illumination strata are reported in Supplementary Fig.~S10.\n\n"
s=s.replace(needle,new+needle)
insert=r'''\begin{figure*}[p]
\centering
\includegraphics[width=\textwidth,height=.70\textheight,keepaspectratio]{Figure7_Agronomic_diagnostics.pdf}
\caption{Reconstruction of agronomic environmental indicators. (A) Hourly air-VPD MAE during joint temperature--humidity outages. (B) Absolute error in gap-integrated outdoor radiation on a logarithmic scale. Points show mean greenhouse errors by gap duration; Step~1 only omits covariate refinement.}
\label{fig:agronomic}
\end{figure*}
'''
s=s.replace('% END embedded sections/results',insert+'\n% END embedded sections/results')
s=s.replace('TimesFM3 uses local and cross-greenhouse covariates; BiTFI uses TimesFM3.', 'Cross denotes cross-greenhouse covariates; BiTFI uses TimesFM3.').replace('TimesFM3 uses local and cross-greenhouse covariates, and BiTFI uses the TimesFM3 backbone.', 'Cross denotes cross-greenhouse covariates; BiTFI uses TimesFM3.')
# Rebuild weak discussion sections around actual experiments and agronomic mechanisms.
disc=r'''\subsection{Fixed-snapshot refinement}
The Step~1-only control shows that bidirectional boundary information does not explain the full improvement: adding covariates reduces NMAE by @@GAIN@@\% under the same temporal context. Freezing the snapshot within each pass makes the reconstruction independent of target-processing order. It also allows missing indoor variables to inform each other through their initialized trajectories without propagating an earlier target's newly refined values within that pass.

Repeated synchronous passes give a smaller additional benefit. Five passes achieve NMAE @@FIVE@@ compared with @@ONE@@ for one pass, at approximately @@COST@@ times the measured directional-inference and covariate-construction time on the same GPU. The one-pass setting provides most of the observed improvement with fewer calls; it is a computational compromise, not a claim that one pass minimizes error. The experiment tests refinement depth while holding the synchronous update rule fixed. It does not isolate synchronous versus sequential updating, and iteration count would require validation-based selection before adoption in a new protocol.
'''
time0=json.loads((R/'refinement/shard0/complete.json').read_text())['seconds_per_stage'];cost=sum(float(x) for x in time0.values())/(float(time0['0'])+float(time0['1']));
for k,v in {'GAIN':f'{pr.reduction:.2f}','FIVE':f(ref.loc['BiTFI-refine5','mean']),'ONE':f(ref.loc['BiTFI-refine1','mean']),'COST':f'{cost:.2f}'}.items():disc=disc.replace('@@'+k+'@@',v)
s=replace_subsection(s,'Fixed-snapshot refinement',disc)
needle='Its independent-channel implementation and 512-h window differ from the inputs provided to BiTFI, so the residual gap cannot be assigned to pretraining quality alone.'
s=s.replace(needle,needle+' SAITS jointly models its local sensor channels; it is not channel-independent. The matched SAITS-local/SAITS-cross experiment demonstrates a reference-information benefit within the same trained architecture. BiTFI still has lower error, but its 1,900-h directional contexts differ from the 512-h SAITS window, so this is not an equal-context architecture ranking.')
new=r'''\subsection{Agronomic interpretation of variable and outage effects}
Humidity combines temperature-dependent saturation with water-vapour inputs and removal. Transpiration, ventilation and condensation can alter RH on different time scales; spatial gradients near leaves can also differ from the sensor's bulk-air measurement. Greenhouse experiments show that ventilation affects both bulk-air humidity and the leaf boundary layer~\citep{boulard2004humidity}. These processes provide plausible explanations for the relatively high RH reconstruction error, especially when humidity and temperature fail together. The present records do not include crop leaf area, transpiration or actuator states, so they cannot identify which process caused a particular error. RH MAE in percentage points and the derived VPD error help interpret its agronomic magnitude without treating NMAE as a direct measure of plant stress.

Outdoor temperature and radiation are influenced by weather shared across facilities, which provides a plausible basis for useful cross-greenhouse reconstruction. Indoor variables additionally reflect enclosure properties and local climate management. The similar physical-unit indoor and outdoor temperature MAEs show why the lower outdoor-temperature NMAE alone cannot establish an easier physical prediction problem: each error is divided by its own greenhouse-variable range. For CO$_2$, assimilation and enrichment can interact with ventilation, producing changes that are not fully described by other facilities. Experimental simultaneous temperature and CO$_2$ control illustrates this coupling~\citep{linker1999climate}. The usefulness of reference series should be understood as access to correlated environmental trajectories, rather than evidence of identical management across sites.

Simultaneous loss of indoor sensors removes local information about coupled heat, moisture and carbon dynamics. Under Scenario~B, outdoor temperature and radiation still describe external forcing, while the indoor trajectories must be reconstructed. Under Scenario~C, reference-greenhouse records and calendar features still constrain broad weather and diurnal patterns. This can sustain useful reconstructions even when all local targets are absent. The B--C aggregate ordering also changes when only common indoor variables and common sites are scored; it should not be interpreted as a benefit of losing more sensors. A matched-window experiment with actuator records would be needed to separate information loss from weather and management conditions.

\subsection{From reconstructed records to agricultural use}
The VPD diagnostic connects joint temperature--humidity reconstruction to the atmospheric demand relevant to greenhouse water relations. Lower sensor errors do not automatically imply a lower error in this nonlinear combination, so evaluating VPD directly is informative. Here, BiTFI reduces VPD error relative to initialization alone and the matched trained controls. Air VPD remains distinct from the leaf-to-air vapour-pressure difference because leaf temperature was not measured. The diagnostic does not validate transpiration, irrigation decisions or disease risk.

Gap-integrated outdoor radiation provides a complementary accumulated-exposure measure. Errors can cancel or accumulate across an outage, so its behavior differs from pointwise radiation MAE. Outdoor broadband radiation cannot be equated with canopy PAR or daily light integral without information on greenhouse transmission, shading and spectral conversion. The results support reconstructing climate histories for retrospective analysis; a subsequent crop-model study should test whether these improvements translate into better calibration or independent crop-state estimates~\citep{Ariesen-Verschuur-2022,Hemming-2020}.

For reuse, reconstructed values should retain an explicit missingness mask, method version and input-availability description, and remain distinguishable from measurements. Backward reconstruction uses post-gap information and is suitable for retrospective climate analysis rather than real-time control. Prospective control, shared target/reference outages and crop-response validation require separate experiments. Greenhouse bootstrap intervals describe uncertainty in aggregate error; probabilistic imputation would be needed to assess uncertainty at individual reconstructed time points~\citep{tashiro2021csdi}.
'''
s=replace_subsection(s,'From reconstructed records to agricultural use',new)
# Highlights: use experimental contribution instead of secondary MOMENT detail.
s=s.replace('\\item Fine-tuning the MOMENT head improves NMAE by 9.80\\% across 10 test sites.',f'\\item Covariate refinement reduces initialization error by {pr.reduction:.2f}\\%.')
# Supplementary main inventories share the same model names.
rows=[[labels[m],f(r['mean']),f(r.sd)] for m,r in allr.sort_values('mean').iterrows()]
su=replace_table(su,'tab:all',table('Mean greenhouse NMAE and between-greenhouse SD for all 21 comparison settings, ordered by NMAE. Step-depth diagnostics are listed separately in Table~S4.','tab:all',['Configuration','NMAE','SD'],rows))
rows=[[labels[r.reference],f'{r.reduction_percent:.2f}',f'[{r.CI_low:.2f}, {r.CI_high:.2f}]',f(r.wilcoxon_p),f(r.holm_p)] for r in tests.itertuples()]
su=replace_table(su,'tab:paired',table('Paired comparisons of BiTFI with the 11 references in the common main comparison. Positive reductions favor BiTFI; intervals are 95\\% paired bootstrap confidence intervals.','tab:paired',['Reference',r'Reduction (\%)',r'95\% CI (\%)','$p$','Holm $p$'],rows))
rows=[[labels[m],*[f(seas[(seas.model==m)&(seas.group_value==season)].iloc[0]['mean']) for season in ['spring','summer','fall','winter']]] for m in MAIN_MODELS]
su=replace_table(su,'tab:season',table('Seasonal mean greenhouse NMAE for the common 12-setting comparison.','tab:season',['Configuration','Spring','Summer','Fall','Winter'],rows))
su=su.replace('Extended comparison of 18 model configurations','Extended comparison of 21 model configurations')
# Add experimental protocols and tables after the existing three tables/figures.
extra=r'''\clearpage
\section*{Refinement and matched-information protocols}
Refinement diagnostics use the same 3,357 masks, fixed 1,900-h contexts and frozen TimesFM3 weights. Step~1 supplies the zero-refinement control. Each additional pass reconstructs all targets from the same preceding-pass snapshot; reference selection is fixed within a case and a target's own gap estimate is excluded from its covariates. All 6 calculated stages (initialization and 1--5 passes) passed hidden-target perturbation checks on 15 scenario--duration cases. One-pass predictions matched the production implementation in those checks; the full rerun also reproduced its aggregate score. Four paired contrasts compare the specified one-pass method with 0, 2, 3 and 5 passes using Holm correction.

SAITS-local and SAITS-cross use a common 24-channel architecture: 5 local variables, 4 calendar features and 15 slots for up to 3 references per variable. The local arm masks all reference slots. Both arms use seed~42, 4 fixed corruptions for each of 673 training windows, one fixed corruption for each of 41 validation windows, batch size~16, Adam at $10^{-3}$, at most 60 epochs and patience~10. Training references contain training-prefix observations only and exclude the target site. Masked target observations are excluded from reference ranking. Both observed-reconstruction and masked-imputation losses use only the 5 local targets. The observed-reconstruction term averages the 3 SAITS estimates. Validation masked MAE selects each checkpoint. At test time, both arms use the same 512-h centered windows and masks. The reference-feature inputs passed 32 hidden-target perturbation checks. Their pairwise comparison isolates reference availability within this implementation; the single-seed experiment does not estimate training-seed uncertainty.

The additional-site experiment uses all 9 excluded facilities with indoor temperature, RH and CO$_2$, frozen training-prefix scalers, and unchanged inference settings. Single-sensor and all-indoor-sensor gaps use 6, 12, 24, 72 and 168~h with up to 3 repeats, seed~4242 plus deterministic case offsets, and at least 1,900 preceding positions. There are 528 masks and 792 variable-level cases. No site in this cohort enters model fitting or context selection. Univariate TimesFM3 predicts each complete gap in one call using only its available pre-gap target history, up to 1,900~h. Three contrasts compare BiTFI with each reference. Because the cohort contains only indoor targets and comes from the same source, it is neither the same estimand as the main 5-variable benchmark nor independently collected external validation.
'''
rows=[]
for step in [0,1,2,3,5]:
 r=ref.loc['BiTFI-refine'+str(step)];rows.append(['Step 1 only' if step==0 else ('BiTFI (1 pass)' if step==1 else str(step)+' refinement passes'),f(r['mean']),f(r.sd),ci(r),('--' if step==1 else f(rp.loc['BiTFI-refine'+str(step),'holm_p']))])
extra+='\n'+table('Effect of covariate-refinement depth. Zero passes use Step~1 only; one pass is BiTFI. Holm-adjusted $p$ values compare each alternative with one pass.','tab:refinement',['Setting','NMAE','SD','95\\% CI','Holm $p$'],rows)+'\n\\clearpage\n'
rows=[[labels[m],f(r['mean']),f(r.sd),ci(r)] for m,r in sa.iterrows()]
extra+=table('Matched SAITS information controls. SAITS-cross adds reference-greenhouse channels to the shared local/calendar architecture.','tab:saits_info',['Setting','NMAE','SD','95\\% CI'],rows)+'\n\\clearpage\n'
rows=[[labels.get(m,'TimesFM3 (univariate)' if m=='TimesFM3-univariate' else m),f(r['mean']),f(r.sd),ci(r)] for m,r in ad.sort_values('mean').iterrows()]
extra+=table('Indoor-variable reconstruction at 9 additional greenhouses.','tab:additional',['Setting','NMAE','SD','95\\% CI'],rows)+'\n\\clearpage\n'
phys=read('all_physical_MAE');rows=[]
for m in MAIN_MODELS:
 rows.append([labels[m],*[f(phys[(phys.model==m)&(phys.variable==v)].iloc[0]['mean']) for v in ['Tin','Tout','RH','CO2','Rad']]])
extra+=table('Physical-unit MAE for sensor reconstruction. RH errors are percentage points.','tab:physical',['Setting',r'$T_{\mathrm{in}}$ ($^{\circ}$C)',r'$T_{\mathrm{out}}$ ($^{\circ}$C)','RH (pp)',r'CO$_2$ (ppm)',r'Rad (W\,m$^{-2}$)'],rows)+'\n\\clearpage\n'
fig6=pd.read_csv(D/'figure6_panel_metrics.csv');rows=[]
for panel,g in fig6.groupby('panel'):
 bit=g[g.model=='DAFI-TimesFM3'].iloc[0];best=g[g.model!='DAFI-TimesFM3'].sort_values('MAE').iloc[0];rows.append([panel,bit.variable,f(bit.R2),f(bit.MAE),labels[best.model],f(best.R2),f(best.MAE)])
extra+=table('Panel scores for main-text Fig.~6. Each baseline is selected by the lowest gap-only MAE among Spatial ridge, MOMENT and SAITS; MAE uses the variable\'s physical unit. G and J are unmasked.','tab:example',['Panel','Variable',r'BiTFI $R^2$','BiTFI MAE','Best baseline',r'Baseline $R^2$','Baseline MAE'],rows,cols='llrrlrr')+'\n\\clearpage\n'
for n,name,caption in [(8,'Refinement_ablation','Contribution of refinement. (A) Mean greenhouse NMAE by refinement depth; bars show 95\\% bootstrap intervals. (B) Variable-specific error for Step~1 alone and BiTFI.'),(9,'Information_and_additional_sites','Additional validation experiments. (A) Matched SAITS-local and SAITS-cross controls. (B) Indoor-variable reconstruction at 9 additional greenhouses. Error bars show 95\\% greenhouse bootstrap intervals.'),(10,'Illumination_diagnostics','BiTFI physical-unit MAE under low-light and illuminated conditions defined by observed outdoor radiation. (A) Indoor temperature. (B) Outdoor temperature. (C) RH. (D) CO$_2$. (E) Radiation. Bars show 95\\% greenhouse bootstrap intervals.')]:
 extra+='\n'+r'\begin{figure}[p]'+'\n'+r'\centering'+'\n'+r'\includegraphics[width=\textwidth,height=.70\textheight,keepaspectratio]{FigureS'+str(n)+'_'+name+'.pdf}\n\\caption{'+caption+'}\n\\end{figure}\n\\clearpage\n'
# Checkpoint identity and measured runtime, with explicit timing boundaries.
extra+=r'''\section*{Software, checkpoint identity and runtime}
TimesFM3 uses the official Google Research checkpoint and model card~\citep{google2026timesfm3}. Checkpoint repository identifiers and cached revisions are listed in Table~S9. TimesFM3 runs use TimesFM~3.0.1, PyTorch~2.14.0 with CUDA~13.0 and highest-precision float32 matrix multiplication. The SAITS controls use PyTorch~2.6.0 with CUDA~12.4. Both environments use NumPy~2.1.3 and pandas~2.3.3. The standard environment also contains Transformers~4.57.6, chronos-forecasting~2.2.2 and AutoGluon-TimeSeries~1.5.0. The NVIDIA driver is 580.173.02.

'''
rows=[['TimesFM3',r'\texttt{google/timesfm-3.0-pytorch}',r'\texttt{43046b85ec22}'],['TimesFM2.5',r'\texttt{google/timesfm-2.5-200m-pytorch}',r'\texttt{1d952420fba8}'],['Chronos 2',r'\texttt{amazon/chronos-2}',r'\texttt{29ec3766d36d}'],['MOMENT',r'\texttt{AutonLab/MOMENT-1-large}',r'\texttt{ca58581bc7be}']]
extra+=table('Foundation checkpoint identities. Revision strings identify the local cached snapshots.','tab:checkpoints',['Model','Repository','Revision (prefix)'],rows,cols='lll')+'\n'
t0=json.loads((R/'refinement/shard0/complete.json').read_text());t1=json.loads((R/'refinement/shard1/complete.json').read_text());t2=json.loads((R/'refinement/shard2/complete.json').read_text());sl=json.loads((R/'saits_information/SAITS-matched-local/complete.json').read_text());sc=json.loads((R/'saits_information/SAITS-spatial/complete.json').read_text());ev=json.loads((R/'saits_information/evaluation/complete.json').read_text())
extra+=f"The 3,357-case main BiTFI inference and scoring run took 667.7~s on an RTX PRO 6000 Blackwell GPU, averaging 0.1989~s per mask job with independent-example batching. The refinement diagnostic shards processed {t0['cases']}, {t1['cases']} and {t2['cases']} masks in {t0['elapsed_s']:.1f}, {t1['elapsed_s']:.1f} and {t2['elapsed_s']:.1f}~s on one RTX PRO 6000 and two RTX A6000 GPUs, respectively; these times include all 6 stages and scoring. On the PRO 6000, the synchronized Step~1 plus one-pass inference/covariate time was {float(time0['0'])+float(time0['1']):.1f}~s, compared with {sum(float(v) for v in time0.values()):.1f}~s through 5 passes. SAITS-local and SAITS-cross training took {sl['elapsed_s']:.1f} and {sc['elapsed_s']:.1f}~s on an RTX A6000; evaluating both checkpoints on all masks took {ev['elapsed_s']:.1f}~s. Model loading is excluded. These are batched throughput measurements with distinct timing scopes and hardware, not a standardized latency comparison.\n"
# Preserve old reproducibility section then append new appendices with numeric S4--S9 table order.
su=su.replace('\\end{document}',extra+'\n\\bibliographystyle{elsarticle-harv}\n\\bibliography{cas-refs}\n\\end{document}')
conclusion=r'''\section{Conclusions}
BiTFI combines bidirectional initialization and one synchronous covariate-refinement pass with a frozen forecasting foundation model. Across 3,357 masking cases at 10 evaluation greenhouses, its TimesFM3 configuration achieved NMAE \BiTFINMAE{} and reduced error by \CovReduction{}\% relative to covariate-enabled TimesFM3. The component experiment confirms a contribution from refinement beyond bidirectional initialization; further passes provide smaller additional gains at greater inference cost. Matched SAITS controls show that reference-greenhouse information also benefits a trained imputer. The additional 9-site experiment supports transfer to facilities with incomplete outdoor instrumentation, and VPD/radiation diagnostics connect sensor reconstruction to environmental quantities used in agronomic analysis. These results support retrospective recovery of greenhouse climate records, with independent cohort and crop-response validation remaining necessary for broader agricultural claims.

'''
a=s.index('\\section{Conclusions}');b=s.index('% END embedded sections/discussion',a);s=s[:a]+conclusion+s[b:]
s=s.replace('Radiation-defined illumination strata are reported in Supplementary Fig.~S10.', 'Radiation-defined illumination strata are reported in Supplementary Fig.~S10. Unclipped BiTFI predictions exceeded 100\\% RH in 113 of 49,884 evaluated hourly temperature--humidity pairs (0.23\\%), producing negative calculated VPD; these values remain in the reported errors.')
s=s.replace('The diagnostic does not validate transpiration, irrigation decisions or disease risk.', 'The diagnostic does not validate transpiration, irrigation decisions or disease risk. The small fraction of negative reconstructed VPD also shows that low mean error does not guarantee physical consistency; explicit humidity constraints would be needed before using these estimates in a physically constrained crop model.')
s=s.replace('mask jobs','masking cases');su=su.replace('mask jobs','masking cases')
(T/'TFM.tex').write_text(s);(T/'supplementary.tex').write_text(su)
assert '@@' not in s
print('Updated manuscript and supplementary text from executed results')
