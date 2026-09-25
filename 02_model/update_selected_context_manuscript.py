"""Preserve author prose and compact tables while updating evaluated quantities."""
from pathlib import Path
import re,json,shutil
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'03_result/reevaluation_context_20260925';ASSETS=RUN/'manuscript_assets';OUT=RUN/'manuscript';OUT.mkdir(exist_ok=True)
nums=json.loads((ASSETS/'manuscript_assets.json').read_text())['generated_numbers']
for name in ['TFM.tex','supplementary.tex']:
 s=(ROOT/'05_thesis'/name).read_text()
 for key,value in nums.items():
  s=re.sub(r'(\\newcommand\{\\'+key+r'\}\{)[^}]+\}',lambda m:m[1]+value+'}',s)
 for table in ['overall','variables','paired_tests','all_configurations','seasons']:
  mark='% BEGIN embedded tables/'+table
  if mark not in s:continue
  start=s.index(mark);end=s.index('% END embedded tables/'+table,start)
  block=s[start:end];generated=(ASSETS/(table+'.tex')).read_text()
  rows=generated.split(r'\midrule',1)[1].split(r'\bottomrule',1)[0]
  block=re.sub(r'(\\midrule).*?(\\bottomrule)',lambda m:m[1]+rows+m[2],block,flags=re.S)
  s=s[:start]+block+s[end:]
 if name=='TFM.tex':
  s=s.replace('BiTFI reduces NMAE by 13.46\\% versus',r'BiTFI reduces NMAE by \CovReduction{}\% versus')
  s=s.replace('A context length of up to 1,440~h is used for each available direction.', 'A validation-selected context length of up to 1,900~h is used for each available direction.')
  a=s.index('Context lengths were set in earlier exploratory experiments');b=s.index('\\subsection{Metrics,',a)
  s=s[:a]+r'''Context length was selected independently for each backbone using univariate reconstruction on the final 20\% of the training-greenhouse timelines (Supplementary Fig.~S2). All backbones used the same 1,393 single-variable validation gaps across 5 environmental variables and gap lengths of 6, 12, 24, 72 and 168~h. Eligible cases were available in 21 of the 24 training greenhouses. Candidate lengths were 96, 168, 336, 720, 1,080, 1,440 and 1,900~h. The selection criterion was the mean greenhouse NMAE, with evaluated-hour weighting within each greenhouse and equal weighting across greenhouses. All 3 backbones selected 1,900~h. The selected length was then fixed across each backbone's univariate, multivariate, covariate-assisted and BiTFI configurations, with the same length used in each BiTFI direction. This design keeps context length fixed when comparing information sources within a backbone; it does not independently optimize each configuration. When less history is available in a test case, only the available positions are used. Test greenhouses were not used in this selection.

'''+s[b:]
  a=s.index('Supplementary Fig.~S2 shows model-specific context sensitivity.');b=s.index('\n\n',a)
  s=s[:a]+r'''In the validation context experiment, all 3 backbones achieved their lowest mean univariate NMAE at 1,900~h among the 7 tested candidates (Supplementary Fig.~S2). The selected validation NMAEs were \ValidationChronos{}, \ValidationTFMtwo{} and \ValidationTFMthree{} for Chronos~2, TimesFM2.5 and TimesFM3, respectively. Selection was completed before reevaluating the test configurations. Because 1,900~h was the longest candidate, the experiment does not establish whether still longer contexts would improve performance.'''+s[b:]
  a=s.index('The matched-input context experiment (Supplementary Fig.~S2)');b=s.index('\n\n',a)
  s=s[:a]+r'''The validation experiment selected context length using the same univariate task for each backbone, then held that choice fixed across input configurations. This prevents changes in context length from being confounded with the addition of local or cross-greenhouse information within a backbone. It does not establish a separately optimal context for BiTFI or its covariate-assisted comparators. The selected value of 1,900~h was the upper end of the candidate range, and broader searches or configuration-specific selection would require further validation. The previously consulted evaluation greenhouses remain an exploratory test set despite this validation-only context selection.'''+s[b:]
 else:
  a=s.index('Context sensitivity (Supplementary Fig.~S2)');b=s.index('\n\n',a)
  s=s[:a]+r'''Context selection (Supplementary Fig.~S2) uses only the final chronological 20\% of the training-greenhouse records. For each variable and gap length (6, 12, 24, 72 and 168~h), up to 3 nonoverlapping gaps were sampled within each validation timeline using seed~42 and deterministic site--variable--duration offsets. Targets must be genuinely observed, and every case must have 1,900 preceding hourly positions and at least one observed target value within the shortest candidate context. These conditions produced 1,393 cases from 21 training greenhouses. The other 3 training greenhouses supplied no eligible cases. Every backbone and candidate context uses this same manifest. Only pre-gap target history is supplied; local sensors, calendar features, cross-greenhouse records and post-gap target observations are excluded. Context may include the earlier training prefix and already observed validation history. Natural missingness is interpolated and endpoint-filled within the supplied past slice only. Existing scalers fitted to training prefixes remain fixed. Validation MAE is divided by the corresponding greenhouse-variable range in its training prefix, rather than by a range estimated from validation or test targets. Errors are weighted by evaluated hours within each greenhouse and averaged equally across greenhouses. The context with the lowest score is selected independently for each backbone; an exact tie would select the shorter length. The candidates are 96, 168, 336, 720, 1,080, 1,440 and 1,900~h. All backbones selected 1,900~h, which was fixed before the main test reevaluation and applied to all configurations of that backbone. Each validation gap is predicted in one call. Chronos~2 disables cross-series learning; TimesFM3 uses full-precision matrix multiplication and batches equal-length inputs. All 9,751 case-by-context input checks were unchanged after poisoning hidden and post-gap target values, and single-case versus batched inference checks passed for all 21 backbone--context configurations.'''+s[b:]
  a=s.index('\\caption{Matched-input context-length sensitivity');b=s.index('\\label',a) if '\\label' in s[a:s.index('\\end{figure}',a)] else s.index('\\end{figure}',a)
  s=s[:a]+r'''\caption{Validation-based selection of backbone context length. (A) Mean greenhouse NMAE for univariate inference at 7 candidate lengths; open circles mark the selected minima. (B) Percentage increase in validation error relative to each backbone's own minimum. The same 1,393 gaps from the validation tails of 21 training greenhouses are used for every comparison. No test-greenhouse data enter selection. Errors are normalized by greenhouse-variable training-prefix ranges, weighted by evaluated hours within greenhouse, and then averaged equally across greenhouses. All 3 backbones select 1,900~h. Selection is limited to the tested range and is not a separate optimization of covariate-assisted or bidirectional configurations.}
'''+s[b:]
  s=s.replace('Panel~D is not a univariate experiment.', 'Panel~D uses the validation-selected 1,900-h context in each direction and is not a univariate experiment. Panels~A--C retain the fixed 1,440-h context as a separate equal-context control.')
  s=s.replace('For each of the 19 settings, changing hidden raw targets', 'For the 9 foundation-model settings rerun at their selected context and the 10 unchanged comparison settings, changing hidden raw targets')
 # S6 is rerun at the selected common context; recompute every reported comparison.
 import pandas as pd
 uni=pd.read_csv(RUN/'figures/source_data/univariate_backbone_summary.csv').set_index('model')
 tests=pd.read_csv(RUN/'figures/source_data/univariate_backbone_paired_tests.csv').set_index('reference')
 chron=f"{uni.loc['Chronos2-UNI','NMAE']:.4f}";tfm2=f"{uni.loc['TimesFM2.5-UNI','NMAE']:.4f}";tfm3=f"{uni.loc['TimesFM3.0-UNI','NMAE']:.4f}"
 pc=f"{tests.loc['Chronos2-UNI','holm_p']:.4f}";pt=f"{tests.loc['TimesFM2.5-UNI','holm_p']:.4f}"
 if name=='TFM.tex':
  a=s.index('Under identical past-only univariate inputs,');b=s.index('In the separate full BiTFI comparison,',a)
  s=s[:a]+f'Under matched past-only univariate contexts of up to 1,900~h, Chronos~2, TimesFM2.5 and TimesFM3 achieved NMAEs of {chron}, {tfm2} and {tfm3}, respectively (Supplementary Fig.~S6A--C). The Holm-adjusted $p$ values for TimesFM3 compared with Chronos~2 and TimesFM2.5 were {pc} and {pt}, respectively. '+s[b:]
  s=s.replace('The backbone conclusion therefore depends on the full reconstruction configuration, not an assumed universal advantage of TimesFM3.', 'Chronos~2 had the lowest univariate error, whereas TimesFM3 performed better within the full BiTFI framework. Backbone performance thus depended on the reconstruction configuration.')
  s=s.replace('Controlled univariate tests did not establish TimesFM3 superiority over the other backbones, whereas the full BiTFI configuration outperformed its Chronos~2 counterpart.', 'Chronos~2 had the lowest error under matched univariate inputs, whereas TimesFM3 performed better within the full BiTFI framework.')
  s=s.replace('Likewise, the controlled univariate comparison does not establish TimesFM3 as the strongest backbone in general; its advantage over Chronos~2 emerges in the evaluated BiTFI configuration.', 'Chronos~2 performed better under the matched univariate control, whereas TimesFM3 performed better within BiTFI. This reversal shows that univariate ranking alone does not determine the best backbone for bidirectional covariate-assisted reconstruction.')
 else:
  s=s.replace("This controlled comparison differs from the main adapters' context lengths and rolling-horizon implementations.", "This controlled comparison uses a common univariate input and single-call prediction, whereas the main benchmark includes multivariate and rolling-horizon adapters.")
  s=s.replace('identical 1,440-h past-only univariate contexts', 'matched past-only univariate contexts of up to 1,900~h')
  s=s.replace('the same 1,440-h past-only target contexts', 'matched past-only target contexts of up to 1,900~h')
  s=s.replace('The 2 paired comparisons of TimesFM3 against the other univariate models both have Holm-adjusted $p=0.1289$.',f'The Holm-adjusted $p$ values for TimesFM3 compared with Chronos~2 and TimesFM2.5 are {pc} and {pt}, respectively.')
  s=s.replace('Panels~A--C retain the fixed 1,440-h context as a separate equal-context control.', 'Panels~A--C use the common validation-selected length, with shorter available histories used identically across backbones.')
 # Additional validation macros, evaluated directly from the new selection source.
 import pandas as pd
 val=pd.read_csv(ROOT/'03_result/context_validation_20260925/summary.csv').query('context==1900').set_index('model')
 macros=''.join('\\newcommand{\\'+key+'}{'+f"{val.loc[model,'mean']:.4f}"+'}\n' for key,model in [('ValidationChronos','Chronos2'),('ValidationTFMtwo','TimesFM2.5'),('ValidationTFMthree','TimesFM3.0')])
 s=s.replace(r'\begin{document}',macros+r'\begin{document}',1)
 (OUT/name).write_text(s)
for p in (ROOT/'05_thesis').iterdir():
 if p.suffix in ['.bib','.cls','.bst']:shutil.copy2(p,OUT/p.name)
for p in (RUN/'figures').glob('Figure*.pdf'):shutil.copy2(p,OUT/p.name)
for p in (RUN/'figures/Supplementary').glob('Figure*.pdf'):shutil.copy2(p,OUT/p.name)
print('Staged manuscript',OUT)
