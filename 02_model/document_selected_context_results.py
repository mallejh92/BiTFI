"""Refresh entry points after publishing the validation-selected results."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'03_result/reevaluation_context_20260925'
n=json.loads((RUN/'manuscript_assets/manuscript_assets.json').read_text())['generated_numbers']
import pandas as pd
rows=[]
for p in (RUN/'evaluation').glob('*/complete.json'):
 if p.parent.name.startswith('CAFI') or 'pilot' in p.parent.name:continue
 new=json.loads(p.read_text());old=json.loads((ROOT/'03_result/reevaluation_clean_20260911/evaluation'/p.parent.name/'complete.json').read_text())
 rows.append(dict(model=p.parent.name,previous_NMAE=old['NMAE'],current_NMAE=new['NMAE'],difference=new['NMAE']-old['NMAE'],rerun=not p.parent.is_symlink(),selected_context=new.get('selected_context','unchanged')))
pd.DataFrame(rows).sort_values('current_NMAE').to_csv(RUN/'previous_vs_current.csv',index=False)
text=f'''# Context selection and reevaluation — 2026-09-25

Active result root: `03_result/reevaluation_context_20260925`. Selection source: `03_result/context_validation_20260925`. The prior evaluation remains unchanged in `reevaluation_clean_20260911`.

## Selection

Three frozen backbones were evaluated on 1,393 identical univariate validation gaps from the final 20% of 21 training-greenhouse timelines. Three of the 24 training greenhouses supplied no eligible cases. All five variables and all five gap durations (6, 12, 24, 72, 168 h) are represented. Candidate contexts were 96, 168, 336, 720, 1080, 1440 and 1900 hourly positions. No test greenhouse entered this selection. Input scalers remained fixed to training-prefix values; validation NMAE used each greenhouse-variable training-prefix range.

Chronos 2, TimesFM2.5 and TimesFM3 all selected **1900 h**, the longest tested candidate. This is the minimum over the tested range, not proof of a global optimum or an independent optimum for covariate-assisted configurations. The selected length is fixed across each backbone's configurations and both BiTFI directions. Test cases with shorter available contexts use the available positions.

## Reevaluation and results

Nine foundation-model configurations were rerun with selected contexts using the original 3,357 test jobs and 6,085 variable-level cases. The ten remaining published baseline configurations retain their identical inputs, masks and checkpoints and reuse their prior results. No baseline training was repeated. S6 was separately rerun for all three backbones at the same 1900-h maximum using one prediction call per gap. The fixed Fig. 6 example was retained and BiTFI predictions regenerated.

- BiTFI (TimesFM3): NMAE **{n['BiTFINMAE']}**, SD {n['BiTFISD']}, 95% CI {n['BiTFICILow']}–{n['BiTFICIHigh']}.
- TimesFM3 with local and cross-greenhouse covariates: **{n['CovNMAE']}**.
- Relative reduction: **{n['CovReduction']}%**, paired bootstrap 95% CI {n['CovReductionLow']}–{n['CovReductionHigh']}%; Holm-adjusted p={n['MainHolmP']}.
- BiTFI (Chronos 2): **{n['ChronosBiTFI']}**; TimesFM3-based BiTFI reduction {n['BackboneGain']}%, paired p={n['BackboneP']}.
- Chronos 2 has the lowest error in the separate matched-input univariate comparison. This ranking does not carry over to the full BiTFI framework.

All numerical figures, compact tables, result-dependent prose, S2/S6 protocols, GA statistic and portable Overleaf bundle were updated. The author-drawn GA layout and the dataset/framework diagrams were retained. Original values and figures are backed up in `05_thesis/archive/pre_context_reevaluation_20260925` and `04_figure/archive/pre_context_reevaluation_20260925`.

The test greenhouses were consulted during earlier development. Validation-only selection in this rerun does not retroactively make that test set an independent external validation cohort.

## Verification

- Validation: 9,751 input-poisoning checks and 21 batch-equivalence checks passed; all 21 prediction files were independently rescored.
- Rerun configurations: 135 hidden-target poisoning checks and nine ABA order checks passed; original test masks are unchanged.
- S6: 18,255 predictions (6,085 per backbone) independently rescored.
- Fig. 6: all 52 metric pairs independently recomputed.
- Publication tables and claims checked against case-level scores; both standalone TeX files compiled without undefined references or overfull boxes.
- Machine-readable records: `validation_report.json`, `publication_validation.json`, selection `prediction_validation.json`, and S6 `validation_report.json`.

## Scripts

`02_model/run_validation_context_selection.py` prepares and evaluates validation candidates; `run_selected_context_evaluation.py` performs the selected-context test reevaluation; `run_selected_univariate_control.py` runs S6. `render_selected_context_figures.py` regenerates figures and source data, and `build_selected_context_assets.py` regenerates numerical macros and table rows. Manuscripts remain single-file sources in `05_thesis`.
'''
(RUN/'README.md').write_text(text)
(ROOT/'05_thesis/notes/Context_selection_reevaluation_20260925.md').write_text(text)
p=ROOT/'05_thesis/README.md';s=p.read_text();s=s.replace('재평가 프로토콜 식별자는 `clean-20260911`이며 결과 원천은 `03_result/reevaluation_clean_20260911/`과 `04_figure/source_data/`입니다.', '현재 재평가 프로토콜은 `validation-context-20260925`이며 결과 원천은 `03_result/reevaluation_context_20260925/`과 `04_figure/source_data/`입니다. 검증 구간의 단변량 실험에서 세 백본 모두 1,900 h가 선택되었고, 해당 길이로 foundation-model 9개 설정과 S6를 다시 평가했습니다. 상세 기록은 `notes/Context_selection_reevaluation_20260925.md`에 있습니다. `BiTFI_Overleaf.zip`에는 최신 단일 파일 원고·보충자료·참고문헌·PDF 그림이 포함되어 있습니다.');p.write_text(s)
p=ROOT/'04_figure/FIGURE_INDEX.md';s=p.read_text().replace('updated 2026-09-22','updated 2026-09-25');s+='\nActive results: `03_result/reevaluation_context_20260925`. S2 now presents training-site validation selection; S6 uses the selected 1900-h common maximum. All numerical plots and the GA statistic use the reevaluated results.\n';p.write_text(s)
p=ROOT/'프로젝트_현황.md';s=p.read_text();s='# 활성 결과: validation-context-20260925\n\n2026-09-25: 검증 구간 단변량 context 선택 후 foundation-model 9개 설정과 S6 재평가 및 원고·그림 갱신 완료. 세 백본 모두 1,900 h가 선택됨. 최신 설명은 `05_thesis/notes/Context_selection_reevaluation_20260925.md`, 원천은 `03_result/reevaluation_context_20260925`를 사용한다. 아래 기록은 과거 버전이다.\n\n---\n\n'+s;p.write_text(s)
# Keep the caption entry point consistent with the standalone author manuscripts.
parts=['# Current figure captions — 2026-09-25','Canonical captions are embedded in TFM.tex and supplementary.tex.']
for name in ['TFM','supplementary']:
 for line in (ROOT/'05_thesis'/f'{name}.tex').read_text().splitlines():
  if line.startswith('\\caption'):parts.append(line)
(ROOT/'04_figure/Figure_captions.md').write_text('\n\n'.join(parts)+'\n')
print('Updated project notes and current-result entry points.')
