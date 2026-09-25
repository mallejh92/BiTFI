"""Promote reviewed clean PDFs, refresh manuscript notes, and package source tables."""
from pathlib import Path
import argparse,json,shutil,zipfile,hashlib
import pandas as pd
import clean_protocol as cp

def main():
 p=argparse.ArgumentParser();p.add_argument('--reviewed',action='store_true',required=True);p.parse_args()
 source=cp.OUT/'figures';dest=cp.ROOT/'04_figure';data=source/'source_data';report=json.loads((source/'validation_report.json').read_text());assert len(report['models'])==23
 pdfs=list(source.glob('Figure*.pdf'))+list((source/'Supplementary').glob('Figure*.pdf'));assert len(pdfs)==14
 assert json.loads((source/'Figure6_export_validation.json').read_text())['all_52_metrics_match_sklearn']
 summaries=pd.read_csv(data/'all_model_summary.csv');main_summary=pd.read_csv(data/'main_summary.csv');mt=json.loads((data/'moment_tuning_statistics.json').read_text());train=json.loads((cp.OUT/'models/SAITS/training_protocol.json').read_text());tune=json.loads((cp.OUT/'models/MOMENT/complete.json').read_text());sel=min(tune['runs'],key=lambda x:x['best_validation_mae'])
 uni=pd.read_csv(data/'univariate_backbone_summary.csv');ut=pd.read_csv(data/'univariate_backbone_paired_tests.csv');bb=pd.read_csv(data/'bitfi_backbone_comparison.csv').iloc[0]
 def fmt(d):
  lines=['| '+' | '.join(map(str,d.columns))+' |','| '+' | '.join(['---']*len(d.columns))+' |']
  for row in d.itertuples(index=False,name=None):lines.append('| '+' | '.join(f'{v:.6f}' if isinstance(v,float) else str(v) for v in row)+' |')
  return '\n'.join(lines)
 methods='''The benchmark evaluates retrospective imputation, permitting post-gap observations and contemporaneous observations from available training-greenhouse sensors. Ten greenhouse sites are excluded from parameter fitting. Normalization uses only the first 80% of each training-greenhouse timeline. Genuine raw observations after physical-range filtering define eligible artificial gaps; no short-gap interpolation precedes masking. All inference-time filling and neighbor selection operate on the current masked input. SAITS and MOMENT are retrained using noncrossing chronological training/validation windows; checkpoint selection uses validation only. AutoGluon models use the same training prefixes and separate validation tails. All configurations share the same 3,357 mask jobs and 6,085 variable-level evaluation cases. Scores are weighted by evaluated hours within greenhouse and equally averaged across greenhouses. Bootstrap uncertainty resamples greenhouses, not individual hours. These test sites were previously consulted for exploratory model/context and illustration choices, so the corrected benchmark is not a newly untouched external validation set.'''
 notes=f'''# 전체 재평가 결과 — 2026-09-11

활성 프로토콜: `clean-20260911`. 원시 관측값에 마스크를 먼저 적용한 뒤 모든 모델을 재평가했다. 23개 설정, 3,357개 공통 마스크, 설정별 6,085개 전체 변수 평가 셀(층화 포함 34,535행). 학습 온실 24개, 테스트 온실 10개. 평가 대상 정답은 총 329,196개 변수·시간 발생 건이며 반복·중첩 구간을 포함한다.

## 누수 수정과 검증

- 마스크 이전 자연 결측 복원값을 BiTFI 입력으로 재사용하는 경로 제거.
- 이웃 온실 선택 캐시를 마스크마다 초기화.
- 짧은 결측을 미리 보간한 값을 정답으로 평가하지 않도록 원시 관측 마스크 재생성.
- 정규화는 학습 온실의 앞 80%만으로 고정. 검증·테스트 정답은 스케일러 학습에서 제외.
- 백엔드 미사용/추론 오류 발생 시 평가 중단. 오류로 대체된 보간 결과를 모델 성능으로 저장하지 않음.
- 모든 3,357개 마스크와 스케일러를 독립 검증. 각 설정에서 A/B/C × 5개 gap의 숨긴 정답 변조 검증 15건과 A–B–A 순서 불변성 검증 수행(총 345건 + 23건).
- 순차/배치 동등성, 마스크 전 기저값 변조 불변성, 백엔드 누락 예외 검증 기록은 결과 폴더의 JSON 참조.

## 본문 비교 결과

낮은 NMAE가 좋다. 아래 값은 온실별 집계 후 평균이다.

{fmt(main_summary[['label','NMAE','CI_low','CI_high']])}

## 전체 확장 비교

S1은 CAFI 4개와 MOMENT zero-shot을 제외한 18개 설정이다. CAFI는 반복 개선 진단에, MOMENT zero-shot은 미세조정 효과 검증에만 별도 보존한다.

{fmt(summaries[['label','NMAE','SD']])}

## MOMENT

같은 원시 관측 기준으로 학습 창 {train['train_windows']}개와 검증 창 {train['validation_windows']}개를 새로 구성했다. 인코더는 고정하고 8,200개 head 파라미터만 학습했다. 선택값은 lr={sel['learning_rate']}, epoch={sel['best_epoch']}. 테스트 NMAE {mt['zero_shot_NMAE']:.6f} → {mt['head_tuned_NMAE']:.6f}, 상대 감소 {mt['relative_reduction_percent']:.2f}% (paired bootstrap 95% CI {mt['CI_low']:.2f}–{mt['CI_high']:.2f}%). 본문·S1·S5·Figure 6의 MOMENT는 모두 이 미세조정 체크포인트다. S8만 zero-shot 대조군을 포함한다.

## 해석 범위

BiTFI의 미래 관측 사용은 사후 결측 복원이라는 정의 아래 허용된다. 실시간 미래 예측 성능으로 해석하면 안 된다. 모델마다 입력 정보가 다르므로 F3의 우열을 동일 정보량에서의 알고리즘 우열로 단정할 수 없다.

이번 수정으로 기존 입력 누수 경로를 제거했지만, 평가 온실을 이미 모델/문맥 길이 개발에 참고했던 이력까지 없어지는 것은 아니다. 논문의 독립 일반화 주장을 더 강하게 하려면 개발에 사용하지 않은 새 온실/기간을 동결해 추가 외부 검증해야 한다. 사전학습 코퍼스 포함 여부와 온실의 물리적 근접·중복 여부는 현재 검사로 확정하지 않았다.

원시 관측만 허용하면서 평가 창도 바뀌었다. 이전과 같은 키·동일 시간창인 마스크는 {report['previous_same_key_window_count']:,}개다. 기존-신규 점수 차이는 누수 수정, 스케일링, 재학습, 평가 창 변경의 복합 결과이며 순수한 누수 크기가 아니다. 동일 시간창 부분집합(9개 온실, 1,609개 마스크, 2,639개 변수 평가 셀)의 BiTFI NMAE는 0.036245 → 0.036127이었다. 이 부분집합도 전처리·스케일링이 바뀌었으므로 누수의 순수 효과를 추정하지는 않는다. 출처: source_data/matched_window_BiTFI_comparison.json 및 온실별 CSV.

Figure 6은 기존에 높은 BiTFI 성능을 보고 선택한 구간을 그대로 유지했다. 예시를 재선택하지 않았고 축 범위를 유지했으며, 13개 masked 패널 × 4개 모델의 52개 R²/MAE 쌍을 sklearn으로 독립 재계산했다. 개별 패널의 최고 baseline은 MAE 기준이다.

## Suggested methods paragraph

{methods}

## 산출물과 재현

- 최종 PDF 6개 + Supplementary PDF 8개: `04_figure/` 및 `04_figure/Supplementary/`.
- 캡션·표·출처·검증: `04_figure/Figure_captions.md`, `source_data/`, `validation_report.json`.
- 전체 재평가: `03_result/reevaluation_clean_20260911/`.
- 이전 파일: `04_figure/archive/pre_leakage_fix_20260911/`.
- 비교 그림 갱신: `.venv/bin/python 02_model/figures/build_prism_figures.py` (활성 clean 결과 사용).
- 특정 모델 재현: 적절한 가상환경에서 `02_model/run_clean_evaluation.py --models <원시 모델 ID>` 실행. 기존 완료 결과는 재개 대상으로 처리하므로 새 추론을 원하면 별도 버전 출력 폴더로 프로토콜을 복제한다.
- 단계별 실행: `clean_protocol.py` → `train_saits.py --clean --out 03_result/reevaluation_clean_20260911/models/SAITS` → `BITFI_CLEAN=1 train_moment_head.py` → `run_clean_queue.py` → `run_clean_postqueue.py` → `validate_clean_protocol.py` → `finalize_clean_figures.py` → PDF 확인 → `promote_clean_results.py --reviewed`.

새 PDF에서는 값과 통계가 모두 clean 평가를 가리킨다. 기존 전체 원고 TeX의 서술·표를 전면 교정한 작업은 아니며, 원고 통합 시 이 결과 보고서와 새 캡션을 기준으로 반영해야 한다.
'''
 thesis=cp.ROOT/'05_thesis';archive=thesis/'archive/pre_leakage_fix_20260911';archive.mkdir(parents=True,exist_ok=True)
 for name in ['MOMENT_finetuning_results.md','Backbone_selection_note.md']:
  if (thesis/name).exists() and not (archive/name).exists():shutil.copy2(thesis/name,archive/name)
 audit_note=thesis/'BiTFI_leakage_audit_20260911.md'
 if '## Corrected reevaluation completed' not in audit_note.read_text():
  audit_note.write_text(audit_note.read_text()+'\n## Corrected reevaluation completed — 2026-09-11\n\nThe findings above document the pre-fix implementation. The identified input paths were subsequently corrected and all 23 configurations reevaluated under clean-20260911. Current results and limitations are in `Clean_reevaluation_results_20260911.md`; pre-fix figures remain archived. The revised checks do not certify pretraining membership, physical duplicate identity, or a newly untouched external test set.\n')
 status=cp.ROOT/'프로젝트_현황.md'
 if '## 2026-09-11 재평가 버전 전환' not in status.read_text():
  status.write_text('# 활성 결과: clean-20260911\n\n## 2026-09-11 재평가 버전 전환\n\n입력 누수 경로 수정 후 23개 설정 전체 재평가 및 PDF 갱신을 완료했다. 현재 수치와 해석은 `05_thesis/Clean_reevaluation_results_20260911.md` 및 `04_figure/Figure_captions.md`를 사용한다. 아래 2026-09-09 브리핑은 연구 진행 이력을 보존한 것으로, 그 수치·우월성·신규성 표현을 최신 결론으로 인용하지 않는다.\n\n---\n\n'+status.read_text())
 (thesis/'Clean_reevaluation_results_20260911.md').write_text(notes)
 (source/'Clean_reevaluation_results_20260911.md').write_text(notes)
 moment=f'''# MOMENT fine-tuning — clean reevaluation, 2026-09-11

Training/validation: {train['train_windows']}/{train['validation_windows']} windows from training greenhouses only, separated at each site's chronological 80/20 boundary. Scaling uses training prefixes only; natural missingness is retained. Encoder frozen; 8,200 head parameters trained. Three learning rates, 30-epoch cap, patience 5, seed 42. Selected lr={sel['learning_rate']}, epoch={sel['best_epoch']}. Validation MAE {tune['zero_shot_validation_mae']:.7f} → {tune['best_validation_mae']:.7f}.

Test NMAE: {mt['zero_shot_NMAE']:.7f} → {mt['head_tuned_NMAE']:.7f}; reduction {mt['relative_reduction_percent']:.2f}%, paired bootstrap CI [{mt['CI_low']:.2f}, {mt['CI_high']:.2f}]%; Wilcoxon p={mt['paired_wilcoxon_p']:.6f}; improved sites {mt['improved_greenhouses']}/10. Single training seed; intervals do not describe training-seed variation.

Figures 3–6 and S1/S5 display this fine-tuned model as MOMENT. Only S8 retains the explicit zero-shot control. Checkpoint: `03_result/reevaluation_clean_20260911/models/MOMENT/best.pt`. Full protocol and interpretation: `Clean_reevaluation_results_20260911.md`.
'''
 (thesis/'MOMENT_finetuning_results.md').write_text(moment)
 backbone=f'''# Backbone comparison — clean reevaluation, 2026-09-11

Identical 1,440 h past-only univariate inputs, raw-observation masks, one complete-gap forecast per call, and training-prefix-only scaling (S7A–C):

{fmt(uni[['label','NMAE','CI_low','CI_high']])}

Paired comparisons against TimesFM3, with Holm correction across the two tests:

{fmt(ut)}

Separate full BiTFI comparison (S7D): TimesFM3 NMAE {bb.TimesFM3_NMAE:.6f}, Chronos 2 {bb.Chronos2_NMAE:.6f}; relative reduction {bb.TFM3_reduction_percent:.2f}%; paired Wilcoxon p={bb.paired_wilcoxon_p:.6f}. This comparison includes retrospective context and covariates and must not be called univariate. Means alone do not establish superiority; interpret the paired tests and uncertainty. The primary backbone was chosen during exploratory development, not using a newly untouched test set. No full BiTFI/TimesFM2.5 configuration was tested.

Inputs/predictions/protocol: `03_result/reevaluation_clean_20260911/univariate_backbone_comparison/`. Full methods and limitations: `Clean_reevaluation_results_20260911.md`.
'''
 (thesis/'Backbone_selection_note.md').write_text(backbone)
 # Source tables are replaced as one version, rather than mixing old and corrected rows.
 shutil.rmtree(dest/'source_data');shutil.copytree(data,dest/'source_data')
 for path in source.iterdir():
  if path.name=='source_data':continue
  target=dest/path.name
  if path.is_dir():shutil.copytree(path,target,dirs_exist_ok=True)
  else:shutil.copy2(path,target)
 for pattern in ['*.png','*.tif','*.tiff','*.jpg','*.jpeg']:
  for path in list(dest.glob(pattern))+list((dest/'Supplementary').glob(pattern)):path.unlink()
 (dest/'PDF_ONLY.md').write_text('Publication figures are vector PDFs only. Current evaluation: clean-20260911. Six main figures and eight supplementary figures. Historical outputs are preserved under archive/.\n')
 active=dict(version='clean-20260911',result_root=str(cp.OUT.relative_to(cp.ROOT)),figures='04_figure',models=23,main_figures=6,supplementary_figures=8)
 (cp.ROOT/'03_result/active_evaluation.json').write_text(json.dumps(active,indent=2))
 # Include small, auditable protocol/equivalence records; leave datasets and model weights in the result tree.
 audit=dest/'source_data/clean_protocol';audit.mkdir(exist_ok=True)
 for name in ['protocol.json','protocol_validation.json','regression_checks.json','batch_equivalence.json','dispatcher_equivalence.json','spatial_dispatcher_equivalence.json','metric_equivalence.json','public_api_regression.json','worker_count_equivalence.json','rf_thread_equivalence.json','mask_manifest.csv','split.json','sites.csv']:
  shutil.copy2(cp.OUT/name,audit/name)
 for model in report['models']:
  inv=cp.OUT/'smoke'/model/'invariance.json';shutil.copy2(inv,audit/f'{model}_invariance.json')
 # Refresh the legacy summary entrypoint from the active clean results.
 import subprocess,sys
 old_table=cp.ROOT/'03_result/final_comparison.csv'
 if old_table.exists():shutil.copy2(old_table,dest/'archive/pre_leakage_fix_20260911/final_comparison.csv')
 with (cp.OUT/'final_table.log').open('w') as log:subprocess.run([sys.executable,str(cp.ROOT/'02_model/final_table.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
 (dest/'FIGURE_INDEX.md').write_text('# Active publication figures — clean-20260911\n\n'+''.join(f'- [{p.stem}]({p.relative_to(source)})\n' for p in sorted(pdfs))+ '\nOnly these 14 PDFs belong to the current submission package. Other legacy PDFs in this directory are not current performance evidence.\n')
 package=dest/'AIIA_BiTFI_figures.zip'
 with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED) as z:
  for path in pdfs:z.write(dest/path.relative_to(source),str(path.relative_to(source)))
  for path in [*[dest/p.relative_to(source) for p in source.glob('*.md')],*[dest/p.relative_to(source) for p in source.glob('*.json')],dest/'PDF_ONLY.md',dest/'FIGURE_INDEX.md',*(dest/'Supplementary').glob('*.md'),*(dest/'source_data').rglob('*')]:
   if path.is_file():z.write(path,str(path.relative_to(dest)))
 with zipfile.ZipFile(package) as z:
  assert sum(n.endswith('.pdf') for n in z.namelist())==14
  assert not any(n.lower().endswith(('.png','.tif','.tiff','.jpg','.jpeg')) for n in z.namelist())
 print(json.dumps(active,indent=2));print('PROMOTED',package,hashlib.sha256(package.read_bytes()).hexdigest())
if __name__=='__main__':main()
