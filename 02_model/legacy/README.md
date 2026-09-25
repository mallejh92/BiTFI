# legacy/

교차-온실 분할(`data_split.py`) 도입 이전, 온실별 80/20 시간순 분할 기반의
옛 실험 프레임워크. 현재 활성 파이프라인은 `02_model/run_comparison.py`
(9개 모델 개별 성능 비교)이며, 이 폴더는 참고용으로 보관한다.

| 파일 | 설명 | 대체됨 |
|------|------|--------|
| `run_experiment.py` | 7온실 80/20 block masking 실험 | `run_comparison.py` |
| `run_experiment_v2.py` | 80/20 + Scenario A/B/C | `run_comparison.py` (교차-온실 버전) |
| `run_context_ablation.py` | context length 24~672h 스윕 | (별도 목적, 9모델 비교와 무관 — 필요시 재사용) |
| `pareto_analysis.py` | context_ablation 결과의 파레토 프런트 분석 | 위 ablation 실행 후 사용 |
| `visualize.py` | 옛 결과 포맷용 시각화 | `figures/fig_comparison.py` |
| `masking.py` | 마스킹 v1(individual/block) | `masking_v2.py` (Scenario A/B/C) |
| `figures/fig1_data_overview.py` | 데이터 개요 | `figures/fig_comparison.py` |
| `figures/fig2_imputation_results.py` | 보간 결과 | 〃 |
| `figures/fig3_context_ablation.py` | context ablation 결과 | 〃 |
| `figures/fig3_timeseries.py` | 시계열 비교 | 〃 |
| `figures/fig4_scatter.py` | scatter plot | 〃 |
| `figures/fig5_metadata.py` | 메타데이터 분석 | 〃 |

`models/patchtst_model.py`(자체 구현 PatchTST)는 위 `run_experiment_v2.py`와
`figures/fig3_timeseries.py`·`fig4_scatter.py`에서만 쓰여 사실상 레거시지만,
`models/` 패키지를 둘로 쪼개지 않기 위해 `02_model/models/`에 그대로 둔다
(현재 `run_comparison.py`는 AutoGluon 내장 PatchTST를 사용).

## 실행 방법
이동 시 `sys.path`를 `02_model/`(부모)까지 포함하도록 경로 설정만 보정했고,
다른 로직은 변경하지 않았다. 기존과 동일하게 실행 가능:
```bash
cd 02_model/legacy
python run_experiment_v2.py --help
```

## 완전히 삭제된 파일 (참고)
`models/lstm_model.py`, `models/unet_model.py` — 어디서도 import되지 않는
완전 미사용 코드라 보관 없이 삭제함(git 히스토리에서 복구 가능).
