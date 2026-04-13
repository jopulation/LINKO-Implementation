# LINKO Implementation (Linux-First Repro Guide)

이 저장소는 다음 연구 아이디어를 코드로 구현한 프로젝트입니다.

Multi-Ontology Integration with Dual-Axis Propagation for Medical Concept Representation

문서 목적:

- Linux 환경에서 바로 재현 가능한 실행 절차 제공
- 현재 워크스페이스에서 생성된 학습 결과 정리
- 첨부된 논문 표(Table 2, 3, 4)와 실제 결과를 항목별로 상세 비교

---

## 1) 저장소 구성

- model/LINKO.py: LINKO 핵심 모델
- train/train.py: 학습 엔트리 (5-fold 실험)
- utils/eval_test.py: AUPRC/ROC/F1/Acc@k/Hit@k 계산
- results_prompting/: 결과 요약 텍스트, JSON, 시각화
- output/OntoFAR_1.0/EXP_fold_*/: fold별 체크포인트와 로그

---

## 2) Linux 환경 준비

권장 OS: Ubuntu 22.04+

### 2.1 Python 가상환경

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2.2 GPU 확인 (선택)

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

참고:

- train/train.py는 기본값이 USE_GPU=1 입니다.
- CUDA가 없으면 USE_GPU=0 을 명시해야 합니다.

---

## 3) 데이터 배치

기본 경로:

datasets/MIMIC_III/

필수 테이블(코드상 사용):

- DIAGNOSES_ICD.csv
- PROCEDURES_ICD.csv
- PRESCRIPTIONS.csv
- PATIENTS.csv
- ADMISSIONS.csv

다른 경로를 쓰려면 실행 시 MIMIC3_ROOT 환경변수로 지정합니다.

---

## 4) 학습 실행 (Linux 기준)

### 4.1 스모크 테스트 (1 fold, 빠른 점검)

```bash
export PYTHONPATH=.
export MIMIC3_ROOT=./datasets/MIMIC_III
export MIMIC_DEV=1
export FOLDS=1
export SMOKE_FOLDS=1
export EPOCHS=3
export USE_GPU=0
python train/train.py
```

### 4.2 논문 재현에 가까운 실행 (5 fold)

```bash
export PYTHONPATH=.
export MIMIC3_ROOT=./datasets/MIMIC_III
export MIMIC_DEV=0
export FOLDS=5
export SMOKE_FOLDS=0
export EPOCHS=230
export USE_GPU=1
python train/train.py
```

### 4.3 학습 재개

```bash
export RESUME_TRAINING=1
python train/train.py
```

특정 체크포인트에서 시작하려면:

```bash
export RESUME_TRAINING=1
export RESUME_CKPT=./output/OntoFAR_1.0/EXP_fold_1/last.ckpt
python train/train.py
```

---

## 5) 결과 파일 위치

학습 완료 후 주로 확인할 파일:

- output/OntoFAR_1.0/EXP_fold_1/log.txt
- output/OntoFAR_1.0/EXP_fold_2/log.txt
- output/OntoFAR_1.0/EXP_fold_3/log.txt
- output/OntoFAR_1.0/EXP_fold_4/log.txt
- output/OntoFAR_1.0/EXP_fold_5/log.txt
- results_prompting/metrics_results_BestModel_OntoFAR_1.0.txt
- results_prompting/metrics_results_BestModel_OntoFAR_1.0_summary.json

시각화:

```bash
python utils/generate_readme_visuals.py \
  --input results_prompting/metrics_results_BestModel_OntoFAR_1.0.txt \
  --output-dir results_prompting
```

---

## 6) 이번 워크스페이스에서 확인된 학습 결과

기준 파일:

- results_prompting/metrics_results_BestModel_OntoFAR_1.0.txt

핵심 지표(평균, 95% CI):

| 지표 | 평균 | 95% CI |
|---|---:|---:|
| AUPRC | 27.49 | ±2.83 |
| ROC-AUC | 92.88 | ±0.16 |
| F1 | 24.23 | ±5.10 |
| Acc@20 | 36.99 | ±3.36 |
| Acc@30 | 42.15 | ±3.18 |

라벨 빈도 구간별 AUPRC:

| 라벨 빈도 구간 | AUPRC 평균 | 95% CI |
|---|---:|---:|
| 0-25% | 25.99 | ±2.75 |
| 25-50% | 59.97 | ±2.10 |
| 50-75% | 70.28 | ±4.19 |
| 75-100% | 43.10 | ±4.31 |

추가 관찰:

- k가 커질수록 Hit@k는 0.85 -> 0.98 수준으로 증가
- Acc@k는 k=3에서 높고 중간 k에서 낮아졌다가 k=30에서 다시 상승하는 패턴

---

## 7) 논문 대비 상세 비교 (Table 2, 3, 4 기반)

주의:

- 논문 표 수치는 퍼센트 스케일로 표기되어 있어, 본 문서도 동일 스케일로 비교했습니다.
- 아래 논문 값은 첨부된 표 이미지(Table 2, 3, 4)의 수치를 기준으로 정리했습니다.

### 7.1 General Performance 비교 (핵심)

| 지표 | 우리 실험 | 논문 MIMIC-III (LINKO w/ GAT) | 차이 (우리-논문, p) | 논문 MIMIC-IV (LINKO w/ GAT) | 차이 (우리-논문, p) |
|---|---:|---:|---:|---:|---:|
| AUPRC | 27.49 | 31.79 | -4.30 | 32.12 | -4.63 |
| F1 | 24.23 | 28.66 | -4.43 | 28.56 | -4.33 |
| Acc@20 | 36.99 | 41.84 | -4.85 | 43.56 | -6.57 |
| Acc@30 | 42.15 | 46.96 | -4.81 | 48.12 | -5.97 |

해석:

- 전체적으로 논문 대비 약 4~7p 낮은 성능입니다.
- 특히 Acc@20/30 갭이 AUPRC/F1보다 약간 더 큽니다.

### 7.2 Label Category AUPRC 비교

| 라벨 빈도 구간 | 우리 실험 AUPRC | 논문 MIMIC-III AUPRC (LINKO w/ GAT) | 차이 (우리-논문, p) |
|---|---:|---:|---:|
| 0-25% | 25.99 | 31.62 | -5.63 |
| 25-50% | 59.97 | 55.68 | +4.29 |
| 50-75% | 70.28 | 56.90 | +13.38 |
| 75-100% | 43.10 | 80.76 | -37.66 |

핵심 포인트:

- 희귀 구간(0-25%)과 최다빈도 구간(75-100%)에서 특히 격차가 큽니다.
- 중간 구간(25-75%)은 논문보다 높게 나왔습니다.
- 이는 라벨 분포/분할 방식 차이, 혹은 평가 스크립트의 그룹 정의 차이 영향일 가능성이 큽니다.

### 7.3 Prompting 전략 관점 (Table 3)

논문 Table 3에서 가장 강한 설정으로 보이는 type-code-concept-parent-task는 다음과 같습니다.

| 데이터셋 | AUPRC | F1 | Acc@20 | Acc@30 |
|---|---:|---:|---:|---:|
| MIMIC-III | 31.79 | 28.66 | 41.84 | 46.96 |
| MIMIC-IV | 32.38 | 30.02 | 43.70 | 48.63 |

우리 결과와 비교하면:

- MIMIC-III 기준으로도 약 4~5p 낮음
- MIMIC-IV 기준으로는 약 5~7p 낮음

즉, 프롬프트/코드-개념 결합 이득을 포함한 논문 상위 설정까지 감안해도 현재 결과는 추가 개선 여지가 큽니다.

### 7.4 Concept Type 조합 관점 (Table 4)

논문 Table 4는 dx/rx/px 조합 및 multi-level integration 유무를 비교합니다.

MIMIC-IV (논문 Table 4):

| Concept Type | 설정 | AUPRC | F1 | Acc@15 | Acc@20 | Acc@30 |
|---|---|---:|---:|---:|---:|---:|
| rx,px | w/ Multi-level | 22.74 | 16.92 | 31.56 | 33.57 | 38.75 |
| rx,px | w/o Multi-level | 21.35 | 15.15 | 30.03 | 32.21 | 37.23 |
| dx,px | w/ Multi-level | 30.67 | 25.87 | 40.25 | 42.21 | 46.99 |
| dx,px | w/o Multi-level | 29.28 | 22.49 | 38.62 | 40.68 | 45.56 |
| dx,rx | w/ Multi-level | 30.86 | 25.34 | 40.44 | 42.53 | 47.31 |
| dx,rx | w/o Multi-level | 30.17 | 25.08 | 39.77 | 41.71 | 46.59 |
| dx,rx,px | w/ Multi-level | 32.38 | 30.02 | 42.05 | 43.70 | 48.63 |
| dx,rx,px | w/o Multi-level | 30.10 | 25.26 | 39.67 | 41.50 | 46.25 |

MIMIC-III (논문 Table 4):

| Concept Type | 설정 | AUPRC | F1 | Acc@15 | Acc@20 | Acc@30 |
|---|---|---:|---:|---:|---:|---:|
| rx,px | w/ Multi-level | 24.24 | 18.29 | 31.92 | 34.43 | 39.87 |
| rx,px | w/o Multi-level | 22.98 | 16.55 | 30.81 | 32.95 | 38.61 |
| dx,px | w/ Multi-level | 30.19 | 26.40 | 38.03 | 40.18 | 46.03 |
| dx,px | w/o Multi-level | 29.01 | 24.48 | 37.04 | 39.16 | 44.89 |
| dx,rx | w/ Multi-level | 30.64 | 27.06 | 39.31 | 41.27 | 46.62 |
| dx,rx | w/o Multi-level | 29.56 | 24.44 | 37.88 | 39.98 | 45.46 |
| dx,rx,px | w/ Multi-level | 31.79 | 28.66 | 39.95 | 41.84 | 46.96 |
| dx,rx,px | w/o Multi-level | 29.41 | 25.28 | 37.55 | 39.61 | 45.25 |

우리 결과와 직접 비교 (dx,rx,px + Multi-level 기준):

| 지표 | 우리 실험 | 논문 MIMIC-III (w/ Multi-level, dx+rx+px) | 차이 (우리-논문, p) | 논문 MIMIC-IV (w/ Multi-level, dx+rx+px) | 차이 (우리-논문, p) |
|---|---:|---:|---:|---:|---:|
| AUPRC | 27.49 | 31.79 | -4.30 | 32.38 | -4.89 |
| F1 | 24.23 | 28.66 | -4.43 | 30.02 | -5.79 |
| Acc@20 | 36.99 | 41.84 | -4.85 | 43.70 | -6.71 |
| Acc@30 | 42.15 | 46.96 | -4.81 | 48.63 | -6.48 |

핵심 해석:

- 논문에서도 dx,rx,px를 모두 쓰고 multi-level을 적용한 경우가 각 데이터셋 최고 성능 구간입니다.
- 우리 실험도 동일 구조를 지향하지만, 해당 최고 구간 대비 약 4~7p 낮습니다.
- 따라서 현재 성능 갭은 모델 구조 자체보다 데이터셋 조건/평가 프로토콜/학습 안정화 차이의 영향이 더 클 가능성이 높습니다.

---

## 8) 왜 논문보다 낮게 나왔는가 (재현 관점 분석)

가능한 원인(우선순위 순):

1. 데이터셋 조건 불일치

- 논문 전처리 버전, 코드 매핑 버전, 환자 필터 조건이 다르면 AUPRC/F1이 크게 변동

2. 평가 분할 및 지표 계산 차이

- train/train.py는 환자 단위 5-fold를 직접 구성
- 라벨 그룹(0-25/25-50/50-75/75-100)은 누적 빈도 기반이라, 논문 그룹 기준과 다를 수 있음

3. LLM 임베딩 생성 조건 차이

- gpt_code_emb 생성 시 모델, 프롬프트, 실패 폴백 여부가 성능에 직접 영향

4. 하이퍼파라미터/학습 안정성

- 학습 로그상 epoch 진행 중 변동이 큼
- fold별 편차가 존재해 평균과 CI에 영향

5. 결과 요약 파일 불일치

- TXT와 JSON의 수치가 다릅니다.
- 본 문서는 fold 평균 및 CI가 포함된 TXT를 우선 신뢰 지표로 사용했습니다.

---

## 9) 재현 정확도 향상 체크리스트

논문 수치에 더 가깝게 가려면 아래를 순서대로 점검하세요.

1. 데이터셋 정합성

- 논문과 동일한 MIMIC 버전/필터/코드 매핑 사용 확인

2. 평가 파이프라인 정합성

- 검증/테스트 split과 metric 정의(특히 Acc@k, 그룹 AUPRC) 재검증

3. 임베딩 고정 및 캐시 재생성

- saved_files/gpt_code_emb 하위 임베딩을 논문 설정과 동일하게 재생성

4. 학습 설정 고정

- seed, EPOCHS, FOLDS, USE_GPU, batch size, lr를 실험표와 동일하게 통일

5. 결과 산출 스크립트 정리

- TXT/JSON 생성 루틴을 하나로 통합해 값 불일치 제거

---

## 10) 빠른 실행 요약

```bash
source .venv/bin/activate
export PYTHONPATH=.
export MIMIC3_ROOT=./datasets/MIMIC_III
export MIMIC_DEV=0
export FOLDS=5
export EPOCHS=230
export USE_GPU=1
python train/train.py
```

결과 확인:

- results_prompting/metrics_results_BestModel_OntoFAR_1.0.txt
- output/OntoFAR_1.0/EXP_fold_*/log.txt
