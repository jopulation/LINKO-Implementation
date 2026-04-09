# LINKO

다중 온톨로지 통합과 이중 축 전파를 활용한 의료 개념 표현

이 레포지토리는 로컬 Ollama(`llama3.1`)와 MIMIC-III 데모 데이터 기준으로 실행되도록 구성되어 있습니다.

## 1) 프로젝트 구조

주요 소스 트리는 아래와 같습니다.

```text
model/
	LINKO.py                     # model definition
train/
	train.py                     # training entrypoint
tasks/
	diagnosis_prediction.py      # MIMIC-III task builder
utils/
	data.py                      # dataset preprocessing / caching
	eval_test.py                 # evaluation metrics
	splitter.py                  # split helpers
saved_files/
	icd_maping/                  # ontology mapping files
	ontology_tables/             # condition/drug/procedure ontology tables
	mimic3_samples/              # preprocessed sample cache
	gpt_code_emb/                # cached LLM/code embeddings
	conditional_prob_matrix*.csv # co-occurrence matrices
datasets/
	MIMIC_III/                   # raw MIMIC-III csv files (not committed)
```

## 2) 다른 가상환경에서 실행

### A. 새 venv 생성 및 활성화

Windows PowerShell :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### B. 의존성 설치

먼저 PyTorch를 설치합니다(CPU 예시).

```powershell
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

그다음 프로젝트 의존성을 설치합니다.

```powershell
pip install -r requirements.txt
```

### C. 로컬 Ollama 실행

다른 터미널에서 아래를 실행합니다.

```powershell
ollama serve
ollama run llama3.1
```

### D. MIMIC-III 데이터 경로 준비

원본 MIMIC-III 파일은 아래 경로에 배치합니다.

```text
datasets/MIMIC_III/
```

학습 스크립트는 `DIAGNOSES_ICD`, `PROCEDURES_ICD`, `PRESCRIPTIONS` 테이블을 사용합니다.

## 3) 학습 명령어

프로젝트 import 경로를 먼저 설정합니다.

```powershell
$env:PYTHONPATH='.'
```

### Smoke test (fast verification)

```powershell
$env:MIMIC_DEV='1'
$env:EPOCHS='1'
$env:SMOKE_SEEDS='1'
python train/train.py
```

### 전체 학습 실행

```powershell
$env:MIMIC_DEV='0'
$env:EPOCHS='230'
Remove-Item Env:SMOKE_SEEDS -ErrorAction SilentlyContinue
python train/train.py
```

참고:

- `EPOCHS`를 설정하지 않으면 기본값은 `230`입니다.
- 기본 설정에서는 seed 5개를 사용합니다.
- `LINKO_SKIP_OLLAMA=1`을 설정하면 로컬 결정론 벡터화로 동작합니다.

## 4) 재현을 위해 커밋된 데이터

새 가상환경에서도 빠르게 시작할 수 있도록 아래 경량 캐시를 포함했습니다.

- `saved_files/mimic3_samples/samples_1.0.pkl`
- `saved_files/mimic3_samples/samples_ccs_1.0.pkl`
- `saved_files/conditional_prob_matrix.csv`
- `saved_files/conditional_prob_matrix1.csv`
- `saved_files/conditional_prob_matrix2.csv`
- `saved_files/gpt_code_emb/tx-emb-3-small/include_all_parents2/*.npy`

아래 데이터는 커밋하지 않습니다.

- Full raw MIMIC datasets (`datasets/MIMIC_III/`)
- Large/generated outputs under `output/`, `results_prompting/`

## 5) 체크포인트 저장 방식

학습 중 다음 파일이 저장됩니다.

- `last.ckpt`: 매 epoch마다 갱신
- `best.ckpt`: 모니터링 지표가 개선될 때 갱신

저장 경로는 아래와 같습니다.

```text
output/OntoFAR_1.0/EXP_seed_<seed>/
```

## 6) 학습 결과 정리와 시각화

학습이 끝나면 아래 파일에서 결과를 확인할 수 있습니다.

- `results_prompting/metrics_results_BestModel_OntoFAR_1.0.txt`
- `results_prompting/metrics_results_BestModel_OntoFAR_1.0_summary.json`
- `results_prompting/metrics_results_BestModel_OntoFAR_1.0_summary.png`

요약 시각화는 다음과 같이 구성됩니다.

![results_prompting/summary](results_prompting/metrics_results_BestModel_OntoFAR_1.0_summary.png)
- 왼쪽 그래프: `pr_auc_samples`, `roc_auc_samples`, `f1_samples` 평균 비교
- 오른쪽 그래프: `acc_at_k`, `hit_at_k`의 `k`별 비교

정리:

- 이 실험은 MIMIC-III 기반 다중 라벨 진단 예측 모델 학습 결과입니다.
- seed별 체크포인트는 `output/OntoFAR_1.0/EXP_seed_<seed>/` 아래에 저장됩니다.
- 최종 요약 지표는 `results_prompting/` 아래에 텍스트, JSON, PNG로 저장됩니다.

## 7) 학습 중단 방법

### 포그라운드 실행 중

- `Ctrl + C`를 입력합니다.

### 백그라운드 실행 중 (PowerShell/VS Code 터미널)

- VS Code 터미널의 중지 버튼을 사용하거나 아래를 실행합니다.

```powershell
Get-Process python | Stop-Process
```

권장:

- 현재 epoch의 `last.ckpt` 저장이 끝난 뒤 중단하는 것이 안전합니다.

## 8) 코드 읽는 순서 (모델 + 데이터 중심)

이 프로젝트는 모델 축과 데이터 축으로 나눠서 보면 이해가 가장 빠릅니다.

### 8.1 모델 축: `model/LINKO.py`

핵심 클래스는 `Mega`이며, 실제 학습에서 생성되는 모델 인스턴스입니다.

#### A. `__init__`에서 하는 일

1. Ontology 테이블 구성
- `conditions`, `drugs`, `procedures`를 `l1`, `l2`, `l3`로 구성합니다.
- 레벨별 토크나이저와 임베딩을 초기화합니다.

2. co-occurrence 그래프 준비
- 저장된 확률 행렬이 있으면 로드합니다.
- 없으면 학습 데이터에서 계산하여 생성합니다.

3. LLM 임베딩 준비
- `*.npy` 캐시가 있으면 로드합니다.
- 없으면 코드 설명 문장을 만들고 LLM 임베딩을 생성합니다.
- 실패 시 로컬 결정론 벡터화로 대체합니다.

4. 모델 모듈 초기화
- Ontology GAT, Hypergraph, bottom-up HAP, Transformer, 최종 FC를 초기화합니다.

#### B. `forward`에서 하는 일

1. 각 feature(`conditions`, `drugs`, `procedures`)를 토큰화
2. `Onto_GAT`으로 ontology 전파 수행
3. `bottom_up_hap`으로 부모-자식 정보 결합
4. `_gram`으로 레벨 임베딩 합성
5. 환자 방문 시퀀스 임베딩 lookup 및 visit 축 합산
6. Transformer 인코딩
7. feature 임베딩 concat 후 FC로 멀티라벨 로짓 생성
8. `loss`, `y_prob`, `y_true` 반환

### 8.2 데이터 축: `utils/data.py`

`customized_set_task_mimic3` 경로에서 MIMIC 샘플을 task 포맷으로 변환하고 캐시합니다.

1. raw patient/visit를 task 함수로 변환
2. 변환 결과를 `saved_files/mimic3_samples/`에 pickle 캐시
3. 이후 실행에서는 캐시를 재사용해 전처리 시간을 단축

추가로 `ICD10toICD9` 매핑, 다중 레벨 데이터셋 래퍼, 사용자 정의 collate 함수가 포함되어 있어 ontology 레벨 학습 입력을 맞출 수 있습니다.

## 9) LLM 임베딩 생성 로직

`LINKO.py`의 임베딩 생성은 다음 순서로 동작합니다.

1. 강화된 프롬프트 구성
- 의료 의미 보존
- 계층 정보 유지
- 간결하고 사실 기반 문장 생성

2. Ollama 임베딩 엔드포인트 우선 호출
- `/api/embed` 우선
- 실패 시 `/api/embeddings` 재시도

3. 다단계 폴백
- 임베딩 API 실패 시 `/api/generate` 결과를 로컬 벡터화
- generation도 실패 시 프롬프트 텍스트를 로컬 결정론 벡터화

## 10) 시각화 재생성 방법

시각화를 다시 만들 때는 아래를 실행합니다.

```powershell
python utils/visualize_results.py --input results_prompting/metrics_results_BestModel_OntoFAR_1.0.txt --output-dir results_prompting
```
