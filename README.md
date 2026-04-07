# LINKO

Implementation of:

Multi-Ontology Integration with Dual-Axis Propagation for Medical Concept Representation

This repository is configured to run with local Ollama (`llama3.1`) and MIMIC-III demo/full data.

## 1) Project Structure

Main source tree:

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

## 2) Run On Another Virtual Environment

### Step A. Create and activate a new venv

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Step B. Install dependencies

Install PyTorch first (CPU build example):

```powershell
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

Then install project requirements:

```powershell
pip install -r requirements.txt
```

### Step C. Run Ollama locally

In another terminal:

```powershell
ollama serve
ollama run llama3.1
```

### Step D. Prepare MIMIC-III data path

Put raw MIMIC-III files under:

```text
datasets/MIMIC_III/
```

The training script expects `DIAGNOSES_ICD`, `PROCEDURES_ICD`, `PRESCRIPTIONS` tables from there.

## 3) Training Commands

Set project import path:

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

### Full training (paper-style long run)

```powershell
$env:MIMIC_DEV='0'
$env:EPOCHS='230'
Remove-Item Env:SMOKE_SEEDS -ErrorAction SilentlyContinue
python train/train.py
```

Notes:

- Default epochs are 230 if `EPOCHS` is not set.
- Current script uses 5 seeds by default.
- `LINKO_SKIP_OLLAMA=1` forces deterministic local vectorization (debug/smoke helper).

## 4) What Data Is Committed For Reproducibility

Committed lightweight caches for quick startup on a fresh venv:

- `saved_files/mimic3_samples/samples_1.0.pkl`
- `saved_files/mimic3_samples/samples_ccs_1.0.pkl`
- `saved_files/conditional_prob_matrix.csv`
- `saved_files/conditional_prob_matrix1.csv`
- `saved_files/conditional_prob_matrix2.csv`
- `saved_files/gpt_code_emb/tx-emb-3-small/include_all_parents2/*.npy`

Not committed:

- Full raw MIMIC datasets (`datasets/MIMIC_III/`)
- Large/generated outputs under `output/`, `results_prompting/`

## 5) Checkpoint Behavior

Training saves:

- `last.ckpt` at every epoch (overwritten each epoch)
- `best.ckpt` when monitored metric improves

Output directory is under:

```text
output/OntoFAR_1.0/EXP_seed_<seed>/
```

## 6) 학습 결과 정리와 시각화

학습이 끝나면 아래 파일에서 최종 결과를 확인할 수 있습니다.

- `results_prompting/metrics_results_BestModel_OntoFAR_1.0.txt`
- `results_prompting/metrics_results_BestModel_OntoFAR_1.0_summary.json`
- `results_prompting/metrics_results_BestModel_OntoFAR_1.0_summary.png`

시각화 자료는 아래처럼 정리했습니다.

- 왼쪽 그래프: `pr_auc_samples`, `roc_auc_samples`, `f1_samples` 평균값을 비교한 막대 그래프
- 오른쪽 그래프: `acc_at_k`, `hit_at_k`를 `k`별로 비교한 선 그래프

한국어로 정리하면 다음과 같습니다.

- 이 실험은 MIMIC-III 기반의 다중 라벨 진단 예측 모델 LINKO를 학습한 결과입니다.
- 학습 완료 후 각 seed별 체크포인트는 `output/OntoFAR_1.0/EXP_seed_<seed>/` 아래에 저장됩니다.
- 최종 요약 지표는 `results_prompting/` 폴더에 텍스트, JSON, PNG 형태로 함께 저장됩니다.
- 위 PNG 파일은 학습이 잘 되었는지 한눈에 보기 위한 요약 시각화 자료입니다.

## 7) How To Stop Training Mid-run

### In terminal foreground

- Press `Ctrl + C`

### If running in background process (PowerShell/VS Code terminal)

- Use terminal stop button in VS Code, or:

```powershell
Get-Process python | Stop-Process
```

Safe stop recommendation:

- Wait until current epoch checkpoint (`last.ckpt`) is written, then stop.
