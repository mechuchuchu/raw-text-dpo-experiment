# Raw-text DPO 실험

`allenai/Dolci-Instruct-DPO`의 preference pair를 raw-text causal LM sequence로 직렬화하고, `allenai/OLMo-2-0425-1B`에 LoRA를 적용해 chosen/rejected likelihood 변화를 관찰하는 실험입니다.

현재 코드는 단일 턴 데이터 정규화, subset 생성, Base 평가, LoRA 학습, LoRA adapter 평가를 지원합니다. 멀티턴 데이터는 초기 실험에서 제외합니다.

## 1. 환경 준비

```bash
source /venv/main/bin/activate
cd /workspace/raw-text-dpo-experiment
export HF_HOME=/workspace/.hf_home
uv pip install -e ".[dev]"
```

GPU와 PyTorch를 확인합니다.

```bash
python - <<'PY'
import torch
print(torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name())
PY
```

## 2. 설정

기본 설정은 [configs/sanity.yaml](configs/sanity.yaml)입니다.

```text
sample_size: 500
eval_fraction: 0.1
max_seq_length: 1024
seed: 42
```

본실험용 설정은 복사해서 만듭니다.

```bash
cp configs/sanity.yaml configs/main.yaml
```

그 뒤 `configs/main.yaml`에서 다음처럼 조정합니다.

```yaml
dataset:
  sample_size: 2000

training:
  max_seq_length: 1536
```

전체 raw sequence는 길이가 긴 편입니다. `1024`는 초기 코드 검증용으로 사용하고, 본실험은 `1536` 또는 `2048`을 우선 검토합니다.

## 3. 데이터 구조 확인

```bash
python scripts/inspect_dataset.py \
  --config configs/sanity.yaml \
  --limit 1000
```

실제 데이터셋에는 별도 `prompt` 컬럼이 없습니다. 단일 턴 행은 다음처럼 정규화합니다.

```text
prompt   = chosen[0].content
chosen   = chosen[1].content
rejected = rejected[1].content
```

## 4. Subset 생성

유효한 단일 턴 pair를 seed에 따라 선택하고 prompt group이 겹치지 않도록 train/evaluation으로 분할합니다.

```bash
python scripts/build_subset.py \
  --config configs/sanity.yaml
```

기본 결과:

```text
outputs/subsets/rawtext_dpo_sanity_seed42_n500/
├── train.jsonl
├── eval.jsonl
└── manifest.json
```

`manifest.json`에는 dataset revision, seed, sample ID, 필터링 개수가 기록됩니다.

## 5. Base model 평가

먼저 한두 개 sample로 동작을 확인합니다.

```bash
python scripts/evaluate.py \
  --config configs/sanity.yaml \
  --subset outputs/subsets/rawtext_dpo_sanity_seed42_n500 \
  --model-variant base \
  --limit 1
```

전체 evaluation subset을 평가하려면 `--limit`을 생략합니다.

```bash
python scripts/evaluate.py \
  --config configs/sanity.yaml \
  --subset outputs/subsets/rawtext_dpo_sanity_seed42_n500 \
  --model-variant base
```

결과는 `outputs/results/base_evaluation.csv`에 저장됩니다.

평가 조건은 다음과 같습니다.

| arrangement | order | 의미 |
|---|---|---|
| aligned | forward | chosen/rejected 답변과 라벨이 일치하는 기본 조건 |
| aligned | reverse | 라벨은 같고 두 segment의 순서만 반대 |
| swapped | forward | 답변 내용을 반대 라벨에 배치 |
| swapped | reverse | swapped 상태에서 순서도 반대 |

`loss_difference`는 다음과 같습니다.

```text
loss_difference = chosen_loss - rejected_loss
```

음수이면 chosen 답변의 평균 token loss가 더 낮다는 뜻입니다.

## 6. LoRA 학습

생성한 train subset으로 LoRA 학습을 실행합니다.

```bash
python scripts/train_lora.py \
  --config configs/sanity.yaml \
  --subset outputs/subsets/rawtext_dpo_sanity_seed42_n500
```

adapter는 다음 위치에 저장됩니다.

```text
outputs/checkpoints/rawtext_dpo_sanity/final/
├── adapter_config.json
├── adapter_model.safetensors
└── tokenizer files
```

RTX 3060에서 메모리가 부족하면 먼저 sequence length를 줄이고 gradient accumulation을 늘립니다.

```yaml
training:
  max_seq_length: 1024
  gradient_accumulation_steps: 16
```

## 7. LoRA model 평가

LoRA adapter를 Base model 위에 올려 평가합니다.

```bash
python scripts/evaluate.py \
  --config configs/sanity.yaml \
  --subset outputs/subsets/rawtext_dpo_sanity_seed42_n500 \
  --model outputs/checkpoints/rawtext_dpo_sanity/final \
  --base-model allenai/OLMo-2-0425-1B \
  --model-variant lora
```

결과는 `outputs/results/lora_evaluation.csv`에 저장됩니다. `--model-variant lora`일 때 `--model`은 전체 모델이 아니라 LoRA adapter directory입니다.

## 8. 결과 해석

```bash
python - <<'PY'
import pandas as pd

for path in [
    "outputs/results/base_evaluation.csv",
    "outputs/results/lora_evaluation.csv",
]:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        continue
    print("\\n", path)
    print(df.groupby(["arrangement", "order", "scoring_mode"])["loss_difference"].agg(["count", "mean", "median"]))
PY
```

주요 비교 항목은 다음과 같습니다.

1. Base와 LoRA 사이의 `loss_difference` 변화
2. aligned와 swapped 사이의 변화
3. forward와 reverse 사이의 변화
4. `full_sequence`와 `isolated_branch` 사이의 차이

`full_sequence`에서는 두 번째 답변이 첫 번째 답변까지 문맥으로 사용됩니다. `isolated_branch`는 답변 하나만 prompt와 라벨 뒤에 놓으므로 라벨 효과를 해석할 때 보조 기준으로 사용합니다.

## 9. 테스트와 검증

```bash
python -m pytest -q
python -m compileall -q src scripts
```

평가 출력의 `written`은 저장된 조건 수이고, `skipped`는 `max_seq_length`를 초과해 제외된 조건 수입니다. 현재 구현은 긴 sequence를 자동으로 잘라 loss를 왜곡하지 않고 해당 조건을 제외합니다.

## 10. 디렉터리와 저장 정책

```text
configs/       실험 설정
src/           재사용 가능한 Python 모듈
scripts/       명령행 실행 스크립트
tests/         단위 테스트
outputs/       subset, checkpoint, 결과 CSV, 로그
```

대용량 dataset, model checkpoint, tokenized cache와 실행 결과는 Git에 저장하지 않습니다. 재현에 필요한 설정, dataset revision, seed, sample ID는 설정 파일과 `manifest.json`에 남깁니다.

현재 초기 구현의 제한사항은 다음과 같습니다.

- 멀티턴 데이터는 제외합니다.
- `max_seq_length`를 넘는 평가 조건은 제외합니다.
- 학습은 전체 raw sequence에 일반 causal LM loss를 적용합니다.
- 평가는 답변별 token-level mean negative log-likelihood를 계산합니다.
